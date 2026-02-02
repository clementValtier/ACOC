"""
ACOC - Trainer (PyTorch)
========================
Orchestrateur de la boucle Training → Checkpoint → Décision → Expansion.
Avec warmup après expansion et exploration forcée des nouveaux blocs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, List, Dict, Optional, Callable

from ..config import SystemConfig, TrainingLog, ExpansionDecision
from ..model import ACOCModel


class ACOCTrainer:
    """
    Orchestrateur de la boucle d'entraînement ACOC.
    
    Cycle complet:
    1. TRAINING: Architecture fixe, backprop normal
    2. CHECKPOINT: Évaluation + vote des variantes (seuil relatif)
    3. DÉCISION: Analyser métriques de saturation
    4. EXPANSION: Modifier architecture si nécessaire
    5. WARMUP: LR élevé + exploration forcée pour nouveaux blocs
    6. MAINTENANCE: Pruning/consolidation périodique
    """
    
    def __init__(
        self, 
        model: ACOCModel, 
        config: SystemConfig,
        learning_rate: float = 0.001
    ):
        self.model = model
        self.config = config
        self.learning_rate = learning_rate
        self.training_logs: List[TrainingLog] = []
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Callbacks optionnels
        self.on_cycle_start: Optional[Callable[[int], None]] = None
        self.on_cycle_end: Optional[Callable[[int, TrainingLog], None]] = None
        
        # État du warmup
        self._in_warmup = False
        self._warmup_steps_remaining = 0
        self._warmup_target_block: Optional[str] = None
    
    def _rebuild_optimizer(self, lr_multipliers: Dict[str, float] = None):
        """
        Reconstruit l'optimizer avec des LR différenciés.
        Utilisé après expansion pour donner un LR plus élevé aux nouveaux params.
        """
        if lr_multipliers is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            return
        
        param_groups = []
        for name, param in self.model.named_parameters():
            multiplier = lr_multipliers.get(name, 1.0)
            param_groups.append({
                'params': [param],
                'lr': self.learning_rate * multiplier
            })
        
        self.optimizer = optim.Adam(param_groups)
    
    def training_phase(
        self, 
        data_loader: DataLoader = None,
        num_steps: int = 100,
        verbose: bool = True
    ) -> float:
        """
        Phase de training avec architecture FIXE.
        """
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === PHASE TRAINING ===")
        
        self.model.train()
        device = self.model.device
        losses = []
        
        # Gérer le warmup
        warmup_blocks = self.model.warmup_manager.get_warmup_blocks()
        if warmup_blocks:
            # Activer l'exploration forcée
            target_block = warmup_blocks[0]
            self.model.set_exploration(
                target_block,
                self.config.new_block_exploration_prob
            )
            if verbose:
                print(f"  [Warmup actif] Exploration forcée vers '{target_block}' "
                      f"(prob={self.config.new_block_exploration_prob:.0%})")
        else:
            self.model.set_exploration(None, 0.0)
        
        step = 0
        while step < num_steps:
            # Obtenir un batch
            if data_loader is not None:
                for batch in data_loader:
                    if step >= num_steps:
                        break
                    
                    if isinstance(batch, (list, tuple)):
                        batch_x, batch_y = batch[0].to(device), batch[1].to(device)
                    else:
                        batch_x = batch.to(device)
                        batch_y = torch.randn_like(batch_x)
                    
                    loss = self._training_step(batch_x, batch_y)
                    losses.append(loss)
                    step += 1
                    
                    # Progress
                    if verbose and step % max(1, num_steps // 5) == 0:
                        print(f"  Step {step}: loss = {loss:.4f}")
            else:
                # Données simulées
                batch_x = torch.randn(32, self.config.input_dim, device=device)
                batch_y = torch.randn(32, self.config.output_dim, device=device)
                
                loss = self._training_step(batch_x, batch_y)
                losses.append(loss)
                step += 1
                
                if verbose and step % max(1, num_steps // 5) == 0:
                    print(f"  Step {step}: loss = {loss:.4f}")
        
        # Mettre à jour le warmup
        self.model.warmup_manager.step()
        self.model.warmup_manager.check_and_cleanup(current_cycle=self.model.current_cycle)
        
        # Enregistrer la loss moyenne
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        self.model.metrics.add_loss(avg_loss)
        
        if verbose:
            print(f"  Avg loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _training_step(
        self, 
        batch_x: torch.Tensor, 
        batch_y: torch.Tensor
    ) -> float:
        """Effectue un step de training."""
        self.optimizer.zero_grad()
        
        # Forward
        outputs, routing_stats = self.model(batch_x)
        
        # Loss
        loss = self.model.compute_loss(outputs, batch_y)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update
        self.optimizer.step()
        
        return loss.item()
    
    def checkpoint_phase(
        self, 
        validation_data: DataLoader = None,
        verbose: bool = True
    ) -> tuple:
        """
        Phase de checkpoint: évaluation + vote des variantes avec seuil RELATIF.
        """
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === PHASE CHECKPOINT ===")
        
        self.model.eval()
        
        # Collecter les métriques de saturation
        metrics = self.model.collect_metrics()
        
        # Initialiser les deltas des variantes
        self.model.variant_system.initialize_deltas(self.model)
        
        # Fonction d'évaluation
        def evaluate_fn(model: nn.Module) -> float:
            """Évalue le modèle sur les données de validation."""
            model.eval()
            device = self.model.device
            
            if validation_data is not None:
                total_loss = 0.0
                count = 0
                with torch.no_grad():
                    for batch in validation_data:
                        if isinstance(batch, (list, tuple)):
                            val_x, val_y = batch[0].to(device), batch[1].to(device)
                        else:
                            val_x = batch.to(device)
                            val_y = torch.randn_like(val_x)
                        
                        outputs, _ = model(val_x)
                        # Utiliser la même loss que le training
                        if self.config.use_cross_entropy:
                            if val_y.dim() == 2:  # Si one-hot, convertir
                                val_y = val_y.argmax(dim=1)
                            loss = nn.functional.cross_entropy(outputs, val_y)
                        else:
                            loss = nn.functional.mse_loss(outputs, val_y)
                        total_loss += loss.item() * val_x.size(0)
                        count += val_x.size(0)
                        
                        if count >= 500:  # Limiter l'évaluation
                            break
                
                avg_loss = total_loss / max(count, 1)
            else:
                # Données simulées
                with torch.no_grad():
                    val_x = torch.randn(100, self.config.input_dim, device=device)
                    if self.config.use_cross_entropy:
                        # Pour CrossEntropy, générer des indices aléatoires
                        val_y = torch.randint(0, self.config.output_dim, (100,), device=device)
                        outputs, _ = model(val_x)
                        avg_loss = nn.functional.cross_entropy(outputs, val_y).item()
                    else:
                        val_y = torch.randn(100, self.config.output_dim, device=device)
                        outputs, _ = model(val_x)
                        avg_loss = nn.functional.mse_loss(outputs, val_y).item()
            
            # Convertir la loss en score (inverse)
            score = 1.0 / (1.0 + avg_loss)
            return score
        
        # Vote sur l'expansion avec seuil RELATIF
        should_expand, confidence, reason = self.model.variant_system.vote_on_expansion(
            self.model, evaluate_fn, metrics
        )
        
        # Ajouter le score à l'historique des métriques
        current_score = evaluate_fn(self.model)
        metrics.add_validation_score(current_score)
        
        if verbose:
            # Afficher les métriques de saturation détaillées
            print(f"  Saturation détaillée:")
            for block_id, sat in metrics.detailed_saturation.items():
                print(f"    {block_id}: score={sat.combined_score:.2f} "
                      f"(grad_flow={sat.gradient_flow_ratio:.2f}, "
                      f"act_sat={sat.activation_saturation:.2f}, "
                      f"dead={sat.dead_neuron_ratio:.2f})")
            
            print(f"  Utilization: {metrics.expert_utilization}")
            print(f"  Vote: expand={should_expand}, conf={confidence:.2f}")
            print(f"  {reason}")
        
        # Merger les meilleurs deltas
        self.model.variant_system.merge_best_deltas(self.model, evaluate_fn)
        
        # Faire évoluer les deltas
        scored = self.model.variant_system.evaluate_variants(self.model, evaluate_fn)
        self.model.variant_system.evolve_deltas(self.model, scored)
        
        self.model.train()
        
        return should_expand, confidence, reason
    
    def decision_phase(
        self,
        variant_vote: bool = False,
        variant_confidence: float = 0.0,
        verbose: bool = True
    ) -> ExpansionDecision:
        """
        Phase de décision basée sur les métriques de saturation.
        """
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === PHASE DÉCISION ===")
        
        # Décision basée sur les métriques
        decision = self.model.evaluate_expansion()
        
        # Combiner avec le vote des variantes
        if variant_vote and not decision.should_expand and variant_confidence > 0.7:
            # Le vote suggère d'expand avec forte confiance
            decision.should_expand = True
            decision.expansion_type = "width"
            decision.confidence = variant_confidence
            decision.reason += f" + Vote variantes fort ({variant_confidence:.0%})"
        
        if verbose:
            print(f"  Decision: expand={decision.should_expand}")
            print(f"  Type: {decision.expansion_type}")
            print(f"  Target: {decision.target_block_id}")
            print(f"  Confidence: {decision.confidence:.2f}")
            print(f"  Reason: {decision.reason}")
        
        return decision
    
    def expansion_phase(
        self, 
        decision: ExpansionDecision,
        verbose: bool = True
    ) -> bool:
        """
        Phase d'expansion avec démarrage du warmup.
        """
        if not decision.should_expand:
            if verbose:
                print(f"\n[Cycle {self.model.current_cycle}] === PAS D'EXPANSION ===")
            return False
        
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === PHASE EXPANSION ===")
        
        # Sauvegarder Fisher info du routeur
        # (simplifié: on le fait avec des données simulées)
        self.model.router.compute_fisher(
            self._create_dummy_dataloader()
        )
        
        # Exécuter l'expansion
        success = self.model.execute_expansion(decision)
        
        if success:
            if verbose:
                print(f"  ✓ Expansion réussie: {decision.expansion_type}")
                if decision.target_block_id:
                    print(f"    Target: {decision.target_block_id}")
                print(f"  Nouvelle taille: {self.model.get_total_params():,} params")
            
            # Reconstruire l'optimizer
            self._rebuild_optimizer()
            
            # Ajuster les pénalités si nécessaire
            adjusted = self.model.penalty_manager.adjust_thresholds(self.model.metrics)
            if adjusted and verbose:
                print(f"  Pénalités ajustées")
            
            if verbose and decision.expansion_type == "new_block":
                print(f"  [Warmup démarré] {self.config.warmup_steps} steps, "
                      f"exploration={self.config.new_block_exploration_prob:.0%}")
        
        return success
    
    def _create_dummy_dataloader(self, num_samples: int = 100):
        """Crée un dataloader avec des données simulées."""
        x = torch.randn(num_samples, self.config.input_dim)
        y = torch.randn(num_samples, self.config.output_dim)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=32)
    
    def maintenance_phase(self, verbose: bool = True) -> Dict:
        """Phase de maintenance: pruning et consolidation."""
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === PHASE MAINTENANCE ===")
        
        actions = self.model.run_maintenance()
        
        if verbose:
            if actions["pruned"]:
                print(f"  Pruned: {actions['pruned']}")
            if actions["consolidated"]:
                print(f"  Consolidated: {actions['consolidated']}")
            if not actions["pruned"] and not actions["consolidated"]:
                print(f"  Aucune action nécessaire")
        
        # Reconstruire l'optimizer si des changements ont eu lieu
        if actions["pruned"] or actions["consolidated"]:
            self._rebuild_optimizer()
        
        return actions
    
    def run_cycle(
        self,
        data_loader: DataLoader = None,
        validation_data: DataLoader = None,
        num_steps: int = 50,
        verbose: bool = True
    ) -> TrainingLog:
        """Exécute un cycle complet."""
        if self.on_cycle_start:
            self.on_cycle_start(self.model.current_cycle)
        
        # 1. Training
        avg_loss = self.training_phase(data_loader, num_steps, verbose=verbose)
        
        # 2. Checkpoint avec seuil relatif
        variant_vote, confidence, _ = self.checkpoint_phase(validation_data, verbose=verbose)
        
        # 3. Décision
        decision = self.decision_phase(variant_vote, confidence, verbose=verbose)
        
        # 4. Expansion (avec warmup automatique)
        expanded = self.expansion_phase(decision, verbose=verbose)
        
        # 5. Maintenance (périodique)
        if self.model.current_cycle % self.config.maintenance_interval == 0:
            self.maintenance_phase(verbose=verbose)
        
        # Créer le log
        sat_details = {
            block_id: sat.combined_score 
            for block_id, sat in self.model.metrics.detailed_saturation.items()
        }
        
        log = TrainingLog(
            cycle=self.model.current_cycle,
            avg_loss=avg_loss,
            total_params=self.model.get_total_params(),
            num_blocks=len(self.model.task_blocks),
            expanded=expanded,
            expansion_type=decision.expansion_type if expanded else None,
            expansion_target=decision.target_block_id if expanded else None,
            warmup_active=self.model.warmup_manager.is_warmup_active(),
            saturation_details=sat_details
        )
        self.training_logs.append(log)

        # Sauvegarder l'utilisation récente avant reset
        self.model.expansion_manager.update_recent_usage(
            self.model.task_blocks,
            self.model.current_cycle
        )

        # Reset pour le prochain cycle
        self.model.reset_usage_counts()
        
        if self.on_cycle_end:
            self.on_cycle_end(self.model.current_cycle, log)
        
        # Incrémenter
        self.model.current_cycle += 1
        
        return log
    
    def run(
        self,
        num_cycles: int = 10,
        data_loader: DataLoader = None,
        validation_data: DataLoader = None,
        num_steps_per_cycle: int = 50,
        verbose: bool = True
    ):
        """Exécute plusieurs cycles."""
        if verbose:
            print("=" * 60)
            print("ACOC Training Start")
            print(f"Device: {self.model.device}")
            print(f"Initial params: {self.model.get_total_params():,}")
            print(f"Initial blocks: {len(self.model.task_blocks)}")
            print("=" * 60)
        
        for _ in range(num_cycles):
            self.run_cycle(
                data_loader=data_loader,
                validation_data=validation_data,
                num_steps=num_steps_per_cycle,
                verbose=verbose
            )
        
        if verbose:
            print("\n" + "=" * 60)
            print("ACOC Training Complete")
            print("=" * 60)
            self.print_summary()
    
    def print_summary(self):
        """Affiche un résumé de l'entraînement."""
        print(self.model.summary())
        
        # Vote summary
        vote_summary = self.model.variant_system.get_vote_summary()
        print(f"\nVote Summary:")
        print(f"  Total votes: {vote_summary['total']}")
        print(f"  Expand votes: {vote_summary['expand_votes']}")
        print(f"  Avg confidence: {vote_summary['avg_confidence']:.2f}")
        print(f"  Current threshold: {vote_summary['current_threshold']:.3f}")
        
        # Historique des expansions
        if self.training_logs:
            print("\nTraining History (last 10):")
            print("-" * 60)
            for log in self.training_logs[-10:]:
                exp_str = f"[{log.expansion_type}]" if log.expanded else ""
                warmup_str = "[W]" if log.warmup_active else ""
                print(
                    f"  Cycle {log.cycle}: loss={log.avg_loss:.4f}, "
                    f"params={log.total_params:,}, "
                    f"blocks={log.num_blocks} {exp_str}{warmup_str}"
                )
    
    def get_training_curve(self) -> tuple:
        """Retourne les données pour tracer la courbe d'apprentissage."""
        cycles = [log.cycle for log in self.training_logs]
        losses = [log.avg_loss for log in self.training_logs]
        params = [log.total_params for log in self.training_logs]
        return cycles, losses, params
