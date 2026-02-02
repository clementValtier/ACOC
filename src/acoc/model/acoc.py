"""
ACOC - Modèle Principal (PyTorch)
=================================
Le modèle ACOC avec architecture modulaire et croissance organique.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set

from ..config import (
    SystemConfig, TaskBlock, ModelMetrics,
    ExpansionDecision, TaskType, SaturationMetrics
)
from ..core import Router, Expert
from ..management import ExpansionManager, PenaltyManager, PruningManager, WarmupManager
from ..variants import VariantSystem


class ACOCModel(nn.Module):
    """
    Modèle principal ACOC (Adaptive Controlled Organic Capacity).
    """
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        
        self.config = config
        self.current_cycle = 0
        
        # Device
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # === Composants du réseau ===
        self.router = Router(
            input_dim=config.input_dim,
            num_routes=3,  # text, image, audio
            hidden_dim=128
        )
        self.task_blocks: Dict[str, TaskBlock] = {}
        self._block_modules: nn.ModuleDict = nn.ModuleDict()
        
        # === Métriques ===
        self.metrics = ModelMetrics()
        
        # === Managers ===
        self.variant_system = VariantSystem(config, self.device)
        self.expansion_manager = ExpansionManager(config)
        self.penalty_manager = PenaltyManager(config)
        self.pruning_manager = PruningManager(config)
        self.warmup_manager = WarmupManager(config)
        
        # === Tracking pour le warmup et l'exploration ===
        self._force_exploration_block: Optional[str] = None
        self._exploration_prob: float = 0.0
        
        # === Initialisation ===
        self._initialize_base_blocks()
        self.to(self.device)
    
    def _initialize_base_blocks(self):
        """Crée les blocs de base pour chaque type de tâche principal."""
        base_types = [TaskType.TEXT, TaskType.IMAGE, TaskType.AUDIO]
        
        for i, task_type in enumerate(base_types):
            block_id = f"base_{task_type.value}"
            expert = Expert(
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                name=f"{block_id}_expert_0"
            )
            
            block = TaskBlock(
                id=block_id,
                task_type=task_type,
                num_params=expert.get_param_count(),
                layers=[expert],
                creation_cycle=0
            )
            
            self.task_blocks[block_id] = block
            self._block_modules[block_id] = expert
    
    def _sync_modules(self):
        """Synchronise les modules PyTorch avec task_blocks."""
        # Supprimer les modules qui ne sont plus dans task_blocks
        to_remove = [k for k in self._block_modules.keys() if k not in self.task_blocks]
        for k in to_remove:
            del self._block_modules[k]
        
        # Ajouter les nouveaux modules
        for block_id, block in self.task_blocks.items():
            if block_id not in self._block_modules:
                # Créer un module combiné pour toutes les layers
                if len(block.layers) == 1:
                    self._block_modules[block_id] = block.layers[0]
                else:
                    self._block_modules[block_id] = nn.Sequential(*block.layers)
    
    def set_exploration(self, block_id: Optional[str], prob: float = 0.0):
        """Configure l'exploration forcée vers un bloc."""
        self._force_exploration_block = block_id
        self._exploration_prob = prob
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_hint: Optional[TaskType] = None
    ) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Forward pass avec routage automatique.
        
        Args:
            x: Input [batch_size, input_dim]
            task_hint: Optionnel, force l'utilisation d'un type de bloc
            
        Returns:
            (output, routing_stats)
        """
        x = x.to(self.device)
        batch_size = x.size(0)
        
        # Routage avec exploration optionnelle
        if self._force_exploration_block is not None and self._exploration_prob > 0:
            block_ids = list(self.task_blocks.keys())
            if self._force_exploration_block in block_ids:
                force_idx = block_ids.index(self._force_exploration_block)
                selected, probs = self.router.forward_with_exploration(
                    x, 
                    force_route=force_idx,
                    exploration_prob=self._exploration_prob
                )
            else:
                selected, probs = self.router(x)
        else:
            selected, probs = self.router(x)
        
        # Mapper les indices aux block_ids
        block_ids = list(self.task_blocks.keys())
        
        # Statistiques de routage
        routing_stats = {bid: 0 for bid in block_ids}
        
        # Grouper par bloc pour le batch processing
        outputs = torch.zeros(batch_size, self.config.output_dim, device=self.device)
        
        for idx, block_id in enumerate(block_ids):
            mask = selected == idx
            if not mask.any():
                continue
            
            block_input = x[mask]
            block = self.task_blocks[block_id]
            
            # Forward dans le bloc
            block_output = block_input
            for layer in block.layers:
                block_output = layer(block_output)
            
            outputs[mask] = block_output
            
            # Mettre à jour les stats
            count = mask.sum().item()
            routing_stats[block_id] = count
            block.usage_count += count
            block.last_used_cycle = self.current_cycle
        
        return outputs, routing_stats
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        include_penalties: bool = True
    ) -> torch.Tensor:
        """
        Calcule la loss totale avec pénalités optionnelles.
        Par défaut utilise CrossEntropy (classification).
        """
        # Loss de base - CrossEntropy pour classification
        if self.config.use_cross_entropy:
            # targets doit être des indices de classe (shape: [batch])
            # predictions est logits (shape: [batch, num_classes])
            if targets.dim() == 2:  # Si one-hot, convertir en indices
                targets = targets.argmax(dim=1)
            base_loss = nn.functional.cross_entropy(predictions, targets)
        else:
            # MSE pour régression (legacy - non recommandé)
            import warnings
            warnings.warn(
                "MSE loss est déprécié pour ACOC. Utilisez use_cross_entropy=True.",
                DeprecationWarning,
                stacklevel=2
            )
            base_loss = nn.functional.mse_loss(predictions, targets)
        
        if not include_penalties:
            return base_loss
        
        # Pénalités de taille
        total_penalty, _, _ = self.penalty_manager.compute_total_penalty(
            self.task_blocks, 
            self.router.get_param_count()
        )
        penalty_tensor = torch.tensor(total_penalty, device=self.device)
        
        # Pénalité EWC pour le routeur
        ewc_penalty = self.router.ewc_loss()
        
        return base_loss + penalty_tensor + ewc_penalty
    
    def collect_metrics(self) -> ModelMetrics:
        """Collecte les métriques de saturation détaillées."""
        # Saturation détaillée par bloc
        for block_id, block in self.task_blocks.items():
            all_metrics = []
            for layer in block.layers:
                if isinstance(layer, Expert):
                    all_metrics.append(layer.get_saturation_metrics())
            
            if all_metrics:
                # Moyenne des métriques
                combined = SaturationMetrics()
                n = len(all_metrics)
                combined.gradient_flow_ratio = sum(m.gradient_flow_ratio for m in all_metrics) / n
                combined.activation_saturation = sum(m.activation_saturation for m in all_metrics) / n
                combined.dead_neuron_ratio = sum(m.dead_neuron_ratio for m in all_metrics) / n
                combined.activation_variance = sum(m.activation_variance for m in all_metrics) / n
                combined.compute_combined_score()
                
                self.metrics.detailed_saturation[block_id] = combined
                self.metrics.saturation_scores[block_id] = combined.combined_score
                
                # Stocker aussi dans le bloc
                block.saturation = combined
        
        # Utilisation des experts
        total_usage = sum(b.usage_count for b in self.task_blocks.values())
        if total_usage > 0:
            for block_id, block in self.task_blocks.items():
                self.metrics.expert_utilization[block_id] = block.usage_count / total_usage
        
        return self.metrics
    
    def reset_usage_counts(self):
        """Réinitialise les compteurs d'utilisation."""
        for block in self.task_blocks.values():
            block.usage_count = 0
    
    def evaluate_expansion(self) -> ExpansionDecision:
        """Évalue le besoin d'expansion basé sur les métriques."""
        return self.expansion_manager.evaluate_expansion_need(
            self.metrics,
            self.task_blocks,
            self.current_cycle
        )
    
    def execute_expansion(self, decision: ExpansionDecision) -> bool:
        """Exécute une décision d'expansion."""
        success = self.expansion_manager.execute_expansion(
            decision,
            self.task_blocks,
            self.current_cycle,
            self.device
        )
        
        if success:
            if decision.expansion_type == "new_block":
                # Ajouter une route pour le nouveau bloc
                self.router.add_route(self.device)
                # Démarrer le warmup
                if decision.target_block_id:
                    self.warmup_manager.start_warmup(
                        decision.target_block_id,
                        self.current_cycle
                    )
                    # Activer l'exploration
                    self.set_exploration(
                        decision.target_block_id,
                        self.config.new_block_exploration_prob
                    )
            
            # Synchroniser les modules
            self._sync_modules()
            self.to(self.device)
        
        return success
    
    def run_maintenance(self) -> Dict[str, List[str]]:
        """Exécute la maintenance (pruning, consolidation)."""
        # Protéger les blocs en warmup
        protected = set(self.warmup_manager.get_warmup_blocks())
        protected.update(self.expansion_manager.get_warmup_blocks(self.current_cycle))
        
        actions = self.pruning_manager.run_maintenance(
            self.task_blocks,
            self.current_cycle,
            protected
        )
        
        # Synchroniser les modules après pruning
        self._sync_modules()
        
        return actions
    
    def get_total_params(self) -> int:
        """Retourne le nombre total de paramètres."""
        return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        """Retourne un résumé textuel du modèle."""
        lines = [
            "=" * 50,
            "ACOC Model Summary",
            "=" * 50,
            f"Device: {self.device}",
            f"Cycle: {self.current_cycle}",
            f"Total params: {self.get_total_params():,}",
            f"Num blocks: {len(self.task_blocks)}",
            "",
            "Blocks:",
        ]
        
        for block_id, block in self.task_blocks.items():
            sat = block.saturation.combined_score if block.saturation else 0
            util = self.metrics.expert_utilization.get(block_id, 0)
            warmup = " [WARMUP]" if self.warmup_manager.is_warmup_active(block_id) else ""
            lines.append(
                f"  {block_id}: {block.num_params:,} params, "
                f"sat={sat:.1%}, util={util:.1%}, "
                f"created@{block.creation_cycle}{warmup}"
            )
        
        lines.append("")
        lines.append(f"Expansions: {len(self.expansion_manager.expansion_history)}")
        lines.append(f"Prunings: {len(self.pruning_manager.pruning_history)}")
        lines.append("=" * 50)
        
        return "\n".join(lines)
