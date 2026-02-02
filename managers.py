"""
ACOC - Gestionnaires (PyTorch)
==============================
ExpansionManager, PenaltyManager, PruningManager, WarmupManager
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set

from .structures import (
    SystemConfig, TaskBlock, ModelMetrics, 
    ExpansionDecision, TaskType, SaturationMetrics
)
from .components import Expert


class ExpansionManager:
    """
    Décide quand et comment étendre le modèle.
    
    Triggers d'expansion basés sur:
    1. Score de saturation combiné (gradient flow + activations)
    2. Loss stagnante
    3. Vote des variantes
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.last_expansion_cycle = -config.expansion_cooldown
        self.expansion_history: List[Tuple[int, str, str]] = []
        
        # Tracking des blocs récemment créés (pour le warmup)
        self.recently_created_blocks: Dict[str, int] = {}  # block_id -> creation_cycle
    
    def evaluate_expansion_need(
        self, 
        metrics: ModelMetrics, 
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int
    ) -> ExpansionDecision:
        """
        Analyse les métriques de saturation et décide si une expansion est nécessaire.
        
        Utilise les métriques détaillées:
        - Gradient flow ratio (signal qui ne passe plus)
        - Activation saturation (neurones au max)
        - Dead neuron ratio (neurones inutiles)
        """
        # --- Vérifier le cooldown ---
        cycles_since_last = current_cycle - self.last_expansion_cycle
        if cycles_since_last < self.config.expansion_cooldown:
            return ExpansionDecision(
                should_expand=False,
                reason=f"Cooldown actif ({cycles_since_last}/{self.config.expansion_cooldown} cycles)"
            )
        
        # --- Vérifier le nombre minimum de cycles ---
        if current_cycle < self.config.min_cycles_before_expand:
            return ExpansionDecision(
                should_expand=False,
                reason=f"Historique insuffisant ({current_cycle}/{self.config.min_cycles_before_expand} cycles)"
            )
        
        # --- Analyser la saturation détaillée par bloc ---
        saturated_blocks = []
        
        for block_id, sat_metrics in metrics.detailed_saturation.items():
            # Ignorer les blocs en période de warmup
            if block_id in self.recently_created_blocks:
                creation_cycle = self.recently_created_blocks[block_id]
                if current_cycle - creation_cycle < self.config.new_block_exploration_cycles:
                    continue
            
            # Utiliser le score combiné
            if sat_metrics.combined_score > self.config.saturation_threshold:
                saturated_blocks.append((block_id, sat_metrics))
        
        # --- Cas 1: Blocs saturés → Expansion en largeur ---
        if saturated_blocks:
            # Trier par score de saturation décroissant
            saturated_blocks.sort(key=lambda x: x[1].combined_score, reverse=True)
            target_block_id, sat_metrics = saturated_blocks[0]
            
            # Déterminer la raison principale
            reasons = []
            if sat_metrics.gradient_flow_ratio < 0.5:
                reasons.append(f"gradient flow faible ({sat_metrics.gradient_flow_ratio:.1%})")
            if sat_metrics.activation_saturation > 0.3:
                reasons.append(f"activations saturées ({sat_metrics.activation_saturation:.1%})")
            if sat_metrics.dead_neuron_ratio > 0.2:
                reasons.append(f"neurones morts ({sat_metrics.dead_neuron_ratio:.1%})")
            
            reason_str = ", ".join(reasons) if reasons else "score combiné élevé"
            
            return ExpansionDecision(
                should_expand=True,
                target_block_id=target_block_id,
                expansion_type="width",
                confidence=sat_metrics.combined_score,
                reason=f"Bloc '{target_block_id}' saturé: {reason_str} (score={sat_metrics.combined_score:.2f})"
            )
        
        # --- Cas 2: Loss stagnante sans saturation → Nouveau bloc ---
        loss_trend = metrics.get_recent_loss_trend(window=10)
        if loss_trend is not None and loss_trend < 0.01:
            return ExpansionDecision(
                should_expand=True,
                expansion_type="new_block",
                confidence=0.5,
                reason=f"Loss stagnante (amélioration: {loss_trend:.2%}), capacité potentiellement insuffisante"
            )
        
        # --- Pas besoin d'expansion ---
        return ExpansionDecision(
            should_expand=False,
            reason="Pas de saturation ni de stagnation détectée"
        )
    
    def execute_expansion(
        self,
        decision: ExpansionDecision,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        device: torch.device = None
    ) -> bool:
        """
        Exécute l'expansion décidée.
        
        Returns:
            True si l'expansion a réussi
        """
        if not decision.should_expand:
            return False
        
        if device is None:
            device = torch.device('cpu')
        
        success = False
        
        if decision.expansion_type == "width":
            success = self._expand_width(decision.target_block_id, task_blocks)
            
        elif decision.expansion_type == "depth":
            success = self._expand_depth(decision.target_block_id, task_blocks)
            
        elif decision.expansion_type == "new_block":
            new_id = self._create_new_block(task_blocks, current_cycle, device)
            if new_id:
                decision.target_block_id = new_id
                self.recently_created_blocks[new_id] = current_cycle
                success = True
        
        if success:
            self.last_expansion_cycle = current_cycle
            self.expansion_history.append((
                current_cycle, 
                decision.target_block_id or "new",
                decision.expansion_type
            ))
        
        return success
    
    def _expand_width(
        self, 
        block_id: str, 
        task_blocks: Dict[str, TaskBlock]
    ) -> bool:
        """Ajoute des neurones aux experts d'un bloc."""
        if block_id not in task_blocks:
            return False
        
        block = task_blocks[block_id]
        
        for layer in block.layers:
            if isinstance(layer, Expert):
                additional = max(1, int(layer.hidden_dim * self.config.expansion_ratio))
                layer.expand_width(additional)
                layer.reset_monitors()
        
        block.update_param_count()
        return True
    
    def _expand_depth(
        self, 
        block_id: str, 
        task_blocks: Dict[str, TaskBlock]
    ) -> bool:
        """Ajoute une couche à un bloc."""
        if block_id not in task_blocks:
            return False
        
        block = task_blocks[block_id]
        
        if block.layers and isinstance(block.layers[-1], Expert):
            last_expert = block.layers[-1]
            # Créer un nouvel expert avec les mêmes dimensions
            new_expert = Expert(
                input_dim=last_expert.output_dim,
                hidden_dim=last_expert.output_dim,
                output_dim=last_expert.output_dim,
                name=f"{block_id}_expert_{len(block.layers)}"
            )
            block.layers.append(new_expert)
            block.update_param_count()
            return True
        
        return False
    
    def _create_new_block(
        self, 
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        device: torch.device
    ) -> Optional[str]:
        """Crée un nouveau bloc générique."""
        new_id = f"block_{len(task_blocks)}"
        
        expert = Expert(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            name=f"{new_id}_expert_0"
        ).to(device)
        
        new_block = TaskBlock(
            id=new_id,
            task_type=TaskType.GENERIC,
            num_params=expert.get_param_count(),
            layers=[expert],
            creation_cycle=current_cycle
        )
        
        task_blocks[new_id] = new_block
        return new_id
    
    def is_in_warmup(self, block_id: str, current_cycle: int) -> bool:
        """Vérifie si un bloc est en période de warmup."""
        if block_id not in self.recently_created_blocks:
            return False
        creation_cycle = self.recently_created_blocks[block_id]
        return current_cycle - creation_cycle < self.config.new_block_exploration_cycles
    
    def get_warmup_blocks(self, current_cycle: int) -> List[str]:
        """Retourne la liste des blocs en période de warmup."""
        return [
            block_id for block_id, creation 
            in self.recently_created_blocks.items()
            if current_cycle - creation < self.config.new_block_exploration_cycles
        ]
    
    def get_expansion_stats(self) -> Dict:
        """Retourne des statistiques sur les expansions."""
        type_counts = {}
        for _, _, exp_type in self.expansion_history:
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1
        
        return {
            "total_expansions": len(self.expansion_history),
            "by_type": type_counts,
            "last_expansion_cycle": self.last_expansion_cycle
        }


class WarmupManager:
    """
    Gère le warmup après une expansion.
    
    Responsabilités:
    - Learning rate plus élevé pour les nouveaux paramètres
    - Forcer l'exploration vers les nouveaux blocs
    - Tracking de la phase de warmup
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.active_warmups: Dict[str, Dict] = {}  # block_id -> warmup_info
    
    def start_warmup(
        self, 
        block_id: str, 
        current_cycle: int,
        new_params: Optional[Set[str]] = None
    ):
        """
        Démarre un warmup pour un bloc.
        
        Args:
            block_id: ID du bloc
            current_cycle: Cycle actuel
            new_params: Noms des paramètres nouvellement ajoutés
        """
        self.active_warmups[block_id] = {
            "start_cycle": current_cycle,
            "steps_done": 0,
            "new_params": new_params or set()
        }
    
    def is_warmup_active(self, block_id: str = None) -> bool:
        """Vérifie si un warmup est actif."""
        if block_id:
            return block_id in self.active_warmups
        return len(self.active_warmups) > 0
    
    def get_warmup_blocks(self) -> List[str]:
        """Retourne les blocs en warmup."""
        return list(self.active_warmups.keys())
    
    def step(self, block_id: str = None):
        """Incrémente le compteur de steps de warmup."""
        if block_id:
            if block_id in self.active_warmups:
                self.active_warmups[block_id]["steps_done"] += 1
        else:
            for info in self.active_warmups.values():
                info["steps_done"] += 1
    
    def should_continue_warmup(self, block_id: str) -> bool:
        """Vérifie si le warmup doit continuer."""
        if block_id not in self.active_warmups:
            return False
        return self.active_warmups[block_id]["steps_done"] < self.config.warmup_steps
    
    def end_warmup(self, block_id: str):
        """Termine le warmup pour un bloc."""
        if block_id in self.active_warmups:
            del self.active_warmups[block_id]
    
    def check_and_cleanup(self):
        """Nettoie les warmups terminés."""
        to_remove = [
            block_id for block_id, info in self.active_warmups.items()
            if info["steps_done"] >= self.config.warmup_steps
        ]
        for block_id in to_remove:
            del self.active_warmups[block_id]
    
    def get_lr_multiplier(self, param_name: str) -> float:
        """
        Retourne le multiplicateur de LR pour un paramètre.
        Les nouveaux paramètres ont un LR plus élevé.
        """
        for info in self.active_warmups.values():
            if param_name in info["new_params"]:
                return self.config.warmup_lr_multiplier
        return 1.0
    
    def get_exploration_prob(self, block_id: str) -> float:
        """
        Retourne la probabilité d'exploration forcée vers un bloc.
        """
        if block_id in self.active_warmups:
            return self.config.new_block_exploration_prob
        return 0.0


class PenaltyManager:
    """
    Calcule les pénalités de taille (double malus).
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.baseline_params = 100_000
        self.penalty_history: List[float] = []
    
    def compute_total_penalty(
        self, 
        task_blocks: Dict[str, TaskBlock],
        router_params: int
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Calcule la pénalité totale.
        
        Returns:
            (total_penalty, global_penalty, task_penalties_dict)
        """
        # --- Pénalité globale (logarithmique) ---
        total_params = router_params + sum(b.num_params for b in task_blocks.values())
        global_penalty = self.config.alpha_global_penalty * np.log(
            1 + total_params / self.baseline_params
        )
        
        # --- Pénalités par tâche (quadratique au-delà du seuil) ---
        task_penalties = {}
        for block_id, block in task_blocks.items():
            excess = max(0, block.num_params - self.config.task_param_threshold)
            task_penalties[block_id] = self.config.beta_task_penalty * (excess / 1e6) ** 2
        
        total_task_penalty = sum(task_penalties.values())
        total_penalty = global_penalty + total_task_penalty
        
        self.penalty_history.append(total_penalty)
        
        return total_penalty, global_penalty, task_penalties
    
    def adjust_thresholds(self, metrics: ModelMetrics) -> bool:
        """
        Ajuste dynamiquement les pénalités si la loss stagne.
        """
        loss_trend = metrics.get_recent_loss_trend(window=20)
        
        if loss_trend is None:
            return False
        
        # Si moins de 0.5% d'amélioration sur 20 cycles, relâcher
        if loss_trend < 0.005:
            self.config.alpha_global_penalty *= 0.95
            self.config.beta_task_penalty *= 0.95
            return True
        
        # Si amélioration > 5%, on peut resserrer légèrement
        if loss_trend > 0.05:
            self.config.alpha_global_penalty *= 1.02
            self.config.beta_task_penalty *= 1.02
            return True
        
        return False


class PruningManager:
    """
    Gère la suppression et consolidation des blocs inutilisés.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.pruning_history: List[Tuple[int, str, str]] = []
    
    def identify_unused_blocks(
        self, 
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        protected_blocks: Set[str] = None
    ) -> List[str]:
        """
        Identifie les blocs qui n'ont pas été utilisés depuis longtemps.
        """
        if protected_blocks is None:
            protected_blocks = set()
        
        unused = []
        
        for block_id, block in task_blocks.items():
            if block_id in protected_blocks:
                continue
                
            cycles_since_use = current_cycle - block.last_used_cycle
            
            if (cycles_since_use > self.config.prune_unused_after_cycles and
                block.usage_count < current_cycle * 0.1):
                unused.append(block_id)
        
        return unused
    
    def find_similar_blocks(
        self, 
        task_blocks: Dict[str, TaskBlock]
    ) -> List[Tuple[str, str, float]]:
        """
        Trouve les paires de blocs suffisamment similaires pour être fusionnés.
        """
        similar_pairs = []
        block_ids = list(task_blocks.keys())
        
        for i, id1 in enumerate(block_ids):
            for id2 in block_ids[i + 1:]:
                block1 = task_blocks[id1]
                block2 = task_blocks[id2]
                
                if block1.task_type != block2.task_type:
                    continue
                
                similarity = self._compute_block_similarity(block1, block2)
                
                if similarity > self.config.consolidation_similarity_threshold:
                    similar_pairs.append((id1, id2, similarity))
        
        return similar_pairs
    
    def _compute_block_similarity(
        self, 
        block1: TaskBlock, 
        block2: TaskBlock
    ) -> float:
        """Calcule la similarité entre deux blocs."""
        size_ratio = min(block1.num_params, block2.num_params) / \
                     max(block1.num_params, block2.num_params)
        
        total_usage = block1.usage_count + block2.usage_count
        if total_usage > 0:
            usage_balance = 1 - abs(block1.usage_count - block2.usage_count) / total_usage
        else:
            usage_balance = 1.0
        
        type_match = 1.0 if block1.task_type == block2.task_type else 0.5
        
        return size_ratio * usage_balance * type_match
    
    def prune_block(
        self, 
        task_blocks: Dict[str, TaskBlock], 
        block_id: str,
        current_cycle: int
    ) -> bool:
        """Supprime un bloc."""
        if block_id in task_blocks:
            del task_blocks[block_id]
            self.pruning_history.append((current_cycle, block_id, "pruned"))
            return True
        return False
    
    def consolidate_blocks(
        self,
        task_blocks: Dict[str, TaskBlock],
        block_id_1: str,
        block_id_2: str,
        current_cycle: int
    ) -> Optional[str]:
        """Fusionne deux blocs en un seul."""
        if block_id_1 not in task_blocks or block_id_2 not in task_blocks:
            return None
        
        block1 = task_blocks[block_id_1]
        block2 = task_blocks[block_id_2]
        
        if block1.usage_count >= block2.usage_count:
            survivor_id, remove_id = block_id_1, block_id_2
            survivor = block1
        else:
            survivor_id, remove_id = block_id_2, block_id_1
            survivor = block2
        
        survivor.usage_count += task_blocks[remove_id].usage_count
        
        del task_blocks[remove_id]
        self.pruning_history.append((current_cycle, remove_id, f"merged_into_{survivor_id}"))
        
        return survivor_id
    
    def run_maintenance(
        self,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        protected_blocks: Set[str] = None
    ) -> Dict[str, List[str]]:
        """Exécute un cycle complet de maintenance."""
        if protected_blocks is None:
            protected_blocks = set()
            
        actions = {"pruned": [], "consolidated": []}
        
        # 1. Identifier et supprimer les blocs inutilisés
        unused = self.identify_unused_blocks(task_blocks, current_cycle, protected_blocks)
        for block_id in unused:
            if self.prune_block(task_blocks, block_id, current_cycle):
                actions["pruned"].append(block_id)
        
        # 2. Consolider les blocs similaires (max 1 par cycle)
        similar = self.find_similar_blocks(task_blocks)
        similar = [(a, b, s) for a, b, s in similar 
                   if a not in protected_blocks and b not in protected_blocks]
        
        if similar:
            id1, id2, _ = similar[0]
            survivor = self.consolidate_blocks(task_blocks, id1, id2, current_cycle)
            if survivor:
                removed = id1 if survivor == id2 else id2
                actions["consolidated"].append(f"{removed} → {survivor}")
        
        return actions
