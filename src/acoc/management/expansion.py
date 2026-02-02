"""
ACOC - Expansion Manager
========================
Décide quand et comment étendre le modèle.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from ..config import (
    SystemConfig, TaskBlock, ModelMetrics,
    ExpansionDecision, TaskType
)
from ..experts import ExpertFactory, ExpertBlock, BaseExpert

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

    def update_recent_usage(self, task_blocks: Dict[str, TaskBlock], current_cycle: int):
        """Met à jour l'historique d'utilisation récente de tous les blocs."""
        for block in task_blocks.values():
            # Ajouter l'utilisation du cycle actuel
            block.recent_usage.append(block.usage_count)

            # Garder seulement les N derniers cycles
            if len(block.recent_usage) > self.config.recent_usage_window:
                block.recent_usage.pop(0)

    def get_most_used_block_recent(self, task_blocks: Dict[str, TaskBlock]) -> Optional[TaskBlock]:
        """Retourne le bloc avec la plus grande utilisation récente (moyenne)."""
        if not task_blocks:
            return None

        def get_recent_avg(block: TaskBlock) -> float:
            if not block.recent_usage:
                return 0.0
            return sum(block.recent_usage) / len(block.recent_usage)

        return max(task_blocks.values(), key=get_recent_avg)

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
            new_id = self._create_new_block(task_blocks, current_cycle, device, decision.target_block_id)
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
            if isinstance(layer, BaseExpert):
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

        if block.layers and isinstance(block.layers[-1], BaseExpert):
            last_expert = block.layers[-1]

            # On récupère le type de l'expert précédent pour créer le même
            # (Si c'était un CNN, on ajoute un CNN, etc.)
            expert_type = getattr(last_expert, 'expert_type', 'mlp')
            
            # Créer un nouvel expert via la Factory
            new_expert = ExpertFactory.create(
                expert_type=expert_type,
                input_dim=last_expert.output_dim,   # L'entrée est la sortie du précédent
                hidden_dim=last_expert.output_dim,  # On garde la même largeur
                output_dim=last_expert.output_dim,  # On garde la même sortie
                name=f"{block_id}_expert_{len(block.layers)}",
                config=self.config
            ).to(last_expert.parameters().__next__().device) # Même device que le précédent
            
            block.layers.append(new_expert)
            block.update_param_count()
            return True

        return False

    def _create_new_block(
        self,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        device: torch.device,
        target_id_hint: Optional[str] = None
    ) -> Optional[str]:
        new_id = f"block_{len(task_blocks)}"

        task_type = TaskType.GENERIC
        expert_type = "mlp"

        # LOGIQUE: Hériter du bloc le plus utilisé récemment
        if task_blocks:
            # Trouver le bloc avec la plus grande utilisation récente
            most_used_block = self.get_most_used_block_recent(task_blocks)

            if most_used_block:
                # Hériter du type et de l'expert type
                task_type = most_used_block.task_type
                if most_used_block.layers and isinstance(most_used_block.layers[0], BaseExpert):
                    expert_type = getattr(most_used_block.layers[0], 'expert_type', "mlp")

            # Override: si target_id_hint est spécifié, l'utiliser en priorité
            if target_id_hint and target_id_hint in task_blocks:
                parent = task_blocks[target_id_hint]
                task_type = parent.task_type
                if parent.layers and isinstance(parent.layers[0], BaseExpert):
                    expert_type = getattr(parent.layers[0], 'expert_type', "mlp")

        # Utilisation de la Factory
        expert = ExpertFactory.create(
            expert_type=expert_type,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            name=f"{new_id}_expert_0",
            config=self.config
        ).to(device)

        new_block = TaskBlock(
            id=new_id,
            task_type=task_type,
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
