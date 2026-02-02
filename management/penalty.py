"""
ACOC - Penalty Manager
======================
Calcule les pénalités de taille (double malus).
"""

import numpy as np
from typing import Dict, List, Tuple

from ..config import SystemConfig, TaskBlock, ModelMetrics


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
