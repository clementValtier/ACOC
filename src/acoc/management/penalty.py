"""
ACOC - Penalty Manager
======================
Computes size penalties (dual penalty system).
"""

import numpy as np
from typing import Dict, List, Tuple

from ..config import SystemConfig, TaskBlock, ModelMetrics


class PenaltyManager:
    """
    Computes size penalties using a dual penalty system.
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
        Computes the total penalty.

        Returns:
            (total_penalty, global_penalty, task_penalties_dict)
        """
        # Global penalty (logarithmic) - penalizes total model size
        total_params = router_params + sum(b.num_params for b in task_blocks.values())
        global_penalty = self.config.alpha_global_penalty * np.log(
            1 + total_params / self.baseline_params
        )

        # Per-task penalties (quadratic beyond threshold)
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
        Dynamically adjusts penalties when loss stagnates.
        """
        loss_trend = metrics.get_recent_loss_trend(window=20)

        if loss_trend is None:
            return False

        # If improvement is less than 0.5% over 20 cycles, relax penalties
        if loss_trend < 0.005:
            self.config.alpha_global_penalty *= 0.95
            self.config.beta_task_penalty *= 0.95
            return True

        # If improvement is greater than 5%, tighten penalties slightly
        if loss_trend > 0.05:
            self.config.alpha_global_penalty *= 1.02
            self.config.beta_task_penalty *= 1.02
            return True

        return False
