"""
ACOC - Expert Block
===================
Container for expert layers.
"""

import torch
import torch.nn as nn
from typing import List
from ..config import SaturationMetrics
from .base import BaseExpert

class ExpertBlock(nn.Module):
    def __init__(self, experts: List[BaseExpert], name: str = "block"):
        super().__init__()
        self.name = name
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for expert in self.experts:
            x = expert(x)  # type: ignore[operator]
        return x

    def get_combined_saturation(self) -> SaturationMetrics:
        if not self.experts: return SaturationMetrics()
        all_metrics = [e.get_saturation_metrics() for e in self.experts]  # type: ignore[operator]

        combined = SaturationMetrics()
        n = len(all_metrics)
        # Average metrics across all experts
        combined.gradient_flow_ratio = sum(m.gradient_flow_ratio for m in all_metrics) / n
        combined.activation_saturation = sum(m.activation_saturation for m in all_metrics) / n
        combined.dead_neuron_ratio = sum(m.dead_neuron_ratio for m in all_metrics) / n
        combined.activation_variance = sum(m.activation_variance for m in all_metrics) / n
        combined.compute_combined_score()
        return combined

    def expand_all_experts(self, ratio: float = 0.1):
        for expert in self.experts:
            additional = max(1, int(expert.hidden_dim * ratio))  # type: ignore[operator]
            expert.expand_width(additional)  # type: ignore[operator]

    def reset_all_monitors(self):
        for expert in self.experts:
            expert.reset_monitors()  # type: ignore[operator]

    def get_param_count(self) -> int:
        return sum(e.get_param_count() for e in self.experts)  # type: ignore[operator]