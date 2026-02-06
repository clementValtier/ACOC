"""
ACOC - Base Expert
==================
Abstract class defining the common interface for all experts.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List

from ..monitoring import ActivationMonitor, GradientFlowMonitor
from ..config import SaturationMetrics, SystemConfig

class BaseExpert(nn.Module, ABC):
    """
    Base class for all expert types (MLP, CNN, Transformer, etc.).
    Manages monitoring and standard interface.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        name: str,
        config: Optional[SystemConfig] = None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name
        self.config = config
        self.expert_type: str = "mlp"  # overridden by subclasses

        # Monitoring
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._hooks = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass

    @abstractmethod
    def expand_width(self, additional_neurons: int):
        """Net2Net width expansion."""
        pass

    @abstractmethod
    def get_param_count(self) -> int:
        """Returns the number of parameters."""
        pass

    def get_saturation_metrics(self) -> SaturationMetrics:
        """Computes saturation metrics (common to all)."""
        metrics = SaturationMetrics()

        # We assume each implementation defines its monitoring keys
        # Typically "name_fc1" for gradient and "name_hidden" for activation
        metrics.gradient_flow_ratio = self.gradient_monitor.get_flow_ratio(f"{self.name}_fc1")

        sat, dead, var = self.activation_monitor.get_saturation_metrics(f"{self.name}_hidden")
        metrics.activation_saturation = sat
        metrics.dead_neuron_ratio = dead
        metrics.activation_variance = var

        metrics.compute_combined_score()
        return metrics

    def reset_monitors(self):
        """Resets monitors (after expansion)."""
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._register_hooks()

    @abstractmethod
    def _register_hooks(self):
        """Registers PyTorch hooks specific to the architecture."""
        pass