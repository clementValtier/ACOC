"""
ACOC - Base Expert
==================
Classe abstraite définissant l'interface commune de tous les experts.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List

from ..monitoring import ActivationMonitor, GradientFlowMonitor
from ..config import SaturationMetrics, SystemConfig

class BaseExpert(nn.Module, ABC):
    """
    Classe de base pour tous les types d'experts (MLP, CNN, Transformer, etc.).
    Gère le monitoring et l'interface standard.
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
        
        # Monitoring
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._hooks = []

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passage avant du réseau."""
        pass

    @abstractmethod
    def expand_width(self, additional_neurons: int):
        """Expansion Net2Net en largeur."""
        pass

    @abstractmethod
    def get_param_count(self) -> int:
        """Retourne le nombre de paramètres."""
        pass

    def get_saturation_metrics(self) -> SaturationMetrics:
        """Calcule les métriques de saturation (Commun à tous)."""
        metrics = SaturationMetrics()
        
        # On suppose que chaque implémentation définit ses clés de monitoring
        # Généralement "name_fc1" pour le gradient et "name_hidden" pour l'activation
        metrics.gradient_flow_ratio = self.gradient_monitor.get_flow_ratio(f"{self.name}_fc1")
        
        sat, dead, var = self.activation_monitor.get_saturation_metrics(f"{self.name}_hidden")
        metrics.activation_saturation = sat
        metrics.dead_neuron_ratio = dead
        metrics.activation_variance = var
        
        metrics.compute_combined_score()
        return metrics

    def reset_monitors(self):
        """Réinitialise les moniteurs (après expansion)."""
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._register_hooks()

    @abstractmethod
    def _register_hooks(self):
        """Enregistre les hooks PyTorch spécifiques à l'architecture."""
        pass