"""
ACOC - Gradient Flow Monitor
============================
Moniteur de gradient flow pour détecter les blocages dans le réseau.
"""

import torch
from typing import Dict
from collections import deque


class GradientFlowMonitor:
    """
    Moniteur de gradient flow pour détecter les blocages.

    Analyse:
    - Magnitude des gradients par couche
    - Ratio de gradients "vivants"
    - Détection de vanishing/exploding gradients
    """

    def __init__(self, threshold: float = 1e-6, history_size: int = 100):
        self.threshold = threshold
        self.history_size = history_size
        self.gradient_history: Dict[str, deque] = {}

    def register_layer(self, name: str):
        """Enregistre une couche à monitorer."""
        if name not in self.gradient_history:
            self.gradient_history[name] = deque(maxlen=self.history_size)

    def record_gradients(self, name: str, gradients: torch.Tensor):
        """Enregistre les gradients d'une couche."""
        if name not in self.gradient_history:
            self.register_layer(name)

        with torch.no_grad():
            grad_abs = gradients.abs()
            stats = {
                'mean': grad_abs.mean().item(),
                'max': grad_abs.max().item(),
                'alive_ratio': (grad_abs > self.threshold).float().mean().item()
            }
            self.gradient_history[name].append(stats)

    def get_flow_ratio(self, name: str) -> float:
        """
        Retourne le ratio de gradients vivants (moyenne récente).
        1.0 = tous les gradients circulent bien
        0.0 = gradient flow bloqué
        """
        if name not in self.gradient_history or len(self.gradient_history[name]) == 0:
            return 1.0

        recent = list(self.gradient_history[name])[-20:]
        avg_alive = sum(s['alive_ratio'] for s in recent) / len(recent)
        return avg_alive

    def get_all_flow_ratios(self) -> Dict[str, float]:
        """Retourne les ratios pour toutes les couches."""
        return {name: self.get_flow_ratio(name) for name in self.gradient_history}
