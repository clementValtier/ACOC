"""
ACOC - Activation Monitor
=========================
Moniteur de saturation des activations pour détecter les neurones saturés/morts.
"""

import torch
from typing import Dict, Tuple
from collections import deque


class ActivationMonitor:
    """
    Moniteur de saturation des activations.

    Détecte:
    - Neurones saturés (toujours au max)
    - Neurones morts (toujours à 0)
    - Variance des activations
    """

    def __init__(
        self,
        saturation_threshold: float = 0.95,
        dead_threshold: float = 1e-6,
        history_size: int = 100
    ):
        self.saturation_threshold = saturation_threshold
        self.dead_threshold = dead_threshold
        self.history_size = history_size
        self.activation_history: Dict[str, deque] = {}

    def register_layer(self, name: str):
        """Enregistre une couche à monitorer."""
        if name not in self.activation_history:
            self.activation_history[name] = deque(maxlen=self.history_size)

    def record_activations(self, name: str, activations: torch.Tensor):
        """Enregistre les activations d'une couche."""
        if name not in self.activation_history:
            self.register_layer(name)

        with torch.no_grad():
            # Flatten si nécessaire
            act = activations.view(activations.size(0), -1)  # [batch, neurons]

            # Statistiques par neurone (moyennées sur le batch)
            neuron_means = act.mean(dim=0)  # [neurons]

            # Théorique max pour ReLU (estimation basée sur les données)
            estimated_max = act.max().item() if act.max().item() > 0 else 1.0

            stats = {
                'neuron_means': neuron_means.cpu(),
                'saturated_ratio': (neuron_means > self.saturation_threshold * estimated_max).float().mean().item(),
                'dead_ratio': (neuron_means < self.dead_threshold).float().mean().item(),
                'variance': act.var().item(),
                'mean': act.mean().item()
            }
            self.activation_history[name].append(stats)

    def get_saturation_metrics(self, name: str) -> Tuple[float, float, float]:
        """
        Retourne (saturation_ratio, dead_ratio, variance) pour une couche.
        """
        if name not in self.activation_history or len(self.activation_history[name]) == 0:
            return 0.0, 0.0, 1.0

        recent = list(self.activation_history[name])[-20:]

        avg_saturated = sum(s['saturated_ratio'] for s in recent) / len(recent)
        avg_dead = sum(s['dead_ratio'] for s in recent) / len(recent)
        avg_variance = sum(s['variance'] for s in recent) / len(recent)

        return avg_saturated, avg_dead, avg_variance
