"""
ACOC - Expert
=============
Expert (MLP) avec monitoring des activations et gradients.
Supporte l'expansion Net2Net.
"""

import torch
import torch.nn as nn
from typing import List, Optional

from ..monitoring import ActivationMonitor, GradientFlowMonitor
from ..config import SaturationMetrics


class Expert(nn.Module):
    """
    Un expert (MLP) avec monitoring des activations et gradients.
    Supporte l'expansion Net2Net.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        name: str = "expert"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.name = name

        # Couches
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation
        self.activation = nn.ReLU()

        # Monitoring
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()

        # Buffer pour les activations
        self._last_hidden: Optional[torch.Tensor] = None
        self._hooks = []

        # Hook pour le gradient
        self._register_hooks()

    def _register_hooks(self):
        """Enregistre les hooks pour le monitoring."""
        # Supprimer les anciens hooks
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

        def save_hidden_hook(module, input, output):
            self._last_hidden = output.detach()
            self.activation_monitor.record_activations(f"{self.name}_hidden", output.detach())

        def save_gradient_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradient_monitor.record_gradients(
                    f"{self.name}_fc1",
                    grad_output[0].detach()
                )

        h1 = self.fc1.register_forward_hook(save_hidden_hook)
        h2 = self.fc1.register_full_backward_hook(save_gradient_hook)
        self._hooks = [h1, h2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        hidden = self.activation(self.fc1(x))
        output = self.fc2(hidden)
        return output

    def get_saturation_metrics(self) -> SaturationMetrics:
        """
        Calcule les métriques de saturation complètes.
        """
        metrics = SaturationMetrics()

        # Gradient flow
        metrics.gradient_flow_ratio = self.gradient_monitor.get_flow_ratio(f"{self.name}_fc1")

        # Activation saturation
        sat, dead, var = self.activation_monitor.get_saturation_metrics(f"{self.name}_hidden")
        metrics.activation_saturation = sat
        metrics.dead_neuron_ratio = dead
        metrics.activation_variance = var

        # Score combiné
        metrics.compute_combined_score()

        return metrics

    def expand_width(self, additional_neurons: int):
        """
        Expansion Net2Net en largeur.
        Préserve la fonction du réseau.
        """
        if additional_neurons <= 0:
            return

        device = self.fc1.weight.device

        with torch.no_grad():
            # Sélectionner des neurones à dupliquer
            indices = torch.randint(0, self.hidden_dim, (additional_neurons,))

            # === Expansion fc1 (colonnes = neurones de sortie) ===
            new_fc1 = nn.Linear(self.input_dim, self.hidden_dim + additional_neurons).to(device)
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.weight[self.hidden_dim:] = self.fc1.weight[indices]
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc1.bias[self.hidden_dim:] = self.fc1.bias[indices]

            # Ajouter du bruit pour casser la symétrie
            noise_scale = 0.001 * self.fc1.weight.std().item()
            new_fc1.weight[self.hidden_dim:] += torch.randn_like(
                new_fc1.weight[self.hidden_dim:]
            ) * noise_scale

            # === Expansion fc2 (lignes = neurones d'entrée) ===
            new_fc2 = nn.Linear(self.hidden_dim + additional_neurons, self.output_dim).to(device)

            # D'abord copier fc2.weight avec division pour les indices dupliqués
            old_weight = self.fc2.weight.clone()
            old_weight[:, indices] /= 2  # Diviser les colonnes dupliquées

            new_fc2.weight[:, :self.hidden_dim] = old_weight
            new_fc2.weight[:, self.hidden_dim:] = self.fc2.weight[:, indices] / 2
            new_fc2.bias = nn.Parameter(self.fc2.bias.clone())

            # Remplacer les couches
            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self.hidden_dim += additional_neurons

            # Ré-enregistrer les hooks
            self._register_hooks()

    def reset_monitors(self):
        """Réinitialise les moniteurs (après expansion)."""
        self.activation_monitor = ActivationMonitor()
        self.gradient_monitor = GradientFlowMonitor()
        self._register_hooks()

    def get_param_count(self) -> int:
        """Retourne le nombre de paramètres."""
        return sum(p.numel() for p in self.parameters())


class ExpertBlock(nn.Module):
    """
    Un bloc contenant potentiellement plusieurs experts en séquence.
    """

    def __init__(
        self,
        experts: List[Expert],
        name: str = "block"
    ):
        super().__init__()
        self.name = name
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for expert in self.experts:
            x = expert(x)
        return x

    def get_combined_saturation(self) -> SaturationMetrics:
        """Retourne les métriques combinées de tous les experts."""
        if not self.experts:
            return SaturationMetrics()

        all_metrics = [e.get_saturation_metrics() for e in self.experts]

        combined = SaturationMetrics()
        n = len(all_metrics)

        combined.gradient_flow_ratio = sum(m.gradient_flow_ratio for m in all_metrics) / n
        combined.activation_saturation = sum(m.activation_saturation for m in all_metrics) / n
        combined.dead_neuron_ratio = sum(m.dead_neuron_ratio for m in all_metrics) / n
        combined.activation_variance = sum(m.activation_variance for m in all_metrics) / n
        combined.compute_combined_score()

        return combined

    def expand_all_experts(self, ratio: float = 0.1):
        """Expand tous les experts du bloc."""
        for expert in self.experts:
            additional = max(1, int(expert.hidden_dim * ratio))
            expert.expand_width(additional)

    def reset_all_monitors(self):
        """Reset tous les moniteurs du bloc."""
        for expert in self.experts:
            expert.reset_monitors()

    def get_param_count(self) -> int:
        return sum(e.get_param_count() for e in self.experts)
