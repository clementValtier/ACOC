"""
ACOC - Router
=============
Routeur central qui dirige les inputs vers les bons experts/blocs.
Avec protection EWC contre le catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from ..monitoring import GradientFlowMonitor


class Router(nn.Module):
    """
    Routeur central qui dirige les inputs vers les bons experts/blocs.

    Avec protection EWC contre le catastrophic forgetting.
    """

    def __init__(
        self,
        input_dim: int,
        num_routes: int,
        hidden_dim: int = 128
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_routes = num_routes

        # Réseau de routage (petit MLP)
        self.routing_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_routes)
        )

        # EWC: Fisher information et anciens poids
        self.fisher_info: Optional[Dict[str, torch.Tensor]] = None
        self.old_params: Optional[Dict[str, torch.Tensor]] = None

        # Monitoring
        self.gradient_monitor = GradientFlowMonitor()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch_size, input_dim]

        Returns:
            (selected_indices, probabilities)
        """
        logits = self.routing_net(x)
        probabilities = F.softmax(logits, dim=-1)
        selected = probabilities.argmax(dim=-1)

        return selected, probabilities

    def forward_with_exploration(
        self,
        x: torch.Tensor,
        force_route: Optional[int] = None,
        exploration_prob: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward avec exploration forcée vers certaines routes.
        Utilisé pour le warmup après expansion.
        """
        logits = self.routing_net(x)
        probabilities = F.softmax(logits, dim=-1)

        if force_route is not None and exploration_prob > 0:
            # Avec probabilité exploration_prob, forcer vers force_route
            batch_size = x.size(0)
            mask = torch.rand(batch_size, device=x.device) < exploration_prob
            selected = probabilities.argmax(dim=-1)
            selected[mask] = force_route
        else:
            selected = probabilities.argmax(dim=-1)

        return selected, probabilities

    def add_route(self, device: torch.device = None):
        """Ajoute une nouvelle route (pour un nouveau bloc)."""
        if device is None:
            device = next(self.parameters()).device

        old_out = self.routing_net[-1]
        new_out = nn.Linear(old_out.in_features, self.num_routes + 1).to(device)

        # Copier les anciens poids
        with torch.no_grad():
            new_out.weight[:self.num_routes] = old_out.weight
            new_out.bias[:self.num_routes] = old_out.bias
            # Initialiser la nouvelle route avec de petits poids
            nn.init.xavier_uniform_(new_out.weight[self.num_routes:self.num_routes+1])
            new_out.bias[self.num_routes] = 0.0

        self.routing_net[-1] = new_out
        self.num_routes += 1

        # Invalider EWC (les anciens paramètres ne sont plus valides)
        self.fisher_info = None
        self.old_params = None

    def compute_fisher(self, data_loader, num_samples: int = 500):
        """
        Calcule la Fisher information matrix pour EWC.
        """
        device = next(self.parameters()).device
        self.fisher_info = {n: torch.zeros_like(p) for n, p in self.named_parameters()}
        self.old_params = {n: p.clone().detach() for n, p in self.named_parameters()}

        self.eval()
        count = 0

        for batch in data_loader:
            if count >= num_samples:
                break

            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            x = x.to(device)
            self.zero_grad()

            _, probs = self.forward(x)
            # Log-likelihood de la prédiction
            log_probs = torch.log(probs + 1e-8)
            selected = probs.argmax(dim=-1)
            loss = -log_probs.gather(1, selected.unsqueeze(1)).mean()
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)

            count += x.size(0)

        # Normaliser
        for n in self.fisher_info:
            self.fisher_info[n] /= max(count, 1)

        self.train()

    def ewc_loss(self, lambda_ewc: float = 100.0) -> torch.Tensor:
        """Calcule la pénalité EWC."""
        device = next(self.parameters()).device

        if self.fisher_info is None or self.old_params is None:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        for n, p in self.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.old_params[n]).pow(2)).sum()

        return lambda_ewc * 0.5 * loss

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
