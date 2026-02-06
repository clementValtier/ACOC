"""
ACOC - Router
=============
Central router that directs inputs to the appropriate experts/blocks.
With EWC protection against catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional, Tuple

from ..monitoring import GradientFlowMonitor


class Router(nn.Module):
    """
    Central router that directs inputs to the appropriate experts/blocks.

    With EWC protection against catastrophic forgetting.
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

        # Routing network (small MLP)
        self.routing_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_routes)
        )

        # Learnable bias to guide routing (e.g., favor CNN for images)
        self.route_bias = nn.Parameter(torch.zeros(num_routes))

        # EWC: Fisher information and old weights
        self.fisher_info: Optional[Dict[str, torch.Tensor]] = None
        self.old_params: Optional[Dict[str, torch.Tensor]] = None

        # Architecture awareness
        self.expert_types: Dict[int, str] = {}  # route index → expert type
        self.detected_data_type: Optional[str] = None

        # Monitoring
        self.gradient_monitor = GradientFlowMonitor()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input [batch_size, input_dim]

        Returns:
            (selected_indices, probabilities)
        """
        logits = self.routing_net(x) + self.route_bias  # Add bias
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
        Forward with forced exploration towards certain routes.
        Used for warmup after expansion.
        """
        logits = self.routing_net(x) + self.route_bias
        probabilities = F.softmax(logits, dim=-1)

        if force_route is not None and exploration_prob > 0:
            # With probability exploration_prob, force towards force_route
            batch_size = x.size(0)
            mask = torch.rand(batch_size, device=x.device) < exploration_prob
            selected = probabilities.argmax(dim=-1)
            selected[mask] = force_route
        else:
            selected = probabilities.argmax(dim=-1)

        return selected, probabilities

    def add_route(self, device: torch.device | None = None):
        """Adds a new route (for a new block)."""
        if device is None:
            device = next(self.parameters()).device

        old_out = self.routing_net[-1]
        new_out = nn.Linear(old_out.in_features, self.num_routes + 1).to(device)  # type: ignore[arg-type]

        # Copy old weights
        with torch.no_grad():
            new_out.weight[:self.num_routes] = old_out.weight  # type: ignore[index, assignment]
            new_out.bias[:self.num_routes] = old_out.bias  # type: ignore[index, assignment]
            # Initialize new route with small weights
            nn.init.xavier_uniform_(new_out.weight[self.num_routes:self.num_routes+1])
            new_out.bias[self.num_routes] = 0.0

        self.routing_net[-1] = new_out
        self.num_routes += 1

        # Extend route_bias for the new route
        with torch.no_grad():
            new_bias = torch.zeros(self.num_routes, device=device)
            new_bias[:self.num_routes-1] = self.route_bias
            new_bias[-1] = 0.0  # Neutral bias for the new route
            self.route_bias = nn.Parameter(new_bias)

        # Invalidate EWC (old parameters are no longer valid)
        self.fisher_info = None
        self.old_params = None

    def compute_fisher(self, data_loader: Iterable, num_samples: int = 500):
        """
        Computes the Fisher information matrix for EWC.
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
            # Log-likelihood of the prediction
            log_probs = torch.log(probs + 1e-8)
            selected = probs.argmax(dim=-1)
            loss = -log_probs.gather(1, selected.unsqueeze(1)).mean()
            loss.backward()

            for n, p in self.named_parameters():
                if p.grad is not None:
                    self.fisher_info[n] += p.grad.pow(2)

            count += x.size(0)

        # Normalize
        for n in self.fisher_info:
            self.fisher_info[n] /= max(count, 1)

        self.train()

    def ewc_loss(self, lambda_ewc: float = 100.0) -> torch.Tensor:
        """Computes the EWC penalty."""
        device = next(self.parameters()).device

        if self.fisher_info is None or self.old_params is None:
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        for n, p in self.named_parameters():
            if n in self.fisher_info:
                loss += (self.fisher_info[n] * (p - self.old_params[n]).pow(2)).sum()

        return lambda_ewc * 0.5 * loss

    # Mapping: detected data type → matching expert types
    _DATA_TYPE_TO_EXPERT: Dict[str, str] = {
        "image": "cnn",
        "text": "mlp",
        "audio": "audio_mlp",
    }

    def set_expert_types(self, expert_types: Dict[int, str]):
        """Register the mapping from route index to expert type string."""
        self.expert_types = expert_types

    def get_matching_indices(self, data_type: Optional[str] = None) -> list[int]:
        """Return route indices whose expert type matches the given data type."""
        if data_type is None or not self.expert_types:
            return []
        target_expert = self._DATA_TYPE_TO_EXPERT.get(data_type)
        if target_expert is None:
            return []
        return [i for i, et in self.expert_types.items() if et == target_expert]

    def update_load_balance(self, routing_counts: Dict[int, int], alpha: float = 0.01):
        """
        Architecture-aware load balancing.
        When a data type is detected and expert types are known, the target
        distribution favours matching blocks (~70%) over non-matching ones.
        Falls back to uniform 1/N when no type info is available.
        """
        total = sum(routing_counts.values())
        if total == 0:
            return

        # Compute target distribution
        matching = self.get_matching_indices(self.detected_data_type)

        if matching and len(matching) < self.num_routes:
            # Architecture-aware: 70% to matching blocks, 30% to others
            n_match = len(matching)
            n_other = self.num_routes - n_match
            targets: Dict[int, float] = {}
            for idx in range(self.num_routes):
                if idx in matching:
                    targets[idx] = 0.7 / n_match
                else:
                    targets[idx] = 0.3 / n_other
        else:
            # No type info or all blocks match → uniform
            targets = {idx: 1.0 / self.num_routes for idx in range(self.num_routes)}

        with torch.no_grad():
            for idx in range(self.num_routes):
                load_i = routing_counts.get(idx, 0) / total
                self.route_bias[idx] -= alpha * (load_i - targets[idx])
            self.route_bias.data.clamp_(-2.0, 2.0)

    def set_route_bias(self, route_idx: int, bias_value: float):
        """
        Args:
            route_idx: Index de la route à favoriser
            bias_value: Valeur du biais (positif = favorise, négatif = défavorise)
        """
        with torch.no_grad():
            self.route_bias[route_idx] = bias_value

    def detect_data_type(self, x: torch.Tensor) -> str:
        """
        Detects data type by analyzing dimension first,
        then statistical characteristics.  The result is also stored in
        ``self.detected_data_type`` so the load balancer can use it.

        Args:
            x: Data batch [batch, features]

        Returns:
            "image", "text", or "audio"
        """
        with torch.no_grad():
            input_dim = x.shape[-1]

            import math

            for channels in [1, 3, 4]:
                size_float = math.sqrt(input_dim / channels)
                size = int(size_float)
                if abs(size_float - size) < 0.01 and size * size * channels == input_dim:
                    self.detected_data_type = "image"
                    return "image"

            mean_val = x.mean().item()
            std_val = x.std().item()
            min_val = x.min().item()
            max_val = x.max().item()
            sparsity = (x.abs() < 1e-6).float().mean().item()

            if sparsity > 0.7 and min_val >= 0:
                self.detected_data_type = "text"
                return "text"

            if -3.0 < min_val < 3.0 and -3.0 < max_val < 3.0 and 0.5 < std_val < 2.5:
                self.detected_data_type = "image"
                return "image"

            elif abs(mean_val) < 0.2 and std_val > 0.3 and sparsity < 0.5:
                self.detected_data_type = "text"
                return "text"

            else:
                self.detected_data_type = "audio"
                return "audio"

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
