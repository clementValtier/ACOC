"""
ACOC - MLP Expert
=================
Implémentation standard Fully Connected.
"""

import torch
import torch.nn as nn
from .base import BaseExpert

class MLPExpert(BaseExpert):
    def __init__(self, input_dim, hidden_dim, output_dim, name, config=None):
        super().__init__(input_dim, hidden_dim, output_dim, name, config)
        self.expert_type = "mlp"
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self._register_hooks()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Aplatir si nécessaire (au cas où on reçoit une image sans CNN)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
            
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)

    def expand_width(self, additional_neurons: int):
        if additional_neurons <= 0: return
        device = self.fc1.weight.device
        
        with torch.no_grad():
            indices = torch.randint(0, self.hidden_dim, (additional_neurons,))
            
            # Expansion FC1
            new_fc1 = nn.Linear(self.input_dim, self.hidden_dim + additional_neurons).to(device)
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.weight[self.hidden_dim:] = self.fc1.weight[indices]
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc1.bias[self.hidden_dim:] = self.fc1.bias[indices]
            
            # Bruit
            noise = 0.001 * self.fc1.weight.std().item() if self.fc1.weight.std().item() > 0 else 0.001
            new_fc1.weight[self.hidden_dim:] += torch.randn_like(new_fc1.weight[self.hidden_dim:]) * noise

            # Expansion FC2
            new_fc2 = nn.Linear(self.hidden_dim + additional_neurons, self.output_dim).to(device)
            old_w = self.fc2.weight.clone()
            old_w[:, indices] /= 2
            
            new_fc2.weight[:, :self.hidden_dim] = old_w
            new_fc2.weight[:, self.hidden_dim:] = self.fc2.weight[:, indices] / 2
            new_fc2.bias = nn.Parameter(self.fc2.bias.clone())

            self.fc1 = new_fc1
            self.fc2 = new_fc2
            self.hidden_dim += additional_neurons
            self._register_hooks()

    def _register_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []

        def save_hidden(m, i, o):
            self.activation_monitor.record_activations(f"{self.name}_hidden", o.detach())

        def save_grad(grad):
            if grad is not None:
                self.gradient_monitor.record_gradients(f"{self.name}_fc1", grad.detach())

        self._hooks.append(self.fc1.register_forward_hook(save_hidden))
        # Hook sur le gradient du poids au lieu du module
        if self.fc1.weight.requires_grad:
            self._hooks.append(self.fc1.weight.register_hook(save_grad))

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())