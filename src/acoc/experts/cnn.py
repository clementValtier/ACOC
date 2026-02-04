"""
ACOC - CNN Expert
=================
Convolutional network with dynamic head.
"""

import torch
import torch.nn as nn
import math
from .base import BaseExpert

class CNNExpert(BaseExpert):
    def __init__(self, input_dim, hidden_dim, output_dim, name, config):
        super().__init__(input_dim, hidden_dim, output_dim, name, config)
        self.expert_type = "cnn"

        # 1. Automatic detection of image dimensions
        in_channels, img_size = self._detect_image_shape(input_dim, config)
        self.in_channels = in_channels
        self.img_size = img_size

        # 2. Build convolutional layers
        layers = []
        current_size = img_size

        cnn_channels = config.cnn_channels if config else [32, 64]

        for out_channels in cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
            current_size //= 2
            
        self.features = nn.Sequential(*layers)
        self.flatten_dim = in_channels * current_size * current_size
        
        # 3. Build head (expandable part)
        # Note: fc1 takes flatten_dim as input, not input_dim
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # He initialization for better convergence with ReLU
        self._init_weights()

        self._register_hooks()

    def _init_weights(self):
        """Initialize weights using He initialization for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def _detect_image_shape(self, input_dim, config):
        """
        Automatically detect the number of channels and image size.
        Tries common combinations: 1 (grayscale), 3 (RGB), 4 (RGBA).
        """
        # Prefer config if specified
        if config and hasattr(config, 'image_channels'):
            channels = config.image_channels
            size = int(math.sqrt(input_dim / channels))
            # Verify consistency
            if size * size * channels == input_dim:
                return channels, size

        # Otherwise, test common combinations
        for channels in [1, 3, 4]:
            size_float = math.sqrt(input_dim / channels)
            size = int(size_float)

            # Verify it's a perfect square
            if abs(size_float - size) < 0.01 and size * size * channels == input_dim:
                return channels, size

        # Fallback: assume grayscale and take the closest square root
        size = int(math.sqrt(input_dim))
        return 1, size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dynamic reshape: [Batch, Pixels] -> [Batch, C, H, W]
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, self.in_channels, self.img_size, self.img_size)

        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        hidden = self.activation(self.fc1(x))
        return self.fc2(hidden)

    def expand_width(self, additional_neurons: int):
        # Expansion is done on the head (fc1/fc2), like MLP
        # Logic is identical to MLP, except fc1 input is flatten_dim
        if additional_neurons <= 0: return
        device = self.fc1.weight.device
        
        with torch.no_grad():
            indices = torch.randint(0, self.hidden_dim, (additional_neurons,))
            
            # FC1 (Input = flatten_dim)
            new_fc1 = nn.Linear(self.flatten_dim, self.hidden_dim + additional_neurons).to(device)
            # Copy and noise (identical to MLP)
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.weight[self.hidden_dim:] = self.fc1.weight[indices]
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc1.bias[self.hidden_dim:] = self.fc1.bias[indices]
            
            noise = 0.001 * self.fc1.weight.std().item() if self.fc1.weight.std().item() > 0 else 0.001
            new_fc1.weight[self.hidden_dim:] += torch.randn_like(new_fc1.weight[self.hidden_dim:]) * noise

            # FC2 (Input = hidden + additional)
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
        # Hook on weight gradient instead of module
        if self.fc1.weight.requires_grad:
            self._hooks.append(self.fc1.weight.register_hook(save_grad))

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())