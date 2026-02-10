"""
ACOC - Modality Projection System
==================================
Manages projection layers for different input modalities and dimensions.
Enables continual learning across heterogeneous data sources.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModalityConfig:
    """Configuration for a specific modality/input type."""
    name: str
    input_dim: int
    modality_type: str  # 'image', 'text', 'audio', etc.
    metadata: Optional[Dict] = None  # Additional info (image_size, channels, etc.)


class ProjectionLayer(nn.Module):
    """
    Learnable projection layer to map input from native dimension to unified space.

    Uses a bottleneck architecture with residual connection when possible.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Native input dimension
            output_dim: Target unified dimension
            hidden_dim: Bottleneck hidden dimension (default: mean of input/output)
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Auto-compute hidden dim if not provided
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2

        # Bottleneck architecture: input -> hidden -> output
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(hidden_dim, output_dim)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(output_dim)

        # Residual connection if dimensions match
        self.use_residual = (input_dim == output_dim)
        if not self.use_residual and input_dim < output_dim:
            # Padding for smaller inputs
            self.residual_proj = nn.Linear(input_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        nn.init.xavier_uniform_(self.down_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.up_proj.weight, gain=0.1)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to unified dimension.

        Args:
            x: Input tensor of shape (batch, input_dim)

        Returns:
            Projected tensor of shape (batch, output_dim)
        """
        identity = x

        # Bottleneck projection
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)

        # Residual connection
        if self.use_residual:
            x = x + identity
        elif hasattr(self, 'residual_proj'):
            x = x + self.residual_proj(identity)

        # Normalize
        x = self.layer_norm(x)

        return x


class ModalityProjector(nn.Module):
    """
    Manages multiple projection layers for different modalities.

    Key features:
    - Register modalities with native dimensions
    - Automatic projection to unified space
    - Modality-specific adapters
    - Support for continual learning (add new modalities on-the-fly)
    """

    def __init__(self, unified_dim: int, device: torch.device | None = None):
        """
        Args:
            unified_dim: Target unified dimension for all modalities
            device: Device to place projections on
        """
        super().__init__()

        self.unified_dim = unified_dim
        self.device = device or torch.device('cpu')

        # Registry of modalities
        self.modalities: Dict[str, ModalityConfig] = {}

        # Projection layers (as ModuleDict for proper parameter tracking)
        self.projections = nn.ModuleDict()

        # Task tokens: learnable embeddings to condition on task/modality
        self.task_tokens = nn.ParameterDict()

        # Statistics for each modality (for tracking)
        self.modality_stats: Dict[str, Dict] = {}

    def register_modality(
        self,
        name: str,
        input_dim: int,
        modality_type: str,
        metadata: Optional[Dict] = None,
        force: bool = False
    ):
        """
        Register a new modality with its projection layer.

        Args:
            name: Unique identifier for this modality (e.g., 'mnist_28x28')
            input_dim: Native input dimension
            modality_type: Type of modality ('image', 'text', 'audio')
            metadata: Additional metadata
            force: If True, overwrite existing modality
        """
        if name in self.modalities and not force:
            raise ValueError(f"Modality '{name}' already registered. Use force=True to overwrite.")

        # Create config
        config = ModalityConfig(
            name=name,
            input_dim=input_dim,
            modality_type=modality_type,
            metadata=metadata or {}
        )

        self.modalities[name] = config

        # Create projection layer if needed
        if input_dim != self.unified_dim:
            projection = ProjectionLayer(
                input_dim=input_dim,
                output_dim=self.unified_dim
            ).to(self.device)
            self.projections[name] = projection

        # Create task token (learnable embedding)
        task_token = nn.Parameter(
            torch.randn(self.unified_dim, device=self.device) * 0.02
        )
        self.task_tokens[name] = task_token

        # Initialize statistics
        self.modality_stats[name] = {
            'num_samples': 0,
            'activation_mean': 0.0,
            'activation_std': 1.0
        }

        print(f"  ✓ Registered modality '{name}': {input_dim} → {self.unified_dim}")

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        add_task_token: bool = True
    ) -> torch.Tensor:
        """
        Project input through modality-specific projection.

        Args:
            x: Input tensor of shape (batch, native_dim)
            modality: Name of the modality
            add_task_token: Whether to add task token embedding

        Returns:
            Projected tensor of shape (batch, unified_dim)
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality '{modality}'. Register it first.")

        config = self.modalities[modality]

        # Project if needed
        if config.input_dim != self.unified_dim:
            if modality not in self.projections:
                raise RuntimeError(f"Projection for '{modality}' not found.")
            x = self.projections[modality](x)

        # Add task token for task conditioning
        if add_task_token and modality in self.task_tokens:
            task_emb = self.task_tokens[modality].unsqueeze(0)  # (1, unified_dim)
            x = x + task_emb  # Broadcasting

        # Update statistics
        with torch.no_grad():
            self.modality_stats[modality]['num_samples'] += x.size(0)

        return x

    def get_modality_info(self, modality: str) -> ModalityConfig:
        """Get configuration for a modality."""
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality '{modality}'")
        return self.modalities[modality]

    def list_modalities(self) -> list[str]:
        """List all registered modalities."""
        return list(self.modalities.keys())

    def get_projection_params(self, modality: str) -> int:
        """Get number of parameters in projection for a modality."""
        if modality not in self.projections:
            return 0
        return sum(p.numel() for p in self.projections[modality].parameters())

    def freeze_projections(self, modalities: Optional[list[str]] = None):
        """
        Freeze projection layers to prevent forgetting.

        Args:
            modalities: List of modalities to freeze (None = all)
        """
        modalities = modalities or list(self.projections.keys())

        for modality in modalities:
            if modality in self.projections:
                for param in self.projections[modality].parameters():
                    param.requires_grad = False
                print(f"  ✓ Frozen projection for '{modality}'")

    def unfreeze_projections(self, modalities: Optional[list[str]] = None):
        """
        Unfreeze projection layers.

        Args:
            modalities: List of modalities to unfreeze (None = all)
        """
        modalities = modalities or list(self.projections.keys())

        for modality in modalities:
            if modality in self.projections:
                for param in self.projections[modality].parameters():
                    param.requires_grad = True

    def summary(self) -> str:
        """Generate summary of registered modalities."""
        lines = ["Modality Projector Summary:"]
        lines.append(f"  Unified dimension: {self.unified_dim}")
        lines.append(f"  Registered modalities: {len(self.modalities)}")
        lines.append("")

        for name, config in self.modalities.items():
            params = self.get_projection_params(name)
            stats = self.modality_stats[name]
            lines.append(
                f"  {name:20s} | {config.input_dim:5d} → {self.unified_dim:5d} | "
                f"{params:,} params | {stats['num_samples']} samples"
            )

        return "\n".join(lines)
