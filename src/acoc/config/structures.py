"""
ACOC - Data Structures
============================
Definition of dataclasses and enums used throughout the system.
PyTorch compatible.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    import torch.nn as nn


class TaskType(Enum):
    """Task types supported by the system."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    GENERIC = "generic"


@dataclass
class SaturationMetrics:
    """
    Detailed saturation metrics for an expert/block.
    Based on gradient flow and activation analysis.
    """
    # Gradient flow: ratio of "living" gradients (> threshold)
    gradient_flow_ratio: float = 1.0

    # Activation saturation: ratio of saturated neurons (> 0.95 * max)
    activation_saturation: float = 0.0

    # Ratio of "dead" neurons (always 0 after ReLU)
    dead_neuron_ratio: float = 0.0

    # Inter-batch variance of activations (low = saturated)
    activation_variance: float = 1.0

    # Combined saturation score (0 = healthy, 1 = saturated)
    combined_score: float = 0.0

    def compute_combined_score(self) -> float:
        """
        Compute combined saturation score.

        Logic:
        - low gradient_flow_ratio = blocked
        - high activation_saturation = neurons at max
        - high dead_neuron_ratio = unused neurons
        - low activation_variance = no diversity
        """
        # Weights of different factors
        w_gradient = 0.35
        w_activation = 0.25
        w_dead = 0.20
        w_variance = 0.20

        # Normalize: we want a score where 1 = problematic
        gradient_problem = 1.0 - self.gradient_flow_ratio
        variance_problem = 1.0 / (1.0 + self.activation_variance)

        self.combined_score = (
            w_gradient * gradient_problem +
            w_activation * self.activation_saturation +
            w_dead * self.dead_neuron_ratio +
            w_variance * variance_problem
        )

        return self.combined_score


@dataclass
class TaskBlock:
    """
    A block specialized for a task or sub-task.
    Contains layers (experts) and usage metadata.
    """
    id: str
    task_type: TaskType
    num_params: int
    layers: List[Any]  # nn.ModuleList in PyTorch
    creation_cycle: int
    usage_count: int = 0
    last_used_cycle: int = 0

    # Recent usage history (list of N last cycles)
    recent_usage: List[int] = field(default_factory=list)

    # Detailed saturation metrics
    saturation: SaturationMetrics = field(default_factory=SaturationMetrics)

    def get_param_count(self) -> int:
        return self.num_params

    def update_param_count(self):
        """Recalculate parameter count based on layers."""
        self.num_params = sum(
            p.numel() for layer in self.layers
            for p in layer.parameters()
        )


@dataclass
class ModelMetrics:
    """
    Metrics collected during training.
    Used for expansion decisions.
    """
    loss_history: List[float] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    gradient_norms: Dict[str, List[float]] = field(default_factory=dict)
    expert_utilization: Dict[str, float] = field(default_factory=dict)
    validation_accuracy: float = 0.0
    saturation_scores: Dict[str, float] = field(default_factory=dict)

    # Detailed saturation metrics
    detailed_saturation: Dict[str, SaturationMetrics] = field(default_factory=dict)

    def add_loss(self, loss: float):
        self.loss_history.append(loss)

    def add_validation_score(self, score: float):
        self.validation_scores.append(score)

    def get_recent_loss_trend(self, window: int = 10) -> Optional[float]:
        """Returns relative loss improvement over the last N cycles."""
        if len(self.loss_history) < window:
            return None
        recent = self.loss_history[-window:]
        if recent[0] == 0:
            return 0.0
        return (recent[0] - recent[-1]) / (recent[0] + 1e-8)

    def get_relative_performance_threshold(self, lookback: int = 5) -> float:
        """
        Compute relative performance threshold based on history.
        Used for variant voting.
        """
        if len(self.validation_scores) < lookback:
            # Not enough history, use default low threshold
            return 0.3

        recent_avg = sum(self.validation_scores[-lookback:]) / lookback
        # Threshold is 95% of recent average
        return recent_avg * 0.95


@dataclass
class ExpansionDecision:
    """Result of an expansion decision."""
    should_expand: bool
    target_block_id: Optional[str] = None
    expansion_type: str = "none"  # "width", "depth", "new_block"
    confidence: float = 0.0
    reason: str = ""


@dataclass
class TrainingLog:
    """Log of a training cycle."""
    cycle: int
    avg_loss: Optional[float]
    total_params: int
    num_blocks: int
    expanded: bool
    expansion_type: Optional[str] = None
    expansion_target: Optional[str] = None
    warmup_active: bool = False
    saturation_details: Optional[Dict[str, float]] = None
