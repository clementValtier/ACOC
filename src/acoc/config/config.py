"""
ACOC - System Configuration
============================
Global configuration for the ACOC system.
All hyperparameters are centralized here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SystemConfig:
    """
    Global configuration for the ACOC system.
    All hyperparameters are centralized here.
    """
    # === Timing ===
    training_cycle_duration: int = 300  # seconds (5 min)
    checkpoint_interval: int = 1  # cycles between checkpoints

    # === Penalties (Double penalty) ===
    alpha_global_penalty: float = 0.01  # global size penalty (log)
    beta_task_penalty: float = 0.05     # per-task penalty (quadratic)
    task_param_threshold: int = 1_000_000  # threshold per task before penalty

    # === Expansion ===
    saturation_threshold: float = 0.6   # trigger if combined score > 60%
    min_cycles_before_expand: int = 3   # wait at least 3 cycles
    expansion_cooldown: int = 5         # minimum cycles between expansions
    expansion_ratio: float = 0.1        # add 10% of neurons
    recent_usage_window: int = 5        # window for recent usage

    # === Variants (Model Soups) ===
    num_variants: int = 5
    delta_magnitude: float = 0.01       # amplitude of perturbations
    top_k_merge: int = 3                # number of variants to merge
    performance_threshold_ratio: float = 0.95  # expand if < 95% of recent average

    # === Warmup after expansion ===
    warmup_steps: int = 50              # warmup steps after expansion
    warmup_lr_multiplier: float = 5.0   # LR multiplied for new params
    new_block_exploration_prob: float = 0.1  # probability to force towards new block
    new_block_exploration_cycles: int = 3    # forced exploration cycles
    max_warmup_cycles: int = 10         # max cycles before forced deactivation

    # === Pruning / Consolidation ===
    prune_unused_after_cycles: int = 20
    consolidation_similarity_threshold: float = 0.9
    maintenance_interval: int = 5       # maintenance every N cycles

    # === Initial architecture ===
    input_dim: int = 256
    hidden_dim: int = 512
    output_dim: int = 256

    # === CNN Configuration ===
    use_cnn: bool = True               # enable CNN for images
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128]) # CNN architecture
    image_channels: int = 3            # 3 for RGB, 1 for grayscale

    # === Saturation metrics ===
    gradient_flow_threshold: float = 1e-6
    activation_saturation_threshold: float = 0.95
    dead_neuron_threshold: float = 1e-6

    # === Device ===
    device: str = "cuda"  # "cuda" or "cpu"

    # === Load Balancing ===
    load_balance_alpha: float = 0.01  # dynamic bias adjustment rate for router load balancing

    # === Replay ===
    replay_loss_weight: float = 1.5  # weight applied to replay loss (>1 = stronger anti-forgetting)

    # === Loss function ===
    use_cross_entropy: bool = True