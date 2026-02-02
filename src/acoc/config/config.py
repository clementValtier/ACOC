"""
ACOC - Configuration Système
============================
Configuration globale du système ACOC.
Tous les hyperparamètres sont centralisés ici.
"""

from dataclasses import dataclass


@dataclass
class SystemConfig:
    """
    Configuration globale du système ACOC.
    Tous les hyperparamètres sont centralisés ici.
    """
    # === Timing ===
    training_cycle_duration: int = 300  # secondes (5 min)
    checkpoint_interval: int = 1  # cycles entre checkpoints

    # === Pénalités (Double Malus) ===
    alpha_global_penalty: float = 0.01  # Pénalité taille globale (log)
    beta_task_penalty: float = 0.05     # Pénalité par tâche (quadratique)
    task_param_threshold: int = 1_000_000  # Seuil par tâche avant pénalité

    # === Expansion ===
    saturation_threshold: float = 0.6   # Trigger si score combiné > 60%
    min_cycles_before_expand: int = 3   # Attendre au moins 3 cycles
    expansion_cooldown: int = 5         # Cycles minimum entre expansions
    expansion_ratio: float = 0.1        # Ajouter 10% de neurones

    # === Variantes (Model Soups) ===
    num_variants: int = 5
    delta_magnitude: float = 0.01       # Amplitude des perturbations
    top_k_merge: int = 3                # Nombre de variantes à fusionner
    # NOUVEAU: Seuil relatif au lieu de fixe
    performance_threshold_ratio: float = 0.95  # Expand si < 95% de la moyenne récente

    # === Warmup après expansion ===
    warmup_steps: int = 50              # Steps de warmup après expansion
    warmup_lr_multiplier: float = 5.0   # LR multiplié pour nouveaux params
    new_block_exploration_prob: float = 0.1  # Prob de forcer vers nouveau bloc (réduit de 0.3)
    new_block_exploration_cycles: int = 3    # Cycles d'exploration forcée
    max_warmup_cycles: int = 10         # Cycles max avant désactivation forcée du warmup

    # === Pruning / Consolidation ===
    prune_unused_after_cycles: int = 20
    consolidation_similarity_threshold: float = 0.9
    maintenance_interval: int = 5       # Maintenance tous les N cycles

    # === Architecture initiale ===
    input_dim: int = 256
    hidden_dim: int = 512
    output_dim: int = 256

    # === Métriques de saturation ===
    gradient_flow_threshold: float = 1e-6  # Gradient considéré "mort" si < seuil
    activation_saturation_threshold: float = 0.95  # Activation saturée si > 95% max
    dead_neuron_threshold: float = 1e-6    # Neurone mort si activation < seuil

    # === Device ===
    device: str = "cuda"  # "cuda" ou "cpu"

    # === Loss function ===
    use_cross_entropy: bool = False  # True pour classification (vs MSE pour régression)
