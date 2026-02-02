"""
ACOC - Structures de Données
============================
Définition des dataclasses et enums utilisés dans tout le système.
Compatible PyTorch.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    import torch.nn as nn


class TaskType(Enum):
    """Types de tâches supportées par le système."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    GENERIC = "generic"


@dataclass
class SaturationMetrics:
    """
    Métriques détaillées de saturation d'un expert/bloc.
    Basé sur l'analyse du gradient flow et des activations.
    """
    # Gradient flow: ratio de gradients "vivants" (> threshold)
    gradient_flow_ratio: float = 1.0
    
    # Saturation des activations: ratio de neurones saturés (> 0.95 * max)
    activation_saturation: float = 0.0
    
    # Ratio de neurones "morts" (toujours à 0 après ReLU)
    dead_neuron_ratio: float = 0.0
    
    # Variance inter-batch des activations (faible = saturé)
    activation_variance: float = 1.0
    
    # Score combiné de saturation (0 = sain, 1 = saturé)
    combined_score: float = 0.0
    
    def compute_combined_score(self) -> float:
        """
        Calcule un score combiné de saturation.
        
        Logique:
        - gradient_flow_ratio bas = bloqué
        - activation_saturation haute = neurones au max
        - dead_neuron_ratio haut = neurones inutiles
        - activation_variance basse = pas de diversité
        """
        # Poids des différents facteurs
        w_gradient = 0.35
        w_activation = 0.25
        w_dead = 0.20
        w_variance = 0.20
        
        # Normaliser: on veut un score où 1 = problématique
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
    Un bloc spécialisé pour une tâche ou sous-tâche.
    Contient les layers (experts) et les métadonnées d'utilisation.
    """
    id: str
    task_type: TaskType
    num_params: int
    layers: List[Any]  # nn.ModuleList en PyTorch
    creation_cycle: int
    usage_count: int = 0
    last_used_cycle: int = 0
    
    # Métriques de saturation détaillées
    saturation: SaturationMetrics = field(default_factory=SaturationMetrics)
    
    def get_param_count(self) -> int:
        return self.num_params
    
    def update_param_count(self):
        """Recalcule le nombre de paramètres basé sur les layers."""
        self.num_params = sum(
            p.numel() for layer in self.layers 
            for p in layer.parameters()
        )


@dataclass
class ModelMetrics:
    """
    Métriques collectées pendant le training.
    Utilisées pour les décisions d'expansion.
    """
    loss_history: List[float] = field(default_factory=list)
    validation_scores: List[float] = field(default_factory=list)
    gradient_norms: Dict[str, List[float]] = field(default_factory=dict)
    expert_utilization: Dict[str, float] = field(default_factory=dict)
    validation_accuracy: float = 0.0
    saturation_scores: Dict[str, float] = field(default_factory=dict)
    
    # Nouvelles métriques détaillées
    detailed_saturation: Dict[str, SaturationMetrics] = field(default_factory=dict)
    
    def add_loss(self, loss: float):
        self.loss_history.append(loss)
    
    def add_validation_score(self, score: float):
        self.validation_scores.append(score)
    
    def get_recent_loss_trend(self, window: int = 10) -> Optional[float]:
        """Retourne l'amélioration relative de la loss sur les N derniers cycles."""
        if len(self.loss_history) < window:
            return None
        recent = self.loss_history[-window:]
        if recent[0] == 0:
            return 0.0
        return (recent[0] - recent[-1]) / (recent[0] + 1e-8)
    
    def get_relative_performance_threshold(self, lookback: int = 5) -> float:
        """
        Calcule un seuil de performance relatif basé sur l'historique.
        Utilisé pour le vote des variantes.
        """
        if len(self.validation_scores) < lookback:
            # Pas assez d'historique, utiliser un seuil par défaut bas
            return 0.3
        
        recent_avg = sum(self.validation_scores[-lookback:]) / lookback
        # Le seuil est 95% de la moyenne récente
        return recent_avg * 0.95


@dataclass
class ExpansionDecision:
    """Résultat d'une décision d'expansion."""
    should_expand: bool
    target_block_id: Optional[str] = None
    expansion_type: str = "none"  # "width", "depth", "new_block"
    confidence: float = 0.0
    reason: str = ""


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
    new_block_exploration_prob: float = 0.3  # Prob de forcer vers nouveau bloc
    new_block_exploration_cycles: int = 3    # Cycles d'exploration forcée
    
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


@dataclass 
class TrainingLog:
    """Log d'un cycle de training."""
    cycle: int
    avg_loss: Optional[float]
    total_params: int
    num_blocks: int
    expanded: bool
    expansion_type: Optional[str] = None
    expansion_target: Optional[str] = None
    warmup_active: bool = False
    saturation_details: Optional[Dict[str, float]] = None
