"""
ACOC (Adaptive Controlled Organic Capacity) - PyTorch Implementation
=====================================================================

Un système de réseau neuronal à croissance dynamique avec:
- Expansion contrôlée entre les phases de training
- Double malus (global + par tâche) pour la parcimonie
- 5 variantes pour vote et model averaging (seuil RELATIF)
- Protection anti-forgetting via EWC
- Métriques de saturation basées sur gradient flow et activations
- Warmup avec exploration forcée après expansion

Usage:
    from acoc import ACOCModel, ACOCTrainer, SystemConfig

    config = SystemConfig(device='cuda')
    model = ACOCModel(config)
    trainer = ACOCTrainer(model, config)
    trainer.run(num_cycles=10)
"""

# Configuration et structures
from .config import (
    TaskType,
    TaskBlock,
    ModelMetrics,
    ExpansionDecision,
    SystemConfig,
    TrainingLog,
    SaturationMetrics
)

# Composants de base
from .core import Router
from .experts import (
    BaseExpert,
    MLPExpert,
    CNNExpert,
    ExpertBlock,
    ExpertFactory
)

# Monitoring
from .monitoring import (
    GradientFlowMonitor,
    ActivationMonitor
)

# Système de variantes
from .variants import VariantSystem

# Gestionnaires
from .management import (
    ExpansionManager,
    PenaltyManager,
    PruningManager,
    WarmupManager
)

# Modèle principal
from .model import ACOCModel

# Trainer
from .training import ACOCTrainer

# Utilities (optional)
from .utils import get_logger, setup_logging


__version__ = "0.2.0"
__author__ = "ACOC Project"

__all__ = [
    # Structures
    "TaskType",
    "TaskBlock",
    "ModelMetrics",
    "ExpansionDecision",
    "SystemConfig",
    "TrainingLog",
    "SaturationMetrics",

    # Components
    "Router",
    "BaseExpert",
    "MLPExpert",
    "CNNExpert",
    "ExpertBlock",
    "ExpertFactory",
    "GradientFlowMonitor",
    "ActivationMonitor",

    # Systems
    "VariantSystem",
    "ExpansionManager",
    "PenaltyManager",
    "PruningManager",
    "WarmupManager",

    # Main
    "ACOCModel",
    "ACOCTrainer",
]
