"""
ACOC (Adaptive Controlled Organic Capacity) - PyTorch Implementation
=====================================================================

A dynamic growth neural network system with:
- Controlled expansion between training phases
- Double penalty (global + per-task) for sparsity
- 5 variants for voting and model averaging (RELATIVE threshold)
- Anti-forgetting protection via EWC
- Saturation metrics based on gradient flow and activations
- Warmup with forced exploration after expansion

Usage:
    from acoc import ACOCModel, ACOCTrainer, SystemConfig

    config = SystemConfig(device='cuda')
    model = ACOCModel(config)
    trainer = ACOCTrainer(model, config)
    trainer.run(num_cycles=10)
"""

# Configuration and structures
from .config import (
    TaskType,
    TaskBlock,
    ModelMetrics,
    ExpansionDecision,
    SystemConfig,
    TrainingLog,
    SaturationMetrics
)

# Base components
from .core import Router
from .core.projections import ModalityProjector, ProjectionLayer, ModalityConfig
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

# Variant system
from .variants import VariantSystem

# Managers
from .management import (
    ExpansionManager,
    PenaltyManager,
    PruningManager,
    WarmupManager
)
from .management.replay import ReplayBuffer, ReplayExample

# Main model
from .model import ACOCModel

# Trainer
from .training import ACOCTrainer, ContinualACOCTrainer

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
    "ModalityProjector",
    "ProjectionLayer",
    "ModalityConfig",
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
    "ReplayBuffer",
    "ReplayExample",

    # Main
    "ACOCModel",
    "ACOCTrainer",
    "ContinualACOCTrainer",
]
