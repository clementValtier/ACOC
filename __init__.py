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

from .structures import (
    TaskType,
    TaskBlock,
    ModelMetrics,
    ExpansionDecision,
    SystemConfig,
    TrainingLog,
    SaturationMetrics
)

from .components import (
    Router, 
    Expert, 
    ExpertBlock,
    GradientFlowMonitor,
    ActivationMonitor
)

from .variants import VariantSystem

from .managers import (
    ExpansionManager,
    PenaltyManager, 
    PruningManager,
    WarmupManager
)

from .model import ACOCModel

from .trainer import ACOCTrainer


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
    "Expert",
    "ExpertBlock",
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
