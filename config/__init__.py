"""
ACOC - Configuration Module
===========================
Export de toutes les structures et configurations.
"""

from .structures import (
    TaskType,
    SaturationMetrics,
    TaskBlock,
    ModelMetrics,
    ExpansionDecision,
    TrainingLog
)

from .config import SystemConfig

__all__ = [
    "TaskType",
    "SaturationMetrics",
    "TaskBlock",
    "ModelMetrics",
    "ExpansionDecision",
    "TrainingLog",
    "SystemConfig",
]
