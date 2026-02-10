"""
ACOC - Training Module
======================
Export du trainer.
"""

from .trainer import ACOCTrainer
from .continual import ContinualACOCTrainer

__all__ = [
    "ACOCTrainer",
    "ContinualACOCTrainer",
]
