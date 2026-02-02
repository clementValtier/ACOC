"""
ACOC - Warmup Manager
=====================
Gère le warmup après une expansion.
"""

from typing import Dict, List, Optional, Set

from ..config import SystemConfig


class WarmupManager:
    """
    Gère le warmup après une expansion.

    Responsabilités:
    - Learning rate plus élevé pour les nouveaux paramètres
    - Forcer l'exploration vers les nouveaux blocs
    - Tracking de la phase de warmup
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.active_warmups: Dict[str, Dict] = {}  # block_id -> warmup_info

    def start_warmup(
        self,
        block_id: str,
        current_cycle: int,
        new_params: Optional[Set[str]] = None
    ):
        """
        Démarre un warmup pour un bloc.

        Args:
            block_id: ID du bloc
            current_cycle: Cycle actuel
            new_params: Noms des paramètres nouvellement ajoutés
        """
        self.active_warmups[block_id] = {
            "start_cycle": current_cycle,
            "steps_done": 0,
            "new_params": new_params or set()
        }

    def is_warmup_active(self, block_id: str = None) -> bool:
        """Vérifie si un warmup est actif."""
        if block_id:
            return block_id in self.active_warmups
        return len(self.active_warmups) > 0

    def get_warmup_blocks(self) -> List[str]:
        """Retourne les blocs en warmup."""
        return list(self.active_warmups.keys())

    def step(self, block_id: str = None):
        """Incrémente le compteur de steps de warmup."""
        if block_id:
            if block_id in self.active_warmups:
                self.active_warmups[block_id]["steps_done"] += 1
        else:
            for info in self.active_warmups.values():
                info["steps_done"] += 1

    def should_continue_warmup(self, block_id: str) -> bool:
        """Vérifie si le warmup doit continuer."""
        if block_id not in self.active_warmups:
            return False
        return self.active_warmups[block_id]["steps_done"] < self.config.warmup_steps

    def end_warmup(self, block_id: str):
        """Termine le warmup pour un bloc."""
        if block_id in self.active_warmups:
            del self.active_warmups[block_id]

    def check_and_cleanup(self):
        """Nettoie les warmups terminés."""
        to_remove = [
            block_id for block_id, info in self.active_warmups.items()
            if info["steps_done"] >= self.config.warmup_steps
        ]
        for block_id in to_remove:
            del self.active_warmups[block_id]

    def get_lr_multiplier(self, param_name: str) -> float:
        """
        Retourne le multiplicateur de LR pour un paramètre.
        Les nouveaux paramètres ont un LR plus élevé.
        """
        for info in self.active_warmups.values():
            if param_name in info["new_params"]:
                return self.config.warmup_lr_multiplier
        return 1.0

    def get_exploration_prob(self, block_id: str) -> float:
        """
        Retourne la probabilité d'exploration forcée vers un bloc.
        """
        if block_id in self.active_warmups:
            return self.config.new_block_exploration_prob
        return 0.0
