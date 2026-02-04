"""
ACOC - Warmup Manager
=====================
Manages warmup phase following model expansion.
"""

from typing import Dict, List, Optional, Set

from ..config import SystemConfig


class WarmupManager:
    """
    Manages warmup phase following model expansion.

    Responsibilities:
    - Higher learning rate for newly added parameters
    - Force exploration towards new blocks
    - Track warmup phase progress
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.active_warmups: Dict[str, Dict] = {}  # Maps block_id to warmup info

    def start_warmup(
        self,
        block_id: str,
        current_cycle: int,
        new_params: Optional[Set[str]] = None
    ):
        """
        Starts warmup for a block.

        Args:
            block_id: Block identifier
            current_cycle: Current training cycle
            new_params: Names of newly added parameters
        """
        self.active_warmups[block_id] = {
            "start_cycle": current_cycle,
            "steps_done": 0,
            "new_params": new_params or set()
        }

    def is_warmup_active(self, block_id: str | None = None) -> bool:
        """Checks if a warmup is currently active."""
        if block_id:
            return block_id in self.active_warmups
        return len(self.active_warmups) > 0

    def get_warmup_blocks(self) -> List[str]:
        """Returns list of blocks currently in warmup."""
        return list(self.active_warmups.keys())

    def step(self, block_id: str | None = None):
        """Increments the warmup step counter."""
        if block_id:
            if block_id in self.active_warmups:
                self.active_warmups[block_id]["steps_done"] += 1
        else:
            for info in self.active_warmups.values():
                info["steps_done"] += 1

    def should_continue_warmup(self, block_id: str, current_cycle: int | None = None) -> bool:
        """
        Checks if warmup should continue.
        Enforces timeout after max_warmup_cycles to prevent infinite warmup.
        """
        if block_id not in self.active_warmups:
            return False

        info = self.active_warmups[block_id]

        # Check if step limit reached
        if info["steps_done"] >= self.config.warmup_steps:
            return False

        # Check if cycle timeout exceeded
        if current_cycle is not None:
            cycles_elapsed = current_cycle - info["start_cycle"]
            if cycles_elapsed >= self.config.max_warmup_cycles:
                return False

        return True

    def end_warmup(self, block_id: str):
        """Ends warmup for a block."""
        if block_id in self.active_warmups:
            del self.active_warmups[block_id]

    def check_and_cleanup(self, current_cycle: int | None = None):
        """
        Cleans up completed warmups.
        Also removes warmups that exceeded cycle timeout.
        """
        to_remove = []
        for block_id, info in self.active_warmups.items():
            # Cleanup by step limit
            if info["steps_done"] >= self.config.warmup_steps:
                to_remove.append(block_id)
                continue

            # Cleanup by cycle timeout
            if current_cycle is not None:
                cycles_elapsed = current_cycle - info["start_cycle"]
                if cycles_elapsed >= self.config.max_warmup_cycles:
                    to_remove.append(block_id)

        for block_id in to_remove:
            del self.active_warmups[block_id]

    def get_lr_multiplier(self, param_name: str) -> float:
        """
        Returns the learning rate multiplier for a parameter.
        Newly added parameters get higher learning rates.
        """
        for info in self.active_warmups.values():
            if param_name in info["new_params"]:
                return self.config.warmup_lr_multiplier
        return 1.0

    def get_exploration_prob(self, block_id: str) -> float:
        """
        Returns the forced exploration probability for a block.
        """
        if block_id in self.active_warmups:
            return self.config.new_block_exploration_prob
        return 0.0
