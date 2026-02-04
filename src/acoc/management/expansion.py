"""
ACOC - Expansion Manager
========================
Determines when and how to expand the model.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

from ..config import (
    SystemConfig, TaskBlock, ModelMetrics,
    ExpansionDecision, TaskType
)
from ..experts import ExpertFactory, ExpertBlock, BaseExpert

class ExpansionManager:
    """
    Decides when and how to expand the model.

    Expansion triggers based on:
    1. Combined saturation score (gradient flow + activations)
    2. Stagnant loss
    3. Variant voting
    """

    def __init__(self, config: SystemConfig):
        self.config = config
        self.last_expansion_cycle = -config.expansion_cooldown
        self.expansion_history: List[Tuple[int, str, str]] = []

        # Track recently created blocks for warmup phase tracking
        self.recently_created_blocks: Dict[str, int] = {}  # block_id -> creation_cycle

    def update_recent_usage(self, task_blocks: Dict[str, TaskBlock], current_cycle: int):
        """Updates recent usage history for all blocks."""
        for block in task_blocks.values():
            block.recent_usage.append(block.usage_count)

            # Keep only the N most recent cycles
            if len(block.recent_usage) > self.config.recent_usage_window:
                block.recent_usage.pop(0)

    def get_most_used_block_recent(self, task_blocks: Dict[str, TaskBlock]) -> Optional[TaskBlock]:
        """Returns the block with the highest recent average usage."""
        if not task_blocks:
            return None

        def get_recent_avg(block: TaskBlock) -> float:
            if not block.recent_usage:
                return 0.0
            return sum(block.recent_usage) / len(block.recent_usage)

        return max(task_blocks.values(), key=get_recent_avg)

    def evaluate_expansion_need(
        self,
        metrics: ModelMetrics,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int
    ) -> ExpansionDecision:
        """
        Analyzes saturation metrics and decides if expansion is needed.

        Uses detailed metrics:
        - Gradient flow ratio (signal flow blockage)
        - Activation saturation (saturated neurons)
        - Dead neuron ratio (inactive neurons)
        """
        # Check cooldown period
        cycles_since_last = current_cycle - self.last_expansion_cycle
        if cycles_since_last < self.config.expansion_cooldown:
            return ExpansionDecision(
                should_expand=False,
                reason=f"Cooldown active ({cycles_since_last}/{self.config.expansion_cooldown} cycles)"
            )

        # Verify minimum cycles requirement
        if current_cycle < self.config.min_cycles_before_expand:
            return ExpansionDecision(
                should_expand=False,
                reason=f"Insufficient history ({current_cycle}/{self.config.min_cycles_before_expand} cycles)"
            )

        # Analyze detailed saturation by block
        saturated_blocks = []

        for block_id, sat_metrics in metrics.detailed_saturation.items():
            # Skip blocks in warmup phase
            if block_id in self.recently_created_blocks:
                creation_cycle = self.recently_created_blocks[block_id]
                if current_cycle - creation_cycle < self.config.new_block_exploration_cycles:
                    continue

            # Use combined saturation score
            if sat_metrics.combined_score > self.config.saturation_threshold:
                saturated_blocks.append((block_id, sat_metrics))

        # Case 1: Saturated blocks -> Width expansion
        if saturated_blocks:
            # Sort by descending saturation score
            saturated_blocks.sort(key=lambda x: x[1].combined_score, reverse=True)
            target_block_id, sat_metrics = saturated_blocks[0]

            # Determine primary reasons for saturation
            reasons = []
            if sat_metrics.gradient_flow_ratio < 0.5:
                reasons.append(f"poor gradient flow ({sat_metrics.gradient_flow_ratio:.1%})")
            if sat_metrics.activation_saturation > 0.3:
                reasons.append(f"saturated activations ({sat_metrics.activation_saturation:.1%})")
            if sat_metrics.dead_neuron_ratio > 0.2:
                reasons.append(f"dead neurons ({sat_metrics.dead_neuron_ratio:.1%})")

            reason_str = ", ".join(reasons) if reasons else "high combined score"

            return ExpansionDecision(
                should_expand=True,
                target_block_id=target_block_id,
                expansion_type="width",
                confidence=sat_metrics.combined_score,
                reason=f"Block '{target_block_id}' saturated: {reason_str} (score={sat_metrics.combined_score:.2f})"
            )

        # Case 2: Stagnant loss without saturation -> Create new block
        loss_trend = metrics.get_recent_loss_trend(window=10)
        if loss_trend is not None and loss_trend < 0.01:
            return ExpansionDecision(
                should_expand=True,
                expansion_type="new_block",
                confidence=0.5,
                reason=f"Stagnant loss (improvement: {loss_trend:.2%}), insufficient capacity"
            )

        # No expansion needed
        return ExpansionDecision(
            should_expand=False,
            reason="No saturation or stagnation detected"
        )

    def execute_expansion(
        self,
        decision: ExpansionDecision,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        device: torch.device = None
    ) -> bool:
        """
        Executes the decided expansion.

        Returns:
            True if expansion succeeded
        """
        if not decision.should_expand:
            return False

        if device is None:
            device = torch.device('cpu')

        success = False

        if decision.expansion_type == "width":
            success = self._expand_width(decision.target_block_id, task_blocks)

        elif decision.expansion_type == "depth":
            success = self._expand_depth(decision.target_block_id, task_blocks)

        elif decision.expansion_type == "new_block":
            new_id = self._create_new_block(task_blocks, current_cycle, device, decision.target_block_id)
            if new_id:
                decision.target_block_id = new_id
                self.recently_created_blocks[new_id] = current_cycle
                success = True

        if success:
            self.last_expansion_cycle = current_cycle
            self.expansion_history.append((
                current_cycle,
                decision.target_block_id or "new",
                decision.expansion_type
            ))

        return success

    def _expand_width(
        self,
        block_id: str,
        task_blocks: Dict[str, TaskBlock]
    ) -> bool:
        """Adds neurons to experts in a block (width expansion)."""
        if block_id not in task_blocks:
            return False

        block = task_blocks[block_id]

        for layer in block.layers:
            if isinstance(layer, BaseExpert):
                additional = max(1, int(layer.hidden_dim * self.config.expansion_ratio))
                layer.expand_width(additional)
                layer.reset_monitors()

        block.update_param_count()
        return True

    def _expand_depth(
        self,
        block_id: str,
        task_blocks: Dict[str, TaskBlock]
    ) -> bool:
        """Adds a layer to a block (depth expansion)."""
        if block_id not in task_blocks:
            return False

        block = task_blocks[block_id]

        if block.layers and isinstance(block.layers[-1], BaseExpert):
            last_expert = block.layers[-1]

            # Retrieve the type of the last expert to maintain consistency
            expert_type = getattr(last_expert, 'expert_type', 'mlp')

            # Create new expert via Factory
            new_expert = ExpertFactory.create(
                expert_type=expert_type,
                input_dim=last_expert.output_dim,   # Input is output of previous layer
                hidden_dim=last_expert.output_dim,  # Maintain same width
                output_dim=last_expert.output_dim,  # Maintain same output dimension
                name=f"{block_id}_expert_{len(block.layers)}",
                config=self.config
            ).to(last_expert.parameters().__next__().device)  # Same device as previous
            
            block.layers.append(new_expert)
            block.update_param_count()
            return True

        return False

    def _create_new_block(
        self,
        task_blocks: Dict[str, TaskBlock],
        current_cycle: int,
        device: torch.device,
        target_id_hint: Optional[str] = None
    ) -> Optional[str]:
        new_id = f"block_{len(task_blocks)}"

        task_type = TaskType.GENERIC
        expert_type = "mlp"

        # Inherit from the most recently used block
        if task_blocks:
            # Find the block with highest recent usage
            most_used_block = self.get_most_used_block_recent(task_blocks)

            if most_used_block:
                # Inherit task type and expert type
                task_type = most_used_block.task_type
                if most_used_block.layers and isinstance(most_used_block.layers[0], BaseExpert):
                    expert_type = getattr(most_used_block.layers[0], 'expert_type', "mlp")

            # Override: use target_id_hint if specified
            if target_id_hint and target_id_hint in task_blocks:
                parent = task_blocks[target_id_hint]
                task_type = parent.task_type
                if parent.layers and isinstance(parent.layers[0], BaseExpert):
                    expert_type = getattr(parent.layers[0], 'expert_type', "mlp")

        # Create expert via Factory
        expert = ExpertFactory.create(
            expert_type=expert_type,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            name=f"{new_id}_expert_0",
            config=self.config
        ).to(device)

        new_block = TaskBlock(
            id=new_id,
            task_type=task_type,
            num_params=expert.get_param_count(),
            layers=[expert],
            creation_cycle=current_cycle
        )

        task_blocks[new_id] = new_block
        return new_id

    def is_in_warmup(self, block_id: str, current_cycle: int) -> bool:
        """Checks if a block is in warmup phase."""
        if block_id not in self.recently_created_blocks:
            return False
        creation_cycle = self.recently_created_blocks[block_id]
        return current_cycle - creation_cycle < self.config.new_block_exploration_cycles

    def get_warmup_blocks(self, current_cycle: int) -> List[str]:
        """Returns the list of blocks currently in warmup phase."""
        return [
            block_id for block_id, creation
            in self.recently_created_blocks.items()
            if current_cycle - creation < self.config.new_block_exploration_cycles
        ]

    def get_expansion_stats(self) -> Dict:
        """Returns statistics about all expansions performed."""
        type_counts = {}
        for _, _, exp_type in self.expansion_history:
            type_counts[exp_type] = type_counts.get(exp_type, 0) + 1

        return {
            "total_expansions": len(self.expansion_history),
            "by_type": type_counts,
            "last_expansion_cycle": self.last_expansion_cycle
        }
