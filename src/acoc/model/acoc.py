"""
ACOC - Main Model (PyTorch)
============================
The ACOC model with modular architecture and organic growth.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set

from ..config import (
    SystemConfig, TaskBlock, ModelMetrics,
    ExpansionDecision, TaskType, SaturationMetrics
)
from ..core import Router
from ..experts import ExpertFactory, BaseExpert
from ..management import ExpansionManager, PenaltyManager, PruningManager, WarmupManager
from ..variants import VariantSystem


class ACOCModel(nn.Module):
    """
    Main ACOC model (Adaptive Controlled Organic Capacity).
    """
    
    def __init__(self, config: SystemConfig):
        super().__init__()
        
        self.config = config
        self.current_cycle = 0
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        self.router = Router(
            input_dim=config.input_dim,
            num_routes=3, 
            hidden_dim=128
        )
        self.task_blocks: Dict[str, TaskBlock] = {}
        self._block_modules: nn.ModuleDict = nn.ModuleDict()
        
        self.metrics = ModelMetrics()
        self.variant_system = VariantSystem(config, self.device)
        self.expansion_manager = ExpansionManager(config)
        self.penalty_manager = PenaltyManager(config)
        self.pruning_manager = PruningManager(config)
        self.warmup_manager = WarmupManager(config)
        self.protected_blocks: Set[str] = set() 
        
        self._force_exploration_block: Optional[str] = None
        self._exploration_prob: float = 0.0
        self._router_bias_initialized: bool = False

        self._initialize_base_blocks()
        self.to(self.device)
    
    def _initialize_base_blocks(self):
        """Creates base blocks via the Factory."""
        base_types = [TaskType.TEXT, TaskType.IMAGE, TaskType.AUDIO]

        for task_type in base_types:
            block_id = f"base_{task_type.value}"

            # Define the type
            expert_type = "mlp"
            if task_type == TaskType.IMAGE and self.config.use_cnn:
                expert_type = "cnn"
            elif task_type == TaskType.AUDIO:
                expert_type = "audio_mlp"

            # Use the Factory
            expert = ExpertFactory.create(
                expert_type=expert_type,
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                name=f"{block_id}_expert_0",
                config=self.config
            )
            
            block = TaskBlock(
                id=block_id,
                task_type=task_type,
                num_params=expert.get_param_count(),
                layers=[expert],
                creation_cycle=0
            )
            
            self.task_blocks[block_id] = block
            self._block_modules[block_id] = expert
            self.protected_blocks.add(block_id)
    
    def _sync_modules(self):
        to_remove = [k for k in self._block_modules.keys() if k not in self.task_blocks]
        for k in to_remove:
            del self._block_modules[k]
        
        for block_id, block in self.task_blocks.items():
            if block_id not in self._block_modules:
                if len(block.layers) == 1:
                    self._block_modules[block_id] = block.layers[0]
                else:
                    self._block_modules[block_id] = nn.Sequential(*block.layers)
    
    def set_exploration(self, block_id: Optional[str], prob: float = 0.0):
        self._force_exploration_block = block_id
        self._exploration_prob = prob
    
    def forward(self, x: torch.Tensor, task_hint: Optional[TaskType] = None) -> Tuple[torch.Tensor, Dict[str, int]]:
        x = x.to(self.device)
        batch_size = x.size(0)

        # Automatic router bias initialization on first batch
        if not self._router_bias_initialized:
            data_type = self.router.detect_data_type(x)
            block_ids = list(self.task_blocks.keys())

            target_block = f"base_{data_type}"
            if target_block in block_ids:
                target_idx = block_ids.index(target_block)
                self.router.set_route_bias(target_idx, 1.0)
                print(f"[Router] Detected type: {data_type} → slight bias (+1.0) towards {target_block}")

            self._router_bias_initialized = True

        if self._force_exploration_block is not None and self._exploration_prob > 0:
            block_ids = list(self.task_blocks.keys())
            if self._force_exploration_block in block_ids:
                force_idx = block_ids.index(self._force_exploration_block)
                selected, probs = self.router.forward_with_exploration(
                    x, force_route=force_idx, exploration_prob=self._exploration_prob
                )
            else:
                selected, probs = self.router(x)
        else:
            selected, probs = self.router(x)
        
        block_ids = list(self.task_blocks.keys())
        routing_stats = {bid: 0 for bid in block_ids}
        outputs = torch.zeros(batch_size, self.config.output_dim, device=self.device)

        for idx, block_id in enumerate(block_ids):
            mask = selected == idx
            if not mask.any():
                continue
            
            block_input = x[mask]
            block = self.task_blocks[block_id]
            
            # Note: Reshaping is handled inside BaseExpert.forward if it's a CNN
            block_output = block_input
            for layer in block.layers:
                block_output = layer(block_output)
            
            outputs[mask] = block_output

            count = int(mask.sum().item())
            routing_stats[block_id] = count
            block.usage_count += count
            block.last_used_cycle = self.current_cycle
        
        return outputs, routing_stats
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor, include_penalties: bool = True) -> torch.Tensor:
        if self.config.use_cross_entropy:
            if targets.dim() == 2:
                targets = targets.argmax(dim=1)
            base_loss = nn.functional.cross_entropy(predictions, targets)
        else:
            import warnings
            warnings.warn("MSE loss deprecated.", DeprecationWarning, stacklevel=2)
            base_loss = nn.functional.mse_loss(predictions, targets)
        
        if not include_penalties:
            return base_loss
        
        total_penalty, _, _ = self.penalty_manager.compute_total_penalty(
            self.task_blocks, self.router.get_param_count()
        )
        penalty_tensor = torch.tensor(total_penalty, device=self.device)
        ewc_penalty = self.router.ewc_loss()
        
        return base_loss + penalty_tensor + ewc_penalty
    
    def collect_metrics(self) -> ModelMetrics:
        for block_id, block in self.task_blocks.items():
            all_metrics = []
            for layer in block.layers:
                if isinstance(layer, BaseExpert):
                    all_metrics.append(layer.get_saturation_metrics())
            
            if all_metrics:
                combined = SaturationMetrics()
                n = len(all_metrics)
                combined.gradient_flow_ratio = sum(m.gradient_flow_ratio for m in all_metrics) / n
                combined.activation_saturation = sum(m.activation_saturation for m in all_metrics) / n
                combined.dead_neuron_ratio = sum(m.dead_neuron_ratio for m in all_metrics) / n
                combined.activation_variance = sum(m.activation_variance for m in all_metrics) / n
                combined.compute_combined_score()
                
                self.metrics.detailed_saturation[block_id] = combined
                self.metrics.saturation_scores[block_id] = combined.combined_score
                block.saturation = combined
        
        total_usage = sum(b.usage_count for b in self.task_blocks.values())
        if total_usage > 0:
            for block_id, block in self.task_blocks.items():
                self.metrics.expert_utilization[block_id] = block.usage_count / total_usage
        
        return self.metrics
    
    def reset_usage_counts(self):
        for block in self.task_blocks.values():
            block.usage_count = 0
    
    def evaluate_expansion(self) -> ExpansionDecision:
        return self.expansion_manager.evaluate_expansion_need(
            self.metrics, self.task_blocks, self.current_cycle
        )
    
    def execute_expansion(self, decision: ExpansionDecision) -> bool:
        """Executes an expansion decision"""

        # Let the standard manager handle the expansion
        success = self.expansion_manager.execute_expansion(
            decision,
            self.task_blocks,
            self.current_cycle,
            self.device
        )

        if success:
            if decision.expansion_type == "new_block":
                # The manager created the block in self.task_blocks
                # The main model just needs to update the router and warmup

                # Add a route if necessary
                # (Compare number of routes to number of blocks)
                if self.router.num_routes < len(self.task_blocks):
                    self.router.add_route(self.device)

                # Find the new block (usually the last one added)
                # Or use more robust logic if the manager returned the ID
                new_block_id = list(self.task_blocks.keys())[-1]

                # Start warmup
                self.warmup_manager.start_warmup(new_block_id, self.current_cycle)

                # Enable exploration to force data into the new block
                self.set_exploration(
                    new_block_id,
                    self.config.new_block_exploration_prob
                )

            # Synchronize PyTorch modules
            self._sync_modules()
            self.to(self.device)

        return success

    def run_maintenance(self) -> Dict[str, List[str]]:
        protected = set(self.warmup_manager.get_warmup_blocks())
        protected.update(self.protected_blocks)
        # protected.update(self.expansion_manager.get_warmup_blocks(self.current_cycle))
        # Note: get_warmup_blocks may not be in expansion_manager depending on version,
        # but we keep the existing logic.
        
        actions = self.pruning_manager.run_maintenance(
            self.task_blocks,
            self.current_cycle,
            protected
        )
        self._sync_modules()
        return actions
    
    def get_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "ACOC Model Summary",
            "=" * 50,
            f"Device: {self.device}",
            f"Cycle: {self.current_cycle}",
            f"Total params: {self.get_total_params():,}",
            f"Num blocks: {len(self.task_blocks)}",
            "",
            "Blocks:",
        ]
        
        for block_id, block in self.task_blocks.items():
            sat = block.saturation.combined_score if block.saturation else 0
            util = self.metrics.expert_utilization.get(block_id, 0)
            warmup = " [WARMUP]" if self.warmup_manager.is_warmup_active(block_id) else ""
            
            # Display expert type if possible
            e_type = ""
            if block.layers and isinstance(block.layers[0], BaseExpert):
                e_type = f" ({block.layers[0].expert_type})"
                
            lines.append(
                f"  {block_id}{e_type}: {block.num_params:,} params, "
                f"sat={sat:.1%}, util={util:.1%}, "
                f"created@{block.creation_cycle}{warmup}"
            )
        
        lines.append("")
        lines.append(f"Expansions: {len(self.expansion_manager.expansion_history)}")
        lines.append(f"Prunings: {len(self.pruning_manager.pruning_history)}")
        lines.append("=" * 50)
        
        return "\n".join(lines)