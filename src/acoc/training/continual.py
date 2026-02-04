"""
ACOC - Continual Learning Trainer
==================================
Specialized trainer for continual learning scenarios with anti-forgetting mechanisms.

This trainer can be used in three modes:
1. Full continual learning (replay + projections enabled) - for multi-modal continual learning
2. Replay only (enable_replay=True, enable_projections=False) - for single-modal continual learning
3. Regular training (both disabled) - equivalent to base ACOCTrainer
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List

from ..config import SystemConfig
from ..model import ACOCModel
from ..management.replay import ReplayBuffer
from ..core.projections import ModalityProjector
from .trainer import ACOCTrainer


class ContinualACOCTrainer(ACOCTrainer):
    """
    Extended ACOC trainer with continual learning support.

    Key features (when enabled):
    - Experience replay to prevent forgetting
    - Modality projections for heterogeneous inputs
    - Task-specific block creation
    - Enhanced EWC on router
    - Interleaved training (current task + replay)
    """

    def __init__(
        self,
        model: ACOCModel,
        config: SystemConfig,
        learning_rate: float = 0.001,
        replay_buffer_size: int = 1000,
        replay_batch_ratio: float = 0.5,
        replay_frequency: int = 1,
        projector: Optional[ModalityProjector] = None,
        enable_replay: bool = True,
        enable_projections: bool = True
    ):
        """
        Args:
            model: ACOC model
            config: System configuration
            learning_rate: Learning rate
            replay_buffer_size: Max examples in replay buffer
            replay_batch_ratio: Ratio of batch from replay (0.5 = 50% replay, 50% current)
            replay_frequency: How often to use replay (1 = every step)
            projector: Modality projector (created if None and enable_projections=True)
            enable_replay: Whether to use experience replay
            enable_projections: Whether to use modality projections
        """
        super().__init__(model, config, learning_rate)

        # Feature flags
        self.enable_replay = enable_replay
        self.enable_projections = enable_projections

        # Replay buffer (only if enabled)
        if self.enable_replay:
            self.replay_buffer = ReplayBuffer(
                capacity=replay_buffer_size,
                sampling_strategy='balanced',
                device=self.model.device
            )
            self.replay_batch_ratio = replay_batch_ratio
            self.replay_frequency = replay_frequency
        else:
            self.replay_buffer = None  # type: ignore[assignment]
            self.replay_batch_ratio = 0.0
            self.replay_frequency = 0

        # Modality projector (only if enabled)
        if self.enable_projections:
            if projector is None:
                # Create default projector with model's input_dim as unified dimension
                self.projector = ModalityProjector(
                    unified_dim=config.input_dim,
                    device=self.model.device
                )
            else:
                self.projector = projector
        else:
            self.projector = None  # type: ignore[assignment]

        # Task tracking
        self.current_task: Optional[str] = None
        self.current_modality: Optional[str] = None
        self.tasks_seen: List[str] = []

        # Statistics
        self.replay_steps = 0
        self.forgetting_metrics: Dict[str, List[float]] = {}

    def start_task(
        self,
        task_id: str,
        modality: str,
        input_dim: int,
        output_dim: int,
        modality_type: str = 'image',
        metadata: Optional[Dict] = None
    ):
        """
        Begin training on a new task/modality.

        Args:
            task_id: Unique task identifier
            modality: Modality name
            input_dim: Native input dimension for this modality
            output_dim: Output dimension
            modality_type: Type of modality
            metadata: Additional metadata
        """
        print(f"\n{'=' * 70}")
        print(f"Starting task: {task_id}")
        print(f"{'=' * 70}")

        self.current_task = task_id
        self.current_modality = modality

        if task_id not in self.tasks_seen:
            self.tasks_seen.append(task_id)

        # Register modality if new (only if projections enabled)
        if self.enable_projections and modality not in self.projector.list_modalities():
            self.projector.register_modality(
                name=modality,
                input_dim=input_dim,
                modality_type=modality_type,
                metadata=metadata
            )

        # Check if we need a new task-specific block
        # (In current ACOC, router + experts handle this automatically)

        print(f"  Task ID: {task_id}")
        print(f"  Modality: {modality} ({modality_type})")
        if self.enable_projections:
            print(f"  Input dim: {input_dim} -> {self.projector.unified_dim}")
        else:
            print(f"  Input dim: {input_dim}")
        print(f"  Output dim: {output_dim}")
        print(f"  Tasks seen so far: {len(self.tasks_seen)}")

    def end_task(self):
        """Complete current task training."""
        if self.current_task is None:
            return

        print(f"\n{'=' * 70}")
        print(f"Completing task: {self.current_task}")
        print(f"{'=' * 70}")

        # Update Fisher Information Matrix for EWC
        # (This would be done automatically by router's compute_fisher)
        print(f"  * Task '{self.current_task}' completed")
        if self.enable_replay:
            print(f"  * Replay buffer: {len(self.replay_buffer)} examples")

        # Freeze previous task projections to prevent forgetting
        if self.enable_projections and self.current_modality:
            # Optionally freeze projection
            # self.projector.freeze_projections([self.current_modality])
            pass

        self.current_task = None
        self.current_modality = None

    def checkpoint_phase(
        self,
        validation_data: DataLoader | None = None,
        verbose: bool = True
    ):
        """
        Override checkpoint to handle projection of validation data.
        """
        if validation_data is None or not self.enable_projections:
            return super().checkpoint_phase(validation_data, verbose)

        # Create wrapper that projects validation data
        def evaluate_fn_with_projection(model):
            model.eval()
            total_loss = 0.0
            num_batches = 0

            with torch.no_grad():
                for val_x, val_y in validation_data:
                    val_x = val_x.to(model.device)
                    val_y = val_y.to(model.device)

                    # Project through current modality (if projections enabled)
                    if self.enable_projections and self.current_modality:
                        val_x = self.projector(val_x, self.current_modality)

                    outputs, _ = model(val_x)
                    loss = model.compute_loss(outputs, val_y, include_penalties=False)
                    total_loss += loss.item()
                    num_batches += 1

            model.train()
            return total_loss / max(num_batches, 1)

        # Use the parent's checkpoint logic but with our projection-aware eval
        if verbose:
            print(f"\n[Cycle {self.model.current_cycle}] === CHECKPOINT PHASE ===")

        # Collect metrics
        metrics = self.model.collect_metrics()
        self.model.metrics = metrics

        # Variant voting with projection
        should_expand, confidence, reason = self.model.variant_system.vote_on_expansion(
            self.model, evaluate_fn_with_projection, metrics
        )

        if verbose:
            # Print saturation details
            if metrics.detailed_saturation:
                print(f"  Detailed saturation:")
                for block_id, sat_metrics in metrics.detailed_saturation.items():
                    print(
                        f"    {block_id}: score={sat_metrics.combined_score:.2f} "
                        f"(grad_flow={sat_metrics.gradient_flow_ratio:.2f}, "
                        f"act_sat={sat_metrics.activation_saturation:.2f}, "
                        f"dead={sat_metrics.dead_neuron_ratio:.2f})"
                    )

            # Print utilization
            utilization = {
                block_id: block.usage_count / max(sum(b.usage_count for b in self.model.task_blocks.values()), 1)
                for block_id, block in self.model.task_blocks.items()
            }
            print(f"  Utilization: {utilization}")

            # Print vote
            print(f"  Vote: expand={should_expand}, conf={confidence:.2f}")

        return should_expand, confidence, reason

    def _training_step(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        use_replay: bool = True
    ) -> float:
        """
        Enhanced training step with replay.

        Args:
            batch_x: Current task batch input
            batch_y: Current task batch target
            use_replay: Whether to use replay buffer

        Returns:
            Combined loss
        """
        self.optimizer.zero_grad()

        # Project input through modality projector (if enabled)
        if self.enable_projections and self.current_modality:
            batch_x = self.projector(batch_x, self.current_modality)

        # Forward on current task
        outputs, routing_stats = self.model(batch_x)
        loss_current = self.model.compute_loss(outputs, batch_y)

        # Add replay if enabled and buffer has examples
        loss_replay = torch.tensor(0.0, device=self.model.device)
        if self.enable_replay and use_replay and len(self.replay_buffer) > 0 and self.replay_steps % self.replay_frequency == 0:
            # Compute replay batch size
            current_batch_size = batch_x.size(0)
            replay_batch_size = int(current_batch_size * self.replay_batch_ratio / (1 - self.replay_batch_ratio))
            replay_batch_size = max(1, min(replay_batch_size, len(self.replay_buffer)))

            try:
                # Sample from replay buffer
                replay_x, replay_y, replay_tasks, replay_modalities = self.replay_buffer.sample(
                    batch_size=replay_batch_size
                )

                # Project replay samples through their respective modalities INDIVIDUALLY (if enabled)
                if self.enable_projections:
                    replay_x_projected = []
                    for i in range(replay_x.size(0)):
                        x_single = replay_x[i]  # Get single sample
                        modality = replay_modalities[i]
                        # Project single sample
                        x_proj = self.projector(x_single.unsqueeze(0), modality)
                        replay_x_projected.append(x_proj.squeeze(0))

                    # Stack AFTER projection (all same dim now)
                    replay_x_projected = torch.stack(replay_x_projected)
                else:
                    # No projection needed
                    replay_x_projected = replay_x

                # Forward on replay samples
                outputs_replay, _ = self.model(replay_x_projected)
                loss_replay = self.model.compute_loss(outputs_replay, replay_y)

            except Exception as e:
                print(f"  Warning: Replay failed: {e}")
                loss_replay = torch.tensor(0.0, device=self.model.device)

        # Combined loss
        total_loss = loss_current + loss_replay

        # Backward
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Optimizer step
        self.optimizer.step()

        self.replay_steps += 1

        return total_loss.item()

    def populate_replay_buffer(
        self,
        data_loader: DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        Populate replay buffer with samples from current task.

        Args:
            data_loader: DataLoader for current task
            num_samples: Number of samples to add (None = all up to buffer capacity)
        """
        if not self.enable_replay:
            print(f"\n  Warning: Replay disabled, skipping buffer population")
            return

        if not self.current_task or not self.current_modality:
            raise RuntimeError("Must call start_task() before populating buffer")

        print(f"\n  Populating replay buffer for '{self.current_task}'...")

        samples_added = 0
        target_samples = num_samples or self.replay_buffer.capacity

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.model.device)
            batch_y = batch_y.to(self.model.device)

            # Add to buffer (without projection - store raw)
            self.replay_buffer.add_batch(
                data_batch=batch_x,
                target_batch=batch_y,
                task_id=self.current_task,
                modality=self.current_modality,
                num_samples=min(batch_x.size(0), target_samples - samples_added)
            )

            samples_added += batch_x.size(0)

            if samples_added >= target_samples:
                break

        print(f"  * Added {samples_added} samples to replay buffer")
        print(f"  * Buffer: {len(self.replay_buffer)}/{self.replay_buffer.capacity}")

    def evaluate_all_tasks(
        self,
        task_loaders: Dict[str, DataLoader]
    ) -> Dict[str, float]:
        """
        Evaluate model on all seen tasks.

        Args:
            task_loaders: Dict mapping task_id -> DataLoader

        Returns:
            Dict mapping task_id -> accuracy
        """
        self.model.eval()
        results = {}

        print(f"\n{'=' * 70}")
        print("Evaluating on all tasks")
        print(f"{'=' * 70}")

        for task_id, data_loader in task_loaders.items():
            if task_id not in self.tasks_seen:
                continue

            # Get modality for this task
            modality = task_id  # Assuming task_id == modality for now

            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in data_loader:
                    batch_x = batch_x.to(self.model.device)
                    batch_y = batch_y.to(self.model.device)

                    # Project (if enabled)
                    if self.enable_projections and modality in self.projector.list_modalities():
                        batch_x = self.projector(batch_x, modality)

                    # Forward
                    outputs, _ = self.model(batch_x)

                    # Get predictions
                    if batch_y.dim() == 2:  # One-hot
                        _, targets_idx = torch.max(batch_y, 1)
                    else:
                        targets_idx = batch_y

                    _, predicted = torch.max(outputs, 1)

                    total += targets_idx.size(0)
                    correct += (predicted == targets_idx).sum().item()

            accuracy = 100 * correct / total if total > 0 else 0.0
            results[task_id] = accuracy

            print(f"  {task_id:20s}: {accuracy:6.2f}%")

        self.model.train()
        return results

    def compute_forgetting(
        self,
        task_id: str,
        current_accuracy: float
    ) -> float:
        """
        Compute forgetting metric for a task.

        Forgetting = max_accuracy_seen - current_accuracy

        Args:
            task_id: Task identifier
            current_accuracy: Current accuracy on this task

        Returns:
            Forgetting amount (positive = forgot, negative = improved)
        """
        if task_id not in self.forgetting_metrics:
            self.forgetting_metrics[task_id] = []

        self.forgetting_metrics[task_id].append(current_accuracy)

        max_accuracy = max(self.forgetting_metrics[task_id])
        forgetting = max_accuracy - current_accuracy

        return forgetting

    def summary(self) -> str:
        """Generate summary of continual learning state."""
        lines = [
            "=" * 70,
            "Continual Learning Trainer Summary",
            "=" * 70,
            f"Tasks seen: {len(self.tasks_seen)}",
            f"Current task: {self.current_task or 'None'}",
        ]

        # Add replay info if enabled
        if self.enable_replay:
            lines.extend([
                f"Replay buffer: {len(self.replay_buffer)}/{self.replay_buffer.capacity}",
                f"Replay steps: {self.replay_steps}",
            ])

        lines.append("")

        # Add projector summary if enabled
        if self.enable_projections:
            lines.append(self.projector.summary())
            lines.append("")

        # Add replay buffer summary if enabled
        if self.enable_replay:
            lines.append(self.replay_buffer.summary())

        return "\n".join(lines)
