"""
ACOC - Replay Buffer for Continual Learning
============================================
Implements experience replay to mitigate catastrophic forgetting.
"""

import torch
import random
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass


@dataclass
class ReplayExample:
    """Single example in replay buffer."""
    data: torch.Tensor
    target: torch.Tensor
    task_id: str
    modality: str
    importance: float = 1.0  # For prioritized sampling


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Key features:
    - Reservoir sampling for memory-efficient storage
    - Task-balanced sampling
    - Importance-weighted sampling
    - Supports multiple modalities
    """

    def __init__(
        self,
        capacity: int = 1000,
        sampling_strategy: str = 'balanced',  # 'balanced', 'random', 'prioritized'
        device: torch.device | None = None
    ):
        """
        Args:
            capacity: Maximum number of examples to store
            sampling_strategy: How to sample from buffer
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.sampling_strategy = sampling_strategy
        self.device = device or torch.device('cpu')

        # Buffer storage
        self.buffer: deque[ReplayExample] = deque(maxlen=capacity)

        # Task tracking
        self.task_counts: Dict[str, int] = {}
        self.task_indices: Dict[str, List[int]] = {}

        # Statistics
        self.total_added = 0
        self.total_sampled = 0

    def add(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        task_id: str,
        modality: str,
        importance: float = 1.0
    ):
        """
        Add a new example to the buffer.

        Uses reservoir sampling to maintain capacity while ensuring
        uniform representation across all seen examples.

        Args:
            data: Input data tensor
            target: Target tensor
            task_id: Task identifier
            modality: Modality identifier
            importance: Importance weight for prioritized sampling
        """
        example = ReplayExample(
            data=data.detach().cpu(),
            target=target.detach().cpu(),
            task_id=task_id,
            modality=modality,
            importance=importance
        )

        # Add to buffer
        if len(self.buffer) < self.capacity:
            # Buffer not full yet
            self.buffer.append(example)
        else:
            # Reservoir sampling: replace with probability capacity/total_added
            idx = random.randint(0, self.total_added)
            if idx < self.capacity:
                self.buffer[idx] = example

        self.total_added += 1

        # Update task tracking
        if task_id not in self.task_counts:
            self.task_counts[task_id] = 0
            self.task_indices[task_id] = []

        self.task_counts[task_id] += 1

        # Rebuild task indices (efficient for small buffers)
        self._rebuild_task_indices()

    def add_batch(
        self,
        data_batch: torch.Tensor,
        target_batch: torch.Tensor,
        task_id: str,
        modality: str,
        num_samples: Optional[int] = None
    ):
        """
        Add a batch of examples to buffer.

        Randomly samples a subset if num_samples is specified.

        Args:
            data_batch: Batch of input data (batch_size, ...)
            target_batch: Batch of targets (batch_size, ...)
            task_id: Task identifier
            modality: Modality identifier
            num_samples: Number of samples to add from batch (None = all)
        """
        batch_size = data_batch.size(0)

        # Sample subset if requested
        if num_samples is not None and num_samples < batch_size:
            indices = random.sample(range(batch_size), num_samples)
        else:
            indices = range(batch_size)

        # Add each example
        for idx in indices:
            self.add(
                data=data_batch[idx],
                target=target_batch[idx],
                task_id=task_id,
                modality=modality
            )

    def sample(
        self,
        batch_size: int,
        task_id: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of examples to sample
            task_id: If specified, sample only from this task

        Returns:
            Tuple of (data, targets, task_ids, modalities)
        """
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from empty buffer")

        batch_size = min(batch_size, len(self.buffer))

        # Select sampling strategy
        if task_id is not None:
            # Sample from specific task
            indices = self._sample_from_task(task_id, batch_size)
        elif self.sampling_strategy == 'balanced':
            indices = self._sample_balanced(batch_size)
        elif self.sampling_strategy == 'prioritized':
            indices = self._sample_prioritized(batch_size)
        else:  # random
            indices = random.sample(range(len(self.buffer)), batch_size)

        # Collect examples
        data_list = []
        target_list = []
        task_list = []
        modality_list = []

        for idx in indices:
            example = self.buffer[idx]
            data_list.append(example.data.to(self.device))
            target_list.append(example.target.to(self.device))
            task_list.append(example.task_id)
            modality_list.append(example.modality)

        # Check if all tensors have the same shape
        first_shape = data_list[0].shape
        is_uniform = all(d.shape == first_shape for d in data_list)

        if is_uniform:
            data = torch.stack(data_list)
        else:
            # Handle mismatch by padding to max size
            max_dim = max(d.shape[0] for d in data_list)
            
            # Create zero-padded tensor
            data = torch.zeros((len(data_list), max_dim), device=self.device)
            
            # Fill with data
            for i, d in enumerate(data_list):
                current_dim = d.shape[0]
                data[i, :current_dim] = d

        # Stack into tensors
        data = torch.stack(data_list)
        targets = torch.stack(target_list)

        self.total_sampled += len(indices)

        return data, targets, task_list, modality_list

    def _sample_balanced(self, batch_size: int) -> List[int]:
        """Sample with balanced representation across tasks."""
        if not self.task_indices:
            return random.sample(range(len(self.buffer)), batch_size)

        num_tasks = len(self.task_indices)
        samples_per_task = max(1, batch_size // num_tasks)

        indices = []
        for task_id in self.task_indices:
            task_idx = self.task_indices[task_id]
            if not task_idx:
                continue
            n_samples = min(samples_per_task, len(task_idx))
            indices.extend(random.sample(task_idx, n_samples))

        # If we need more samples to reach batch_size
        if len(indices) < batch_size:
            remaining = batch_size - len(indices)
            all_indices = set(range(len(self.buffer)))
            available = list(all_indices - set(indices))
            if available:
                indices.extend(random.sample(available, min(remaining, len(available))))

        return indices[:batch_size]

    def _sample_prioritized(self, batch_size: int) -> List[int]:
        """Sample based on importance weights."""
        # Compute sampling probabilities
        importances = [ex.importance for ex in self.buffer]
        total_importance = sum(importances)

        if total_importance == 0:
            return random.sample(range(len(self.buffer)), batch_size)

        probs = [imp / total_importance for imp in importances]

        # Sample without replacement
        indices = random.choices(
            range(len(self.buffer)),
            weights=probs,
            k=batch_size
        )

        return indices

    def _sample_from_task(self, task_id: str, batch_size: int) -> List[int]:
        """Sample only from a specific task."""
        if task_id not in self.task_indices:
            raise ValueError(f"Task '{task_id}' not found in buffer")

        task_idx = self.task_indices[task_id]
        if not task_idx:
            raise ValueError(f"No examples for task '{task_id}'")

        n_samples = min(batch_size, len(task_idx))
        return random.sample(task_idx, n_samples)

    def _rebuild_task_indices(self):
        """Rebuild index mapping for tasks."""
        self.task_indices = {}

        for idx, example in enumerate(self.buffer):
            task_id = example.task_id
            if task_id not in self.task_indices:
                self.task_indices[task_id] = []
            self.task_indices[task_id].append(idx)

    def get_task_distribution(self) -> Dict[str, float]:
        """Get proportion of each task in buffer."""
        if not self.buffer:
            return {}

        distribution = {}
        total = len(self.buffer)

        for task_id in self.task_indices:
            count = len(self.task_indices[task_id])
            distribution[task_id] = count / total

        return distribution

    def clear_task(self, task_id: str):
        """Remove all examples from a specific task."""
        self.buffer = deque(
            [ex for ex in self.buffer if ex.task_id != task_id],
            maxlen=self.capacity
        )
        self._rebuild_task_indices()

    def clear(self):
        """Clear the entire buffer."""
        self.buffer.clear()
        self.task_indices.clear()
        self.task_counts.clear()

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def summary(self) -> str:
        """Generate summary of buffer contents."""
        lines = [
            "Replay Buffer Summary:",
            f"  Size: {len(self.buffer)}/{self.capacity}",
            f"  Strategy: {self.sampling_strategy}",
            f"  Total added: {self.total_added}",
            f"  Total sampled: {self.total_sampled}",
            ""
        ]

        if self.task_indices:
            lines.append("  Task Distribution:")
            dist = self.get_task_distribution()
            for task_id, proportion in sorted(dist.items()):
                count = len(self.task_indices[task_id])
                lines.append(f"    {task_id:20s}: {count:4d} ({proportion:5.1%})")

        return "\n".join(lines)
