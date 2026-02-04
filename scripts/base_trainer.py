#!/usr/bin/env python3
"""
Base Trainer for ACOC
=====================
Base class for factoring out common training code.
Specific trainers inherit from this class.
"""

import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

from acoc import ACOCModel, ACOCTrainer, SystemConfig


class BaseACOCTrainer(ABC):
    """
    Base class for all ACOC trainers.
    Factors out common code and defines the interface.
    """

    def __init__(self, num_cycles: int = 50, batch_size: int = 128):
        self.num_cycles = num_cycles
        self.batch_size = batch_size
        self.device = self._get_device()

    @abstractmethod
    def get_config(self) -> SystemConfig:
        """Return dataset-specific configuration."""
        pass

    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Return (train_loader, test_loader)."""
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Return class names."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Return dataset name (for saving)."""
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Return information to display (input_dim, etc.)."""
        pass

    def _get_device(self) -> str:
        """Detect the best available device."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def print_header(self):
        """Display startup header."""
        print("=" * 70)
        print(f"ACOC Training sur {self.get_dataset_name()}")
        print(f"Device: {self.device}")
        print("=" * 70)
        print()
        print("✓ Configuration:")
        for key, value in self.get_dataset_info().items():
            print(f"  - {key}: {value}")
        print()

    def run(self):
        """Run complete training."""
        self.print_header()

        # Preparation
        config = self.get_config()
        train_loader, test_loader = self.get_dataloaders()
        class_names = self.get_class_names()

        print("📥 Loading data...")
        print(f"  - Train: {len(train_loader.dataset)} samples")  # type: ignore[arg-type]
        print(f"  - Test: {len(test_loader.dataset)} samples")  # type: ignore[arg-type]
        print()

        # Model creation
        model = ACOCModel(config)
        print(f"✓ Model created: {model.get_total_params():,} parameters")
        print()

        # Training
        print("=" * 70)
        print(f"Starting training ({self.num_cycles} cycles)")
        print("=" * 70)

        trainer = ACOCTrainer(model, config, learning_rate=0.001)
        trainer.run(
            num_cycles=self.num_cycles,
            data_loader=train_loader,
            validation_data=test_loader,
            num_steps_per_cycle=150,
            verbose=True
        )

        # Model saving
        save_path = f"acoc_{self.get_dataset_name()}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n💾 Model saved: {save_path}")

        # Final evaluation
        print(f"\n📊 Final evaluation...")
        model.eval()
        correct = 0
        total = 0
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)

        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(model.device), targets.to(model.device)
                outputs, _ = model(data)
                _, predicted = torch.max(outputs.data, 1)
                _, labels = torch.max(targets.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Per class
                for i in range(len(labels)):
                    label = labels[i]
                    label_idx = int(label.item())
                    class_correct[label_idx] += int((predicted[i] == label).item())
                    class_total[label_idx] += 1

        accuracy = 100 * correct / total
        print(f"  Global accuracy: {accuracy:.2f}%")
        print(f"\n  Per-class precision:")
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"    {name:12s}: {acc:.1f}%")

        print()
        print("=" * 70)
        print("✅ Training completed!")
        print("=" * 70)

class OneHotCollate:
    """
    Callable class for one-hot encoding.
    Necessary to be 'picklable' by DataLoader with num_workers > 0.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, batch):
        """Convert labels to one-hot encoding."""
        data, labels = zip(*batch)
        data = torch.stack(data)
        labels = torch.tensor(labels)
        labels_onehot = torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        ).float()
        return data, labels_onehot

def create_onehot_collate_fn(num_classes: int):
    """Factory that returns a picklable callable instance."""
    return OneHotCollate(num_classes)
