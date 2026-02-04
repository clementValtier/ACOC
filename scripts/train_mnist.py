#!/usr/bin/env python3
"""
Training ACOC on MNIST (Refactored)
===================================
Handwritten digit classification 0-9.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class MNISTTrainer(BaseACOCTrainer):
    """Trainer spécifique pour MNIST."""

    CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    def get_config(self) -> SystemConfig:
        return SystemConfig(
            device=self.device,
            input_dim=784,  # 28×28
            hidden_dim=256,
            output_dim=10,
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=True,
            cnn_channels=[16, 32],
            image_channels=1
        )

    def get_dataloaders(self) -> tuple:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # Standard MNIST normalization
            transforms.Lambda(torch.flatten)
        ])

        train_dataset = datasets.MNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            './data', train=False, transform=transform
        )

        collate_fn = create_onehot_collate_fn(10)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=2, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=2, persistent_workers=True
        )

        return train_loader, test_loader

    def get_class_names(self) -> list:
        return self.CLASSES

    def get_dataset_name(self) -> str:
        return "mnist"

    def get_dataset_info(self) -> dict:
        return {
            "Input": "784 (28×28 grayscale)",
            "Hidden": 256,
            "Classes": "Digits 0-9"
        }


if __name__ == '__main__':
    trainer = MNISTTrainer(num_cycles=25, batch_size=128)
    trainer.run()
