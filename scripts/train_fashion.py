#!/usr/bin/env python3
"""
Training ACOC on Fashion-MNIST (Refactored)
==========================================
Refactored version using BaseACOCTrainer.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class FashionMNISTTrainer(BaseACOCTrainer):
    """Trainer spécifique pour Fashion-MNIST."""

    CLASSES = [
        'T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
        'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
    ]

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
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(torch.flatten)
        ])

        train_dataset = datasets.FashionMNIST(
            './data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
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
        return "fashion"

    def get_dataset_info(self) -> dict:
        return {
            "Input": "784 (28×28 grayscale)",
            "Hidden": 256,
            "Classes": ", ".join(self.CLASSES)
        }


if __name__ == '__main__':
    trainer = FashionMNISTTrainer(num_cycles=25, batch_size=128)
    trainer.run()
