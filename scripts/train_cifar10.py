#!/usr/bin/env python3
"""
Training ACOC sur CIFAR-10 (Refactorisé)
========================================
Version refactorisée utilisant BaseACOCTrainer.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class CIFAR10Trainer(BaseACOCTrainer):
    """Trainer spécifique pour CIFAR-10."""

    CLASSES = [
        'Avion', 'Auto', 'Oiseau', 'Chat', 'Cerf',
        'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
    ]

    def get_config(self) -> SystemConfig:
        return SystemConfig(
            device=self.device,
            input_dim=3072,  # 3×32×32
            hidden_dim=512,
            output_dim=10,
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=True,
            cnn_channels=[32, 64, 128],
            image_channels=3
        )

    def get_dataloaders(self) -> tuple:
        # Normalisation standard CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        train_dataset = datasets.CIFAR10(
            './data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            './data', train=False, transform=transform_test
        )

        collate_fn = create_onehot_collate_fn(10)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0
        )

        return train_loader, test_loader

    def get_class_names(self) -> list:
        return self.CLASSES

    def get_dataset_name(self) -> str:
        return "cifar10"

    def get_dataset_info(self) -> dict:
        return {
            "Input": "3072 (32×32×3 RGB)",
            "Hidden": 512,
            "Classes": ", ".join(self.CLASSES)
        }


if __name__ == '__main__':
    trainer = CIFAR10Trainer(num_cycles=50, batch_size=128)
    trainer.run()
