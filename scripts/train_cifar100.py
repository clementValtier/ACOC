#!/usr/bin/env python3
"""
Training ACOC on CIFAR-100 (Refactored)
======================================
Classification on 100 fine-grained classes (vs 10 for CIFAR-10).
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class CIFAR100Trainer(BaseACOCTrainer):
    """Trainer spécifique pour CIFAR-100."""

    # 100 fine-grained classes of CIFAR-100
    CLASSES = [
        'pomme', 'aquarium_fish', 'bébé', 'ours', 'castor', 'lit', 'abeille', 'scarabée',
        'vélo', 'bouteille', 'bol', 'garçon', 'pont', 'bus', 'papillon', 'chameau',
        'boîte', 'château', 'chenille', 'bétail', 'chaise', 'chimpanzé', 'horloge',
        'nuage', 'cafard', 'canapé', 'crabe', 'crocodile', 'tasse', 'dinosaure',
        'dauphin', 'éléphant', 'rascasse', 'forêt', 'renard', 'fille', 'hamster',
        'maison', 'kangourou', 'clavier', 'lampe', 'tondeuse', 'léopard', 'lion',
        'lézard', 'homard', 'homme', 'érable', 'moto', 'montagne', 'souris', 'champignon',
        'chêne', 'orange', 'orchidée', 'loutre', 'palmier', 'poire', 'camion', 'pin',
        'plaine', 'assiette', 'coquelicot', 'porc-épic', 'opossum', 'lapin', 'raton_laveur',
        'raie', 'route', 'fusée', 'rose', 'mer', 'phoque', 'requin', 'musaraigne',
        'mouffette', 'gratte-ciel', 'escargot', 'serpent', 'araignée', 'écureuil',
        'tramway', 'tournesol', 'patate_douce', 'table', 'char', 'téléphone', 'télé',
        'tigre', 'tracteur', 'train', 'truite', 'tulipe', 'tortue', 'armoire', 'baleine',
        'saule', 'loup', 'femme', 'ver'
    ]

    def get_config(self) -> SystemConfig:
        return SystemConfig(
            device=self.device,
            input_dim=3072,  # 3×32×32
            hidden_dim=768,
            output_dim=100,  # 100 classes
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=True,
            cnn_channels=[32, 64, 128],
            image_channels=3
        )

    def get_dataloaders(self) -> tuple:
        # Standard CIFAR-100 normalization (similar to CIFAR-10)
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.Lambda(torch.flatten)
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            transforms.Lambda(torch.flatten)
        ])

        train_dataset = datasets.CIFAR100(
            './data', train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            './data', train=False, transform=transform_test
        )

        collate_fn = create_onehot_collate_fn(100)

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
        return "cifar100"

    def get_dataset_info(self) -> dict:
        return {
            "Input": "3072 (32×32×3 RGB)",
            "Hidden": 512,
            "Classes": "100 fine-grained classes"
        }


if __name__ == '__main__':
    trainer = CIFAR100Trainer(num_cycles=100, batch_size=128)
    trainer.run()
