#!/usr/bin/env python3
"""
Base Trainer for ACOC
=====================
Classe de base pour factoriser le code de training commun.
Les trainers spécifiques héritent de cette classe.
"""

import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

from acoc import ACOCModel, ACOCTrainer, SystemConfig


class BaseACOCTrainer(ABC):
    """
    Classe de base pour tous les trainers ACOC.
    Factorise le code commun et définit l'interface.
    """

    def __init__(self, num_cycles: int = 50, batch_size: int = 128):
        self.num_cycles = num_cycles
        self.batch_size = batch_size
        self.device = self._get_device()

    @abstractmethod
    def get_config(self) -> SystemConfig:
        """Retourne la config spécifique au dataset."""
        pass

    @abstractmethod
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Retourne (train_loader, test_loader)."""
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Retourne les noms des classes."""
        pass

    @abstractmethod
    def get_dataset_name(self) -> str:
        """Retourne le nom du dataset (pour sauvegarde)."""
        pass

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Retourne les infos à afficher (input_dim, etc.)."""
        pass

    def _get_device(self) -> str:
        """Détecte le meilleur device disponible."""
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def print_header(self):
        """Affiche le header de début."""
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
        """Lance le training complet."""
        self.print_header()

        # Préparation
        config = self.get_config()
        train_loader, test_loader = self.get_dataloaders()
        class_names = self.get_class_names()

        print("📥 Chargement des données...")
        print(f"  - Train: {len(train_loader.dataset)} samples")
        print(f"  - Test: {len(test_loader.dataset)} samples")
        print()

        # Création du modèle
        model = ACOCModel(config)
        print(f"✓ Modèle créé: {model.get_total_params():,} paramètres")
        print()

        # Training
        print("=" * 70)
        print(f"Démarrage du training ({self.num_cycles} cycles)")
        print("=" * 70)

        trainer = ACOCTrainer(model, config, learning_rate=0.001)
        trainer.run(
            num_cycles=self.num_cycles,
            data_loader=train_loader,
            validation_data=test_loader,
            num_steps_per_cycle=150,
            verbose=True
        )

        # Sauvegarde du modèle
        save_path = f"acoc_{self.get_dataset_name()}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\n💾 Modèle sauvegardé: {save_path}")

        # Évaluation finale
        print(f"\n📊 Évaluation finale...")
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

                # Par classe
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        accuracy = 100 * correct / total
        print(f"  Accuracy globale: {accuracy:.2f}%")
        print(f"\n  Précision par classe:")
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"    {name:12s}: {acc:.1f}%")

        print()
        print("=" * 70)
        print("✅ Training terminé!")
        print("=" * 70)


def create_onehot_collate_fn(num_classes: int):
    """Factory pour créer une fonction collate avec one-hot encoding."""
    def collate_fn(batch):
        """Convertit les labels en one-hot."""
        data, labels = zip(*batch)
        data = torch.stack(data)
        labels = torch.tensor(labels)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
        return data, labels_onehot
    return collate_fn
