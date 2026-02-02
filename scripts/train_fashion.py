#!/usr/bin/env python3
"""
Training ACOC sur Fashion-MNIST
================================
Dataset de vêtements (plus difficile que MNIST classique).
10 classes : T-shirt, Pantalon, Pull, Robe, Manteau, Sandale, Chemise, Sneaker, Sac, Bottine
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from acoc import ACOCModel, ACOCTrainer, SystemConfig


FASHION_CLASSES = [
    'T-shirt', 'Pantalon', 'Pull', 'Robe', 'Manteau',
    'Sandale', 'Chemise', 'Sneaker', 'Sac', 'Bottine'
]


def get_fashion_mnist_loaders(batch_size=64, data_dir='./data'):
    """Télécharge et prépare Fashion-MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        data_dir, train=False, transform=transform
    )

    def collate_fn(batch):
        """Convertit les labels en one-hot."""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
        return images, labels_onehot

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, test_loader


def main():
    print("=" * 70)
    print("ACOC Training sur Fashion-MNIST")
    print("=" * 70)

    # Configuration
    config = SystemConfig(
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        input_dim=784,
        hidden_dim=256,
        output_dim=10,
        num_variants=5,
        saturation_threshold=0.55,  # Baissé de 0.6
        min_cycles_before_expand=2,  # Réduit de 3
        expansion_cooldown=3,  # Réduit de 5
        performance_threshold_ratio=0.90  # Baissé de 0.95
    )

    print(f"\n✓ Configuration:")
    print(f"  - Device: {config.device}")
    print(f"  - Classes: {', '.join(FASHION_CLASSES)}")

    # Charger Fashion-MNIST
    print(f"\n📥 Téléchargement de Fashion-MNIST...")
    train_loader, test_loader = get_fashion_mnist_loaders(batch_size=128)
    print(f"  - Train: {len(train_loader.dataset)} samples")
    print(f"  - Test: {len(test_loader.dataset)} samples")

    # Modèle
    model = ACOCModel(config)
    print(f"\n✓ Modèle créé: {model.get_total_params():,} paramètres")

    # Trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    # Training
    print(f"\n{'='*70}")
    print("Démarrage du training (25 cycles)...")
    print(f"{'='*70}")

    trainer.run(
        num_cycles=25,
        data_loader=train_loader,
        validation_data=test_loader,
        num_steps_per_cycle=150,
        verbose=True
    )

    # Sauvegarder
    print(f"\n💾 Sauvegarde du modèle...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'cycle': model.current_cycle,
        'training_logs': trainer.training_logs,
        'classes': FASHION_CLASSES,
    }, 'acoc_fashion.pth')
    print(f"  ✓ Sauvegardé: acoc_fashion.pth")

    # Évaluation
    print(f"\n📊 Évaluation finale...")
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labels_idx = torch.max(labels, 1)  # Convertir one-hot en indices

            total += labels.size(0)
            correct += (predicted == labels_idx).sum().item()

            for i in range(len(labels_idx)):
                label = labels_idx[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print(f"  Accuracy globale: {accuracy:.2f}%")
    print(f"\n  Précision par classe:")
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f"    {FASHION_CLASSES[i]:12s}: {acc:.1f}%")

    print(f"\n  Paramètres finaux: {model.get_total_params():,}")
    print(f"  Blocs finaux: {len(model.task_blocks)}")
    print(f"  Expansions: {len(model.expansion_manager.expansion_history)}")

    print(f"\n{'='*70}")
    print("✅ Training terminé!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
