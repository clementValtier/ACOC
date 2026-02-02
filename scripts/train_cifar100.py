#!/usr/bin/env python3
"""
Training ACOC sur CIFAR-100
============================
Dataset très difficile avec 100 classes fines.
Idéal pour voir la croissance organique sur tâche complexe.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from acoc import ACOCModel, ACOCTrainer, SystemConfig


def get_cifar100_loaders(batch_size=128, data_dir='./data'):
    """Télécharge et prépare CIFAR-100."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.CIFAR100(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR100(
        data_dir, train=False, transform=transform_test
    )

    def collate_fn(batch):
        """Convertit les labels en one-hot."""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=100).float()
        return images, labels_onehot

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    return train_loader, test_loader


def main():
    print("=" * 70)
    print("ACOC Training sur CIFAR-100")
    print("Dataset difficile - 100 classes - Training très long")
    print("=" * 70)

    # Configuration pour dataset très complexe
    config = SystemConfig(
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        input_dim=3072,
        hidden_dim=768,  # Encore plus grand pour 100 classes
        output_dim=100,  # 100 classes
        num_variants=5,
        saturation_threshold=0.55,
        min_cycles_before_expand=2,
        expansion_cooldown=3,
        performance_threshold_ratio=0.90
    )

    print(f"\n✓ Configuration:")
    print(f"  - Device: {config.device}")
    print(f"  - Input: {config.input_dim}, Hidden: {config.hidden_dim}, Output: {config.output_dim}")
    print(f"  - 100 classes (très difficile)")

    # Charger CIFAR-100
    print(f"\n📥 Téléchargement de CIFAR-100 (~170 MB)...")
    train_loader, test_loader = get_cifar100_loaders(batch_size=128)
    print(f"  - Train: {len(train_loader.dataset)} samples")
    print(f"  - Test: {len(test_loader.dataset)} samples")

    # Modèle
    model = ACOCModel(config)
    initial_params = model.get_total_params()
    print(f"\n✓ Modèle créé: {initial_params:,} paramètres")
    print(f"  ⚠️  Training long attendu (~1-2h pour 100 cycles)")

    # Trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    # Training TRÈS LONG (100 cycles)
    print(f"\n{'='*70}")
    print("Démarrage du training (100 cycles)")
    print("💡 Conseil: Laissez tourner toute la nuit ou utilisez Ctrl+C pour arrêter")
    print(f"{'='*70}")

    try:
        trainer.run(
            num_cycles=100,
            data_loader=train_loader,
            validation_data=test_loader,
            num_steps_per_cycle=200,
            verbose=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrompu par l'utilisateur")
        print("Sauvegarde du modèle en cours...")

    # Sauvegarder
    print(f"\n💾 Sauvegarde du modèle...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'cycle': model.current_cycle,
        'training_logs': trainer.training_logs,
        'expansion_history': model.expansion_manager.expansion_history,
    }, 'acoc_cifar100.pth')
    print(f"  ✓ Sauvegardé: acoc_cifar100.pth")

    # Évaluation finale
    print(f"\n📊 Évaluation finale...")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)

            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            _, labels_idx = torch.max(labels, 1)

            total += labels.size(0)
            correct += (predicted == labels_idx).sum().item()

    accuracy = 100 * correct / total
    print(f"  Accuracy globale: {accuracy:.2f}%")

    # Stats croissance
    final_params = model.get_total_params()
    growth = ((final_params - initial_params) / initial_params) * 100

    print(f"\n  📈 Croissance du modèle:")
    print(f"    Initial: {initial_params:,} params")
    print(f"    Final: {final_params:,} params")
    print(f"    Croissance: +{growth:.1f}%")
    print(f"    Expansions: {len(model.expansion_manager.expansion_history)}")
    print(f"    Blocs: {len(model.task_blocks)}")

    if model.expansion_manager.expansion_history:
        print(f"\n  📊 Historique des expansions:")
        for cycle, target, exp_type in model.expansion_manager.expansion_history[:15]:
            print(f"    • Cycle {cycle}: {exp_type} sur {target}")
        if len(model.expansion_manager.expansion_history) > 15:
            print(f"    ... et {len(model.expansion_manager.expansion_history) - 15} autres")

    print(f"\n{'='*70}")
    print("✅ Training terminé!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
