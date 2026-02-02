#!/usr/bin/env python3
"""
Training ACOC sur CIFAR-10
===========================
Dataset plus complexe avec images couleur 32x32.
Training plus long pour voir la croissance organique.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from acoc import ACOCModel, ACOCTrainer, SystemConfig


CIFAR10_CLASSES = [
    'Avion', 'Auto', 'Oiseau', 'Chat', 'Cerf',
    'Chien', 'Grenouille', 'Cheval', 'Bateau', 'Camion'
]


def get_cifar10_loaders(batch_size=128, data_dir='./data'):
    """Télécharge et prépare CIFAR-10."""
    # Normalisation standard CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten 3x32x32 -> 3072
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, transform=transform_test
    )

    def collate_fn(batch):
        """Convertit les labels en one-hot."""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=10).float()
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
    print("ACOC Training sur CIFAR-10")
    print("Training long pour voir la croissance organique")
    print("=" * 70)

    # Configuration pour dataset plus complexe
    config = SystemConfig(
        device='mps' if torch.backends.mps.is_available() else 'cpu',
        input_dim=3072,  # 3x32x32 images aplaties
        hidden_dim=512,  # Plus grand pour images couleur
        output_dim=10,
        num_variants=5,
        saturation_threshold=0.65,
        min_cycles_before_expand=2,
        expansion_cooldown=8,
        performance_threshold_ratio=0.95,
        warmup_steps=200,
        use_cross_entropy=True,  # CrossEntropy pour classification
        new_block_exploration_prob=0.1,  # Réduit de 0.3 à 0.1
        max_warmup_cycles=10  # Timeout warmup après 10 cycles
    )

    print(f"\n✓ Configuration:")
    print(f"  - Device: {config.device}")
    print(f"  - Input: {config.input_dim}, Hidden: {config.hidden_dim}")
    print(f"  - Classes: {', '.join(CIFAR10_CLASSES)}")

    # Charger CIFAR-10
    print(f"\n📥 Téléchargement de CIFAR-10 (~170 MB)...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    print(f"  - Train: {len(train_loader.dataset)} samples")
    print(f"  - Test: {len(test_loader.dataset)} samples")

    # Modèle
    model = ACOCModel(config)
    initial_params = model.get_total_params()
    print(f"\n✓ Modèle créé: {initial_params:,} paramètres")

    # Trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    # Training LONG (50 cycles)
    print(f"\n{'='*70}")
    print("Démarrage du training (50 cycles, ~30-45 min)")
    print(f"{'='*70}")

    try:
        trainer.run(
            num_cycles=50,
            data_loader=train_loader,
            validation_data=test_loader,
            num_steps_per_cycle=150,  # Plus de steps
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
    }, 'acoc_cifar10.pth')
    print(f"  ✓ Sauvegardé: acoc_cifar10.pth")

    # Évaluation finale
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
            _, labels_idx = torch.max(labels, 1)

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
        print(f"    {CIFAR10_CLASSES[i]:12s}: {acc:.1f}%")

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
        for cycle, target, exp_type in model.expansion_manager.expansion_history[:10]:
            print(f"    • Cycle {cycle}: {exp_type} sur {target}")
        if len(model.expansion_manager.expansion_history) > 10:
            print(f"    ... et {len(model.expansion_manager.expansion_history) - 10} autres")

    print(f"\n{'='*70}")
    print("✅ Training terminé!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
