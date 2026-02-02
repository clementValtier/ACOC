#!/usr/bin/env python3
"""
Quick Test - ACOC
=================
Test rapide pour valider que la boucle de base fonctionne.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from acoc import ACOCModel, ACOCTrainer, SystemConfig


def create_synthetic_data(num_samples=200, input_dim=256, output_dim=256, batch_size=32):
    """Crée des données synthétiques pour le test."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    print("=" * 70)
    print("ACOC - Quick Test")
    print("=" * 70)

    # Configuration minimale
    config = SystemConfig(
        device='cpu',
        input_dim=256,
        hidden_dim=128,
        output_dim=256,
        num_variants=3,  # Réduit pour être plus rapide
        saturation_threshold=0.7,
        min_cycles_before_expand=2,
        expansion_cooldown=3
    )

    print(f"\n✓ Configuration créée")
    print(f"  - Device: {config.device}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Variants: {config.num_variants}")

    # Données
    train_loader = create_synthetic_data(num_samples=200)
    val_loader = create_synthetic_data(num_samples=50)

    print(f"✓ Données synthétiques créées")
    print(f"  - Train: 200 samples")
    print(f"  - Val: 50 samples")

    # Modèle
    model = ACOCModel(config)
    initial_params = model.get_total_params()

    print(f"✓ Modèle créé")
    print(f"  - Paramètres initiaux: {initial_params:,}")
    print(f"  - Blocs initiaux: {len(model.task_blocks)}")

    # Trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    print(f"✓ Trainer créé")
    print(f"  - Learning rate: 0.001")

    # Test rapide sur 5 cycles
    print("\n" + "=" * 70)
    print("Démarrage du test rapide (5 cycles)...")
    print("=" * 70)

    try:
        for cycle in range(5):
            print(f"\n--- Cycle {cycle} ---")

            # Training
            avg_loss = trainer.training_phase(train_loader, num_steps=10, verbose=False)
            print(f"  Training loss: {avg_loss:.4f}")

            # Checkpoint (retourne tuple)
            should_expand, confidence, reason = trainer.checkpoint_phase(val_loader, verbose=False)
            print(f"  Saturation: max={max(model.metrics.saturation_scores.values(), default=0):.2%}")

            # Décision (combine checkpoint + métriques)
            decision = trainer.decision_phase(
                variant_vote=should_expand,
                variant_confidence=confidence,
                verbose=False
            )

            if decision.should_expand:
                print(f"  → Expansion: {decision.expansion_type} (confidence: {decision.confidence:.2f})")
                trainer.expansion_phase(decision, verbose=False)

                # Warmup si nécessaire
                if trainer.model.warmup_manager.is_warmup_active():
                    trainer.warmup_phase(train_loader, num_steps=20, verbose=False)
                    print(f"  → Warmup effectué")
            else:
                print(f"  → Pas d'expansion ({decision.reason[:50]}...)")

        # Résultats finaux
        print("\n" + "=" * 70)
        print("Test terminé avec succès!")
        print("=" * 70)

        final_params = model.get_total_params()
        print(f"\n✓ Résultats:")
        print(f"  - Paramètres: {initial_params:,} → {final_params:,} ({final_params - initial_params:+,})")
        print(f"  - Blocs: {len(model.task_blocks)}")
        print(f"  - Expansions: {len(model.expansion_manager.expansion_history)}")

        if model.expansion_manager.expansion_history:
            print(f"\n  Historique des expansions:")
            for cycle, target, exp_type in model.expansion_manager.expansion_history:
                print(f"    • Cycle {cycle}: {exp_type} sur {target}")

        print("\n✅ Tous les tests passés!")
        return 0

    except Exception as e:
        print(f"\n❌ Erreur pendant le test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
