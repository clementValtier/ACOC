#!/usr/bin/env python3
"""
Quick Test - ACOC
=================
Quick test to validate that the basic loop works.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

from acoc import ACOCModel, ACOCTrainer, SystemConfig


def create_synthetic_data(num_samples=200, input_dim=256, output_dim=256, batch_size=32):
    """Create synthetic data for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randn(num_samples, output_dim)
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    print("=" * 70)
    print("ACOC - Quick Test")
    print("=" * 70)

    # Minimal configuration
    config = SystemConfig(
        device='cpu',
        input_dim=256,
        hidden_dim=128,
        output_dim=256,
        num_variants=3,  # Reduced for faster execution
        saturation_threshold=0.55,
        min_cycles_before_expand=2,
        expansion_cooldown=3,
        performance_threshold_ratio=0.90
    )

    print(f"\n✓ Configuration created")
    print(f"  - Device: {config.device}")
    print(f"  - Hidden dim: {config.hidden_dim}")
    print(f"  - Variants: {config.num_variants}")

    # Data
    train_loader = create_synthetic_data(num_samples=200)
    val_loader = create_synthetic_data(num_samples=50)

    print(f"✓ Synthetic data created")
    print(f"  - Train: 200 samples")
    print(f"  - Val: 50 samples")

    # Model
    model = ACOCModel(config)
    initial_params = model.get_total_params()

    print(f"✓ Model created")
    print(f"  - Initial parameters: {initial_params:,}")
    print(f"  - Initial blocks: {len(model.task_blocks)}")

    # Trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    print(f"✓ Trainer created")
    print(f"  - Learning rate: 0.001")

    # Quick test over 5 cycles
    print("\n" + "=" * 70)
    print("Starting quick test (5 cycles)...")
    print("=" * 70)

    try:
        for cycle in range(5):
            print(f"\n--- Cycle {cycle} ---")

            # Training
            avg_loss = trainer.training_phase(train_loader, num_steps=10, verbose=False)
            print(f"  Training loss: {avg_loss:.4f}")

            # Checkpoint (returns tuple)
            should_expand, confidence, reason = trainer.checkpoint_phase(val_loader, verbose=False)
            print(f"  Saturation: max={max(model.metrics.saturation_scores.values(), default=0):.2%}")

            # Decision (combines checkpoint + metrics)
            decision = trainer.decision_phase(
                variant_vote=should_expand,
                variant_confidence=confidence,
                verbose=False
            )

            if decision.should_expand:
                print(f"  → Expansion: {decision.expansion_type} (confidence: {decision.confidence:.2f})")
                trainer.expansion_phase(decision, verbose=False)

                # Check warmup status (warmup is automatically handled in training loop)
                warmup_blocks = trainer.model.warmup_manager.get_warmup_blocks()
                if warmup_blocks:
                    print(f"  → Warmup active for blocks: {', '.join(warmup_blocks)}")
            else:
                print(f"  → No expansion ({decision.reason[:50]}...)")

        # Final results
        print("\n" + "=" * 70)
        print("Test completed successfully!")
        print("=" * 70)

        final_params = model.get_total_params()
        print(f"\n✓ Results:")
        print(f"  - Parameters: {initial_params:,} → {final_params:,} ({final_params - initial_params:+,})")
        print(f"  - Blocks: {len(model.task_blocks)}")
        print(f"  - Expansions: {len(model.expansion_manager.expansion_history)}")

        if model.expansion_manager.expansion_history:
            print(f"\n  Expansion history:")
            for cycle, target, exp_type in model.expansion_manager.expansion_history:
                print(f"    • Cycle {cycle}: {exp_type} on {target}")

        print("\n✅ All tests passed!")
        return 0

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
