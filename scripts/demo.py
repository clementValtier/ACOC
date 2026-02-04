#!/usr/bin/env python3
"""
ACOC - Demonstration (PyTorch)
==============================

Demonstration script for ACOC v0.2 system.
- RELATIVE voting threshold
- Saturation metrics (gradient flow + activations)
- Warmup with forced exploration after expansion

Usage:
    python -m acoc.demo
    python -m acoc.demo --cycles 20 --device cuda
"""

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

from acoc import ACOCModel, ACOCTrainer, SystemConfig


def create_synthetic_data(
    num_train: int = 2000,
    num_val: int = 500,
    input_dim: int = 256,
    output_dim: int = 256,
    batch_size: int = 32,
    complexity: float = 1.0
) -> tuple:
    """
    Create synthetic data for demonstration.

    Data is generated with non-linear transformation
    to simulate a real task.
    """
    # Random transformation matrix (fixed)
    torch.manual_seed(42)
    W1 = torch.randn(input_dim, input_dim) * 0.5
    W2 = torch.randn(input_dim, output_dim) * 0.5

    def generate_batch(n):
        x = torch.randn(n, input_dim)
        # Non-linear transformation: y = relu(x @ W1) @ W2 + noise
        h = torch.relu(x @ W1)
        y = h @ W2
        noise = torch.randn_like(y) * 0.1 * complexity
        return x, y + noise

    # Training data
    train_x, train_y = generate_batch(num_train)
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation data
    val_x, val_y = generate_batch(num_val)
    val_dataset = TensorDataset(val_x, val_y)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="ACOC Demo")
    parser.add_argument("--cycles", type=int, default=15, help="Number of training cycles")
    parser.add_argument("--steps", type=int, default=50, help="Steps per cycle")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 70)
    print("ACOC v0.2 (Adaptive Controlled Organic Capacity) - Demonstration")
    print("=" * 70)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Configuration
    config = SystemConfig(
        # Device
        device=device,

        # Penalties (double penalty)
        alpha_global_penalty=0.01,
        beta_task_penalty=0.05,
        task_param_threshold=500_000,

        # Expansion - threshold based on combined saturation score
        saturation_threshold=0.55,  # 55% of combined score
        min_cycles_before_expand=2,
        expansion_cooldown=3,
        expansion_ratio=0.15,  # 15% of neurons added

        # Variants with RELATIVE threshold
        num_variants=5,
        delta_magnitude=0.01,
        top_k_merge=3,
        performance_threshold_ratio=0.90,  # Expand if < 90% of average

        # Warmup after expansion
        warmup_steps=30,
        warmup_lr_multiplier=3.0,
        new_block_exploration_prob=0.4,  # 40% chance to force
        new_block_exploration_cycles=3,

        # Maintenance
        maintenance_interval=5,
        prune_unused_after_cycles=15,

        # Architecture
        input_dim=256,
        hidden_dim=512,
        output_dim=256,

        # Saturation metrics
        gradient_flow_threshold=1e-6,
        activation_saturation_threshold=0.95,
        dead_neuron_threshold=1e-6
    )

    print("Configuration:")
    print(f"  - Variants: {config.num_variants} (relative threshold: {config.performance_threshold_ratio:.0%})")
    print(f"  - Saturation threshold: {config.saturation_threshold:.0%} of combined score")
    print(f"  - Expansion cooldown: {config.expansion_cooldown} cycles")
    print(f"  - Warmup: {config.warmup_steps} steps, exploration={config.new_block_exploration_prob:.0%}")
    print(f"  - Global penalty (α): {config.alpha_global_penalty}")
    print(f"  - Per-task penalty (β): {config.beta_task_penalty}")
    print()

    # Create data
    print("Generating synthetic data...")
    train_loader, val_loader = create_synthetic_data(
        num_train=2000,
        num_val=500,
        input_dim=config.input_dim,
        output_dim=config.output_dim,
        batch_size=32,
        complexity=1.0
    )
    print(f"  - Train: {len(train_loader.dataset)} samples")
    print(f"  - Val: {len(val_loader.dataset)} samples")
    print()

    # Create model
    print("Initializing model...")
    model = ACOCModel(config)
    print(f"  - Initial blocks: {len(model.task_blocks)}")
    print(f"  - Initial parameters: {model.get_total_params():,}")
    print()

    # Create trainer
    trainer = ACOCTrainer(model, config, learning_rate=0.001)

    # Tracking
    expansion_events = []

    def on_cycle_end(cycle: int, log):
        if log.expanded:
            expansion_events.append({
                'cycle': cycle,
                'type': log.expansion_type,
                'target': log.expansion_target,
                'params': log.total_params
            })

    trainer.on_cycle_end = on_cycle_end

    # Run training
    print(f"Starting training ({args.cycles} cycles)...")
    print("-" * 70)
    
    trainer.run(
        num_cycles=args.cycles,
        data_loader=train_loader,
        validation_data=val_loader,
        num_steps_per_cycle=args.steps,
        verbose=not args.quiet
    )
    
    # Expansion summary
    print("\n" + "=" * 70)
    print("EXPANSION SUMMARY")
    print("=" * 70)

    if expansion_events:
        for event in expansion_events:
            print(f"  Cycle {event['cycle']}: {event['type']} "
                  f"({event['target']}) → {event['params']:,} params")
    else:
        print("  No expansions during this session")

    # Voting statistics
    print("\n" + "=" * 70)
    print("VOTING STATISTICS (RELATIVE THRESHOLD)")
    print("=" * 70)
    vote_summary = model.variant_system.get_vote_summary()
    print(f"  Total votes: {vote_summary['total']}")
    print(f"  Votes for expansion: {vote_summary['expand_votes']}")
    print(f"  Average confidence: {vote_summary['avg_confidence']:.2f}")
    print(f"  Current threshold: {vote_summary['current_threshold']:.3f}")

    # Final saturation metrics
    print("\n" + "=" * 70)
    print("FINAL SATURATION METRICS")
    print("=" * 70)
    for block_id, sat in model.metrics.detailed_saturation.items():
        print(f"  {block_id}:")
        print(f"    - Combined score: {sat.combined_score:.2%}")
        print(f"    - Gradient flow: {sat.gradient_flow_ratio:.2%}")
        print(f"    - Activation sat: {sat.activation_saturation:.2%}")
        print(f"    - Dead neurons: {sat.dead_neuron_ratio:.2%}")
        print(f"    - Variance: {sat.activation_variance:.4f}")

    # Learning curve
    cycles, losses, params = trainer.get_training_curve()

    print("\n" + "=" * 70)
    print("LEARNING CURVE")
    print("=" * 70)
    print("  Cycle | Loss     | Params     | Status")
    print("  ------|----------|------------|--------")
    for i, (c, l, p) in enumerate(zip(cycles, losses, params)):
        status = ""
        for event in expansion_events:
            if event['cycle'] == c:
                status = f"[{event['type']}]"
        if trainer.training_logs[i].warmup_active:
            status += " [W]"
        print(f"  {c:5} | {l:8.4f} | {p:10,} | {status}")

    print("\n" + "=" * 70)
    print("Demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
