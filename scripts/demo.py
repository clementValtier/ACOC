#!/usr/bin/env python3
"""
ACOC - Démonstration (PyTorch)
==============================

Script de démonstration du système ACOC v0.2.
- Seuil de vote RELATIF
- Métriques de saturation (gradient flow + activations)
- Warmup avec exploration forcée après expansion

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
    Crée des données synthétiques pour la démonstration.
    
    Les données sont générées avec une transformation non-linéaire
    pour simuler une tâche réelle.
    """
    # Matrice de transformation aléatoire (fixe)
    torch.manual_seed(42)
    W1 = torch.randn(input_dim, input_dim) * 0.5
    W2 = torch.randn(input_dim, output_dim) * 0.5
    
    def generate_batch(n):
        x = torch.randn(n, input_dim)
        # Transformation non-linéaire: y = relu(x @ W1) @ W2 + noise
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
    
    # Déterminer le device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 70)
    print("ACOC v0.2 (Adaptive Controlled Organic Capacity) - Démonstration")
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
        
        # Pénalités (double malus)
        alpha_global_penalty=0.01,
        beta_task_penalty=0.05,
        task_param_threshold=500_000,
        
        # Expansion - seuil basé sur score de saturation combiné
        saturation_threshold=0.55,  # 55% du score combiné
        min_cycles_before_expand=2,
        expansion_cooldown=3,
        expansion_ratio=0.15,  # 15% de neurones ajoutés

        # Variantes avec seuil RELATIF
        num_variants=5,
        delta_magnitude=0.01,
        top_k_merge=3,
        performance_threshold_ratio=0.90,  # Expand si < 90% de la moyenne
        
        # Warmup après expansion
        warmup_steps=30,
        warmup_lr_multiplier=3.0,
        new_block_exploration_prob=0.4,  # 40% de chances de forcer
        new_block_exploration_cycles=3,
        
        # Maintenance
        maintenance_interval=5,
        prune_unused_after_cycles=15,
        
        # Architecture
        input_dim=256,
        hidden_dim=512,
        output_dim=256,
        
        # Métriques de saturation
        gradient_flow_threshold=1e-6,
        activation_saturation_threshold=0.95,
        dead_neuron_threshold=1e-6
    )
    
    print("Configuration:")
    print(f"  - Variantes: {config.num_variants} (seuil relatif: {config.performance_threshold_ratio:.0%})")
    print(f"  - Seuil saturation: {config.saturation_threshold:.0%} du score combiné")
    print(f"  - Cooldown expansion: {config.expansion_cooldown} cycles")
    print(f"  - Warmup: {config.warmup_steps} steps, exploration={config.new_block_exploration_prob:.0%}")
    print(f"  - Pénalité globale (α): {config.alpha_global_penalty}")
    print(f"  - Pénalité par tâche (β): {config.beta_task_penalty}")
    print()
    
    # Créer les données
    print("Génération des données synthétiques...")
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
    
    # Créer le modèle
    print("Initialisation du modèle...")
    model = ACOCModel(config)
    print(f"  - Blocs initiaux: {len(model.task_blocks)}")
    print(f"  - Paramètres initiaux: {model.get_total_params():,}")
    print()
    
    # Créer le trainer
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
    
    # Exécuter l'entraînement
    print(f"Démarrage de l'entraînement ({args.cycles} cycles)...")
    print("-" * 70)
    
    trainer.run(
        num_cycles=args.cycles,
        data_loader=train_loader,
        validation_data=val_loader,
        num_steps_per_cycle=args.steps,
        verbose=not args.quiet
    )
    
    # Résumé des expansions
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES EXPANSIONS")
    print("=" * 70)
    
    if expansion_events:
        for event in expansion_events:
            print(f"  Cycle {event['cycle']}: {event['type']} "
                  f"({event['target']}) → {event['params']:,} params")
    else:
        print("  Aucune expansion durant cette session")
    
    # Statistiques de vote
    print("\n" + "=" * 70)
    print("STATISTIQUES DE VOTE (SEUIL RELATIF)")
    print("=" * 70)
    vote_summary = model.variant_system.get_vote_summary()
    print(f"  Total votes: {vote_summary['total']}")
    print(f"  Votes pour expansion: {vote_summary['expand_votes']}")
    print(f"  Confiance moyenne: {vote_summary['avg_confidence']:.2f}")
    print(f"  Seuil actuel: {vote_summary['current_threshold']:.3f}")
    
    # Métriques de saturation finales
    print("\n" + "=" * 70)
    print("MÉTRIQUES DE SATURATION FINALES")
    print("=" * 70)
    for block_id, sat in model.metrics.detailed_saturation.items():
        print(f"  {block_id}:")
        print(f"    - Score combiné: {sat.combined_score:.2%}")
        print(f"    - Gradient flow: {sat.gradient_flow_ratio:.2%}")
        print(f"    - Activation sat: {sat.activation_saturation:.2%}")
        print(f"    - Neurones morts: {sat.dead_neuron_ratio:.2%}")
        print(f"    - Variance: {sat.activation_variance:.4f}")
    
    # Courbe d'apprentissage
    cycles, losses, params = trainer.get_training_curve()
    
    print("\n" + "=" * 70)
    print("COURBE D'APPRENTISSAGE")
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
    print("Démonstration terminée!")
    print("=" * 70)


if __name__ == "__main__":
    main()
