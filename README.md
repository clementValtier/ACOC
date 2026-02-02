# ACOC - Adaptive Controlled Organic Capacity

Architecture de réseau neuronal à croissance dynamique avec expansion contrôlée et double pénalité de taille.

## Concept

ACOC est un modèle d'IA qui démarre avec une architecture minimale et s'agrandit progressivement selon ses besoins réels, évitant le sur-dimensionnement tout en maintenant la capacité d'apprentissage.

### Principes Clés

- **Croissance Organique** : Le modèle commence petit et ajoute des neurones/couches uniquement quand nécessaire
- **Double Malus** : Pénalité globale (logarithmique) + pénalité par tâche (quadratique) pour forcer la parcimonie
- **Architecture Modulaire** : Structure en arbre avec branches spécialisées (texte, image, audio) contrôlées par un routeur central
- **Vote par Consensus** : 5 variantes légères (deltas) votent sur les décisions d'expansion avec seuil adaptatif
- **Protection Anti-Forgetting** : EWC sur le routeur + isolation des blocs de tâches

## Installation

```bash
# Cloner le repository
git clone https://github.com/clementValtier/ACOC.git
cd ACOC

# Installer en mode développement
pip install -e .

# Ou installer uniquement les dépendances
pip install -r requirements.txt
```

## Usage

```python
from acoc import ACOCModel, ACOCTrainer, SystemConfig

# Configuration
config = SystemConfig(
    device='cuda',
    input_dim=256,
    hidden_dim=512,
    num_variants=5,
    saturation_threshold=0.6
)

# Création du modèle
model = ACOCModel(config)

# Entraînement
trainer = ACOCTrainer(model, config, learning_rate=0.001)
trainer.run(num_cycles=20, num_steps=100)
```

## Architecture

```
acoc/
├── config/          Configuration et structures de données
├── core/            Router + Expert (composants de base)
├── monitoring/      Monitoring des gradients et activations
├── management/      Expansion, Warmup, Penalty, Pruning
├── variants/        Système de vote par variantes
├── model/           Modèle ACOC principal
└── training/        Boucle d'entraînement
```

## Boucle d'Entraînement

1. **TRAINING** : Architecture fixe, backpropagation normale (5 min - quelques heures)
2. **CHECKPOINT** : Évaluation + vote des 5 variantes (seuil relatif à l'historique)
3. **DÉCISION** : Analyse des métriques de saturation (gradient flow, activations, neurones morts)
4. **EXPANSION** : Modification de l'architecture si nécessaire (width/depth/new_block)
5. **WARMUP** : LR × 5 pour nouveaux paramètres + exploration forcée (30%)
6. **MAINTENANCE** : Pruning des blocs inutilisés + consolidation des blocs similaires

## Métriques de Saturation

Le système combine 4 métriques pour détecter le besoin d'expansion :

- **Gradient Flow Ratio** : Proportion de gradients "vivants" (> seuil)
- **Activation Saturation** : Ratio de neurones saturés (> 95% du max)
- **Dead Neuron Ratio** : Ratio de neurones toujours à 0
- **Activation Variance** : Diversité des activations inter-batch

Score combiné pondéré : `0.35×gradient + 0.25×saturation + 0.20×dead + 0.20×variance`

## Expansion

### Types d'Expansion

- **Width** : Ajout de neurones (Net2Net avec duplication + bruit)
- **Depth** : Ajout de couches
- **New Block** : Création d'un nouveau bloc de tâche

### Déclencheurs

- Score de saturation combiné > 60% (configurable)
- Loss stagnante (< 1% d'amélioration sur 10 cycles)
- Vote majoritaire des variantes (consensus)

### Stabilisation Post-Expansion

- Learning rate multiplié (×5) pour nouveaux paramètres
- Exploration forcée vers nouveaux blocs (30% de probabilité)
- Période de warmup configurable (50 steps par défaut)

## Double Malus

```python
Loss_total = Loss_task 
           + α × log(1 + params_global / params_baseline)
           + β × Σ max(0, params_task_i - threshold_i)²
```

- **α = 0.01** : Pénalité globale (logarithmique)
- **β = 0.05** : Pénalité par tâche (quadratique au-delà du seuil)

Le malus s'adapte automatiquement : se relâche si la loss stagne, se resserre si amélioration rapide.

## Système de Variantes

5 variantes légères du même modèle (deltas) pour explorer l'espace des poids :

```python
model_base = load_model()                    # 1 modèle en mémoire
deltas = [small_perturbation() for _ in 5]  # 5 petits deltas

# Vote avec seuil relatif
threshold = 0.95 × mean(last_5_scores)
votes = [evaluate(model + delta) < threshold for delta in deltas]
should_expand = majority(votes)
```

Coût mémoire minimal : les deltas sont ~0.1% de la taille du modèle.

## Catastrophic Forgetting

### Mitigation Architecturale

- Blocs de tâches séparés (isolation naturelle)
- Malus par tâche (empêche l'invasion)
- Ajout plutôt que modification (Progressive Networks style)

### Protection du Routeur

- **EWC (Elastic Weight Consolidation)** sur le routeur central
- Fisher Information Matrix calculée périodiquement
- Pénalité sur les changements des poids critiques

### Maintenance

- **Pruning** : Suppression des blocs inutilisés (< 10% utilisation après 20 cycles)
- **Consolidation** : Fusion de blocs similaires (similarité > 90%)

## Configuration

### Hyperparamètres Principaux

```python
SystemConfig(
    # Timing
    training_cycle_duration=300,      # 5 minutes par cycle
    checkpoint_interval=1,            # Checkpoint chaque cycle
    
    # Pénalités
    alpha_global_penalty=0.01,        # Pénalité globale
    beta_task_penalty=0.05,           # Pénalité par tâche
    task_param_threshold=1_000_000,   # Seuil avant pénalité
    
    # Expansion
    saturation_threshold=0.6,         # Trigger si score > 60%
    min_cycles_before_expand=3,       # Attendre au moins 3 cycles
    expansion_cooldown=5,             # 5 cycles entre expansions
    expansion_ratio=0.1,              # Ajouter 10% de neurones
    
    # Variantes
    num_variants=5,                   # 5 variantes pour le vote
    delta_magnitude=0.01,             # Amplitude des perturbations
    performance_threshold_ratio=0.95, # Seuil relatif (95% moyenne)
    
    # Warmup
    warmup_steps=50,                  # Steps de warmup
    warmup_lr_multiplier=5.0,         # LR × 5 pour nouveaux params
    new_block_exploration_prob=0.3,   # 30% exploration forcée
    
    # Architecture
    input_dim=256,
    hidden_dim=512,
    output_dim=256,
    
    # Device
    device='cuda'
)
```

## Exemple Complet

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from acoc import ACOCModel, ACOCTrainer, SystemConfig

# Données
X_train = torch.randn(1000, 256)
y_train = torch.randn(1000, 256)
train_loader = DataLoader(
    TensorDataset(X_train, y_train), 
    batch_size=32, 
    shuffle=True
)

# Configuration
config = SystemConfig(
    device='cuda' if torch.cuda.is_available() else 'cpu',
    saturation_threshold=0.6,
    expansion_ratio=0.1
)

# Modèle et trainer
model = ACOCModel(config)
trainer = ACOCTrainer(model, config, learning_rate=0.001)

# Entraînement avec monitoring
for cycle in range(20):
    # Phase de training
    avg_loss = trainer.training_phase(train_loader, num_steps=100)
    
    # Phase de checkpoint et décision
    decision = trainer.checkpoint_phase(train_loader)
    
    if decision.should_expand:
        print(f"Cycle {cycle}: Expansion ({decision.expansion_type})")
        trainer.expansion_phase(decision)
        trainer.warmup_phase(train_loader)
    
    # Maintenance périodique
    if cycle % 5 == 0:
        trainer.maintenance_phase()

# Statistiques finales
print(f"Paramètres finaux: {model.get_total_params():,}")
print(f"Nombre de blocs: {len(model.task_blocks)}")
print(f"Expansions: {trainer.expansion_manager.get_expansion_stats()}")
```

## Références

### Concepts Utilisés

- **NEAT** (Stanley, 2002) : Neuroévolution avec topologie augmentée
- **Net2Net** (Chen et al., 2015) : Expansion de réseaux préservant la fonction
- **LEMON** (ICLR 2024) : Expansion lossless pour Transformers
- **Mixture of Experts** : GPT-4, Mixtral, DeepSeek-V3
- **Model Soups** (2022) : Moyennage de poids sans coût d'inférence
- **Progressive Neural Networks** (DeepMind) : Anti-forgetting par ajout de colonnes
- **EWC** (Kirkpatrick et al., 2017) : Elastic Weight Consolidation

### État de l'Art

- **DynMoE** (ICLR 2025) : Ajustement dynamique du nombre d'experts
- **Growth-based NAS** : Construction layer-by-layer
- **Continual Learning** : CoMA/CoFiMA avec Fisher information
- **Multimodal Unified Models** (2024-2025) : GPT-4o, Gemini

## Limitations Actuelles

- Données synthétiques uniquement (pas de tests sur MNIST/CIFAR-10)
- Support GPU basique (pas de parallélisation multi-GPU)
- Pas de benchmark vs baselines (MoE statique, Progressive Networks)
- Pas de mécanisme de partage inter-branches

## Licence

MIT

## Contact

ACOC Project - v0.2.0
