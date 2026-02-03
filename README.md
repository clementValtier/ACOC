# ACOC - Adaptive Controlled Organic Capacity

Architecture de réseau neuronal à croissance dynamique avec expansion contrôlée, routage intelligent et support multi-modal (Images/Texte/Audio).

## 🎯 Concept

ACOC est un modèle d'IA qui démarre avec une architecture minimale et s'agrandit progressivement selon ses besoins réels, évitant le sur-dimensionnement tout en maintenant la capacité d'apprentissage. Le système détecte automatiquement le type de données (images, texte, audio) et utilise l'architecture appropriée (CNN/MLP).

### ✨ Principes Clés

- **Croissance Organique** : Le modèle commence petit et ajoute des neurones/couches uniquement quand nécessaire
- **Détection Automatique** : Reconnaît images/texte/audio et applique un biais léger vers l'architecture adaptée
- **Support Multi-Modal** : CNN automatiques pour images, MLP pour texte/audio, avec routeur intelligent
- **Double Malus** : Pénalité globale (logarithmique) + pénalité par tâche (quadratique) pour forcer la parcimonie
- **Vote par Consensus** : 5 variantes légères (deltas) votent sur les décisions d'expansion avec seuil adaptatif
- **Protection Anti-Forgetting** : EWC sur le routeur + isolation des blocs de tâches

## 📊 Résultats

| Dataset | Type | Accuracy | CNN/MLP | Expansions |
|---------|------|----------|---------|------------|
| **MNIST** | Images 28×28 | **~98%+** | CNN 100% | 0 |
| **Fashion-MNIST** | Images 28×28 | **91.15%** | CNN 100% | 0 |
| **CIFAR-10** | Images 32×32×3 | **75.38%** | CNN 82% | 0 |
| **CIFAR-100** | Images 32×32×3 | **~45-50%** | CNN 90%+ | 0-2 |
| **IMDB** | Texte (sentiment) | **~85%+** | MLP 100% | 0-2 |
| **Speech Commands** | Audio | **~85%+** | MLP 100% | 0-2 |

Le système converge de manière stable sans expansions inutiles, en utilisant l'architecture appropriée automatiquement.

## 🚀 Installation

```bash
# Cloner le repository
git clone https://github.com/clementValtier/ACOC.git
cd acoc

# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` sur Windows

# Installer les dépendances de base
pip install -r requirements.txt

# Installer en mode développement
pip install -e .

# Pour le support texte (IMDB)
pip install datasets transformers

# Pour le support audio (Speech Commands)
pip install torchaudio
```

## 🎮 Quick Start

### Images (MNIST - Chiffres)
```bash
python3 scripts/train_mnist.py
```

### Images (Fashion-MNIST - Vêtements)
```bash
python3 scripts/train_fashion.py
```

### Images (CIFAR-10 - 10 classes)
```bash
python3 scripts/train_cifar10.py
```

### Images (CIFAR-100 - 100 classes)
```bash
python3 scripts/train_cifar100.py
```

### Texte (IMDB Sentiment Analysis)
```bash
python3 scripts/train_imdb.py
```

### Audio (Speech Commands)
```bash
python3 scripts/train_speech_commands.py
```

## 📖 Usage Avancé

```python
from acoc import ACOCModel, ACOCTrainer, SystemConfig

# Configuration
config = SystemConfig(
    device='cuda',
    input_dim=3072,      # 32×32×3 pour CIFAR-10
    hidden_dim=512,
    output_dim=10,
    use_cnn=True,        # Active les CNN pour images
    saturation_threshold=0.8,
    min_cycles_before_expand=10,
    expansion_cooldown=15
)

# Création du modèle
model = ACOCModel(config)

# Le routeur détecte automatiquement le type de données et applique un biais léger
# vers l'architecture appropriée (CNN pour images, MLP pour texte/audio)

# Entraînement
trainer = ACOCTrainer(model, config, class_names=['class1', 'class2'])
trainer.train(
    train_loader=train_loader,
    test_loader=test_loader,
    num_cycles=50,
    save_path='model.pth'
)
```

## 🏗️ Architecture

### Structure du Projet

```
acoc/
├── config/          # Configuration et structures de données
├── core/            # Router avec détection automatique du type de données
├── experts/         # BaseExpert, MLPExpert, CNNExpert, ExpertFactory
├── monitoring/      # Monitoring des gradients et activations
├── management/      # Expansion, Warmup, Penalty, Pruning
├── variants/        # Système de vote par variantes
├── model/           # Modèle ACOC principal avec routage intelligent
├── training/        # Boucle d'entraînement
└── scripts/         # Scripts de training pour différents datasets
```

### Architecture Modulaire avec Factory Pattern

```python
# Système d'experts modulaire
BaseExpert (classe abstraite)
├── MLPExpert        # Pour texte et audio
└── CNNExpert        # Pour images avec détection auto des dimensions

# Factory pour créer automatiquement le bon type d'expert
expert = ExpertFactory.create(
    expert_type="cnn",  # ou "mlp"
    input_dim=3072,
    config=config
)
```

### Détection Automatique du Type de Données

Le routeur détecte automatiquement le type de données en analysant :

1. **Dimension** : Si `input_dim` forme un carré parfait (784=28², 3072=32²×3) → **Image**
2. **Statistiques** : Distribution, variance, plage de valeurs → **Texte/Audio**

Un biais léger (+1.0 à +2.0) est appliqué vers l'architecture appropriée, laissant le routeur apprendre naturellement :

```python
# Détection automatique au premier forward
data_type = router.detect_data_type(x)  # "image", "text", ou "audio"

# Biais léger vers l'architecture appropriée
if data_type == "image":
    router.set_route_bias(base_image_idx, 2.0)  # Oriente vers CNN
```

## 🔄 Boucle d'Entraînement

1. **TRAINING** : Architecture fixe, backpropagation normale (5 min par cycle)
2. **CHECKPOINT** : Évaluation + vote des 5 variantes (seuil relatif à l'historique)
3. **DÉCISION** : Analyse des métriques de saturation (gradient flow, activations, neurones morts)
4. **EXPANSION** : Modification de l'architecture si nécessaire (width/depth/new_block)
5. **WARMUP** : LR × 5 pour nouveaux paramètres + exploration forcée (10%)
6. **MAINTENANCE** : Pruning des blocs inutilisés + consolidation des blocs similaires

## 📈 Métriques de Saturation

Le système combine 4 métriques pour détecter le besoin d'expansion :

- **Gradient Flow Ratio** : Proportion de gradients "vivants" (> seuil)
- **Activation Saturation** : Ratio de neurones saturés (> 95% du max)
- **Dead Neuron Ratio** : Ratio de neurones toujours à 0
- **Activation Variance** : Diversité des activations inter-batch

Score combiné pondéré : `0.35×gradient + 0.25×saturation + 0.20×dead + 0.20×variance`

## 🔧 Expansion

### Types d'Expansion

- **Width** : Ajout de neurones (Net2Net avec duplication + bruit)
- **Depth** : Ajout de couches
- **New Block** : Création d'un nouveau bloc de tâche

### Déclencheurs (Paramètres Recommandés)

- Score de saturation combiné > **80%** (configurable, augmenté pour stabilité)
- Minimum **10 cycles** avant première expansion (patience accrue)
- **15 cycles** de cooldown entre expansions (stabilité)
- Loss stagnante (< 1% d'amélioration sur 10 cycles)
- Vote majoritaire des variantes (consensus)

### Stabilisation Post-Expansion

- Learning rate multiplié (×5) pour nouveaux paramètres
- Exploration forcée vers nouveaux blocs (10% de probabilité)
- Période de warmup configurable (50 steps par défaut)

## 💰 Double Malus

```python
Loss_total = Loss_task
           + α × log(1 + params_global / params_baseline)
           + β × Σ max(0, params_task_i - threshold_i)²
```

- **α = 0.01** : Pénalité globale (logarithmique)
- **β = 0.05** : Pénalité par tâche (quadratique au-delà du seuil)

Le malus s'adapte automatiquement : se relâche si la loss stagne, se resserre si amélioration rapide.

## 🎲 Système de Variantes

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

## 🧠 Catastrophic Forgetting

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

## ⚙️ Configuration

### Hyperparamètres Principaux (Valeurs Recommandées 2026)

```python
SystemConfig(
    # Architecture
    input_dim=3072,              # Dépend du dataset
    hidden_dim=512,
    output_dim=10,

    # CNN (pour images)
    use_cnn=True,
    cnn_channels=[32, 64, 128],  # Structure CNN
    image_channels=3,            # 3 pour RGB, 1 pour grayscale

    # Expansion (valeurs plus conservatrices pour stabilité)
    saturation_threshold=0.8,         # 80% au lieu de 60%
    min_cycles_before_expand=10,      # 10 au lieu de 3
    expansion_cooldown=15,            # 15 au lieu de 5
    expansion_ratio=0.1,              # Ajouter 10% de neurones
    recent_usage_window=5,            # Fenêtre pour utilisation

    # Pénalités
    alpha_global_penalty=0.01,        # Pénalité globale
    beta_task_penalty=0.05,           # Pénalité par tâche
    task_param_threshold=1_000_000,   # Seuil avant pénalité

    # Variantes
    num_variants=5,                   # 5 variantes pour le vote
    delta_magnitude=0.01,             # Amplitude des perturbations
    performance_threshold_ratio=0.95, # Seuil relatif (95% moyenne)

    # Warmup
    warmup_steps=50,                  # Steps de warmup
    warmup_lr_multiplier=5.0,         # LR × 5 pour nouveaux params
    new_block_exploration_prob=0.1,   # 10% exploration (réduit)
    new_block_exploration_cycles=3,   # Cycles d'exploration
    max_warmup_cycles=10,             # Cycles max avant désactivation

    # Maintenance
    prune_unused_after_cycles=20,
    consolidation_similarity_threshold=0.9,
    maintenance_interval=5,

    # Device
    device='cuda'  # 'cuda', 'mps', ou 'cpu'
)
```

## 📝 Ajouter un Nouveau Dataset

Tous les scripts de training utilisent `BaseACOCTrainer` pour factoriser le code commun. Pour ajouter un dataset :

```python
from scripts.base_trainer import BaseACOCTrainer
from acoc import SystemConfig

class MyTrainer(BaseACOCTrainer):
    CLASSES = ['ClasseA', 'ClasseB']

    def get_config(self):
        return SystemConfig(
            device=self.device,
            input_dim=1000,
            output_dim=2,
            use_cnn=False  # True pour images
        )

    def get_dataloaders(self):
        # Charger et retourner (train_loader, test_loader)
        return train_loader, test_loader

    def get_class_names(self):
        return self.CLASSES

    def get_dataset_name(self):
        return "my_dataset"

    def get_dataset_info(self):
        return {"Input": 1000, "Classes": "A, B"}

if __name__ == '__main__':
    trainer = MyTrainer(num_cycles=50)
    trainer.run()
```

Voir `scripts/README.md` pour plus de détails.

## 🧪 Tests

```bash
# Tests unitaires
pytest tests/

# Test spécifique
pytest tests/test_expansion.py -v
```

## 📚 Références

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

## 🎯 Roadmap

- [x] Support CNN automatique pour images
- [x] Détection automatique du type de données
- [x] Factory pattern pour experts modulaires
- [x] Scripts de training refactorisés
- [x] Support multi-modal (Images/Texte/Audio)
- [ ] Support GPU multi-GPU (DataParallel/DistributedDataParallel)
- [ ] Benchmark vs baselines (MoE statique, Progressive Networks)
- [ ] Mécanisme de partage inter-branches
- [ ] Support pour transformers et attention

## 📄 Licence

MIT

## 👥 Contact

ACOC Project - v0.3.0 (2026)

Auteur : Clément Valtier
