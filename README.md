# ACOC - Adaptive Controlled Organic Capacity

Dynamic neural network architecture with controlled expansion, intelligent routing and multi-modal support (Images/Text/Audio).

## 🎯 Concept

ACOC is an AI model that starts with a minimal architecture and progressively grows according to its actual needs, avoiding over-dimensioning while maintaining learning capacity. The system automatically detects the data type (images, text, audio) and uses the appropriate architecture (CNN/MLP).

### ✨ Key Principles

- **Organic Growth**: The model starts small and adds neurons/layers only when necessary
- **Automatic Detection**: Recognizes images/text/audio and applies a slight bias towards the adapted architecture
- **Multi-Modal Support**: Automatic CNNs for images, MLP for text/audio, with intelligent router
- **Double Penalty**: Global penalty (logarithmic) + per-task penalty (quadratic) to enforce sparsity
- **Consensus Voting**: 5 lightweight variants (deltas) vote on expansion decisions with adaptive threshold
- **Anti-Forgetting Protection**: EWC on the router + task block isolation

## 📊 Results

| Dataset | Type | Accuracy | CNN/MLP | Expansions |
|---------|------|----------|---------|------------|
| **MNIST** | Images 28×28 | **~98%+** | CNN 100% | 0 |
| **Fashion-MNIST** | Images 28×28 | **91.15%** | CNN 100% | 0 |
| **CIFAR-10** | Images 32×32×3 | **75.38%** | CNN 82% | 0 |
| **CIFAR-100** | Images 32×32×3 | **~45-50%** | CNN 90%+ | 0-2 |
| **IMDB** | Text (sentiment) | **~85%+** | MLP 100% | 0-2 |
| **Speech Commands** | Audio | **TODO** | TODO | TODO |

The system converges stably without unnecessary expansions, automatically using the appropriate architecture.

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/clementValtier/ACOC.git
cd acoc

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install base dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .

# For text support (IMDB)
pip install datasets transformers

# For audio support (Speech Commands)
pip install torchaudio
```

## 🎮 Quick Start

### Images (MNIST - Digits)
```bash
python3 scripts/train_mnist.py
```

### Images (Fashion-MNIST - Clothing)
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

### Text (IMDB Sentiment Analysis)
```bash
python3 scripts/train_imdb.py
```

### Audio (Speech Commands)
```bash
python3 scripts/train_speech_commands.py
```

## 📖 Advanced Usage

```python
from acoc import ACOCModel, ACOCTrainer, SystemConfig

# Configuration
config = SystemConfig(
    device='cuda',
    input_dim=3072,      # 32×32×3 for CIFAR-10
    hidden_dim=512,
    output_dim=10,
    use_cnn=True,        # Enable CNNs for images
    saturation_threshold=0.8,
    min_cycles_before_expand=10,
    expansion_cooldown=15
)

# Model creation
model = ACOCModel(config)

# The router automatically detects the data type and applies a slight bias
# towards the appropriate architecture (CNN for images, MLP for text/audio)

# Training
trainer = ACOCTrainer(model, config, class_names=['class1', 'class2'])
trainer.train(
    train_loader=train_loader,
    test_loader=test_loader,
    num_cycles=50,
    save_path='model.pth'
)
```

## 🏗️ Architecture

### Project Structure

```
acoc/
├── config/          # Configuration and data structures
├── core/            # Router with automatic data type detection
├── experts/         # BaseExpert, MLPExpert, CNNExpert, ExpertFactory
├── monitoring/      # Gradient and activation monitoring
├── management/      # Expansion, Warmup, Penalty, Pruning
├── variants/        # Variant voting system
├── model/           # Main ACOC model with intelligent routing
├── training/        # Training loop
└── scripts/         # Training scripts for different datasets
```

### Modular Architecture with Factory Pattern

```python
# Modular expert system
BaseExpert (abstract class)
├── MLPExpert        # For text and audio
└── CNNExpert        # For images with automatic dimension detection

# Factory to automatically create the right expert type
expert = ExpertFactory.create(
    expert_type="cnn",  # or "mlp"
    input_dim=3072,
    config=config
)
```

### Automatic Data Type Detection

The router automatically detects the data type by analyzing:

1. **Dimension**: If `input_dim` forms a perfect square (784=28², 3072=32²×3) → **Image**
2. **Statistics**: Distribution, variance, value range → **Text/Audio**

A slight bias (+1.0 to +2.0) is applied towards the appropriate architecture, letting the router learn naturally:

```python
# Automatic detection on first forward
data_type = router.detect_data_type(x)  # "image", "text", or "audio"

# Slight bias towards the appropriate architecture
if data_type == "image":
    router.set_route_bias(base_image_idx, 2.0)  # Direct towards CNN
```

## 🔄 Training Loop

1. **TRAINING**: Fixed architecture, normal backpropagation (5 min per cycle)
2. **CHECKPOINT**: Evaluation + voting from 5 variants (threshold relative to history)
3. **DECISION**: Saturation metrics analysis (gradient flow, activations, dead neurons)
4. **EXPANSION**: Modify architecture if necessary (width/depth/new_block)
5. **WARMUP**: LR × 5 for new parameters + forced exploration (10%)
6. **MAINTENANCE**: Pruning unused blocks + consolidation of similar blocks

## 📈 Saturation Metrics

The system combines 4 metrics to detect expansion needs:

- **Gradient Flow Ratio**: Proportion of "alive" gradients (> threshold)
- **Activation Saturation**: Ratio of saturated neurons (> 95% of max)
- **Dead Neuron Ratio**: Ratio of always-zero neurons
- **Activation Variance**: Inter-batch activation diversity

Weighted combined score: `0.35×gradient + 0.25×saturation + 0.20×dead + 0.20×variance`

## 🔧 Expansion

### Expansion Types

- **Width**: Adding neurons (Net2Net with duplication + noise)
- **Depth**: Adding layers
- **New Block**: Creating a new task block

### Triggers (Recommended Parameters)

- Combined saturation score > **80%** (configurable, increased for stability)
- Minimum **10 cycles** before first expansion (increased patience)
- **15 cycles** cooldown between expansions (stability)
- Stagnant loss (< 1% improvement over 10 cycles)
- Majority vote from variants (consensus)

### Post-Expansion Stabilization

- Learning rate multiplied (×5) for new parameters
- Forced exploration towards new blocks (10% probability)
- Configurable warmup period (50 steps by default)

## 💰 Double Penalty

```python
Loss_total = Loss_task
           + α × log(1 + params_global / params_baseline)
           + β × Σ max(0, params_task_i - threshold_i)²
```

- **α = 0.01**: Global penalty (logarithmic)
- **β = 0.05**: Per-task penalty (quadratic beyond threshold)

The penalty automatically adapts: relaxes if loss stagnates, tightens if rapid improvement.

## 🎲 Variants System

5 lightweight variants of the same model (deltas) to explore the weight space:

```python
model_base = load_model()                    # 1 model in memory
deltas = [small_perturbation() for _ in 5]  # 5 small deltas

# Vote with relative threshold
threshold = 0.95 × mean(last_5_scores)
votes = [evaluate(model + delta) < threshold for delta in deltas]
should_expand = majority(votes)
```

Minimal memory cost: deltas are ~0.1% of the model size.

## 🧠 Catastrophic Forgetting

### Architectural Mitigation

- Separate task blocks (natural isolation)
- Per-task penalty (prevents invasion)
- Addition rather than modification (Progressive Networks style)

### Router Protection

- **EWC (Elastic Weight Consolidation)** on the central router
- Fisher Information Matrix calculated periodically
- Penalty on changes to critical weights

### Maintenance

- **Pruning**: Removal of unused blocks (< 10% usage after 20 cycles)
- **Consolidation**: Merging similar blocks (similarity > 90%)

## ⚙️ Configuration

### Main Hyperparameters (Recommended Values 2026)

```python
SystemConfig(
    # Architecture
    input_dim=3072,              # Depends on dataset
    hidden_dim=512,
    output_dim=10,

    # CNN (for images)
    use_cnn=True,
    cnn_channels=[32, 64, 128],  # CNN structure
    image_channels=3,            # 3 for RGB, 1 for grayscale

    # Expansion (more conservative values for stability)
    saturation_threshold=0.8,         # 80% instead of 60%
    min_cycles_before_expand=10,      # 10 instead of 3
    expansion_cooldown=15,            # 15 instead of 5
    expansion_ratio=0.1,              # Add 10% of neurons
    recent_usage_window=5,            # Window for usage tracking

    # Penalties
    alpha_global_penalty=0.01,        # Global penalty
    beta_task_penalty=0.05,           # Per-task penalty
    task_param_threshold=1_000_000,   # Threshold before penalty

    # Variants
    num_variants=5,                   # 5 variants for voting
    delta_magnitude=0.01,             # Perturbation magnitude
    performance_threshold_ratio=0.95, # Relative threshold (95% mean)

    # Warmup
    warmup_steps=50,                  # Warmup steps
    warmup_lr_multiplier=5.0,         # LR × 5 for new params
    new_block_exploration_prob=0.1,   # 10% exploration (reduced)
    new_block_exploration_cycles=3,   # Exploration cycles
    max_warmup_cycles=10,             # Max cycles before deactivation

    # Maintenance
    prune_unused_after_cycles=20,
    consolidation_similarity_threshold=0.9,
    maintenance_interval=5,

    # Device
    device='cuda'  # 'cuda', 'mps', or 'cpu'
)
```

## 📝 Adding a New Dataset

All training scripts use `BaseACOCTrainer` to factor out common code. To add a dataset:

```python
from scripts.base_trainer import BaseACOCTrainer
from acoc import SystemConfig

class MyTrainer(BaseACOCTrainer):
    CLASSES = ['ClassA', 'ClassB']

    def get_config(self):
        return SystemConfig(
            device=self.device,
            input_dim=1000,
            output_dim=2,
            use_cnn=False  # True for images
        )

    def get_dataloaders(self):
        # Load and return (train_loader, test_loader)
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

See `scripts/README.md` for more details.

## 🧪 Tests

```bash
# Run all tests with the global test runner
python run_tests.py

# Run a specific test suite
python run_tests.py --suite config
python run_tests.py --suite router
python run_tests.py --suite experts
python run_tests.py --suite model

# Run with coverage report
python run_tests.py --coverage

# List all available test suites
python run_tests.py --list

# Direct pytest usage (alternative)
pytest tests/ -v

# Specific test file
pytest tests/test_expansion.py -v
```

## 📚 References

### Concepts Used

- **NEAT** (Stanley, 2002): Neuroevolution with augmented topology
- **Net2Net** (Chen et al., 2015): Function-preserving network expansion
- **LEMON** (ICLR 2024): Lossless expansion for Transformers
- **Mixture of Experts**: GPT-4, Mixtral, DeepSeek-V3
- **Model Soups** (2022): Weight averaging without inference cost
- **Progressive Neural Networks** (DeepMind): Anti-forgetting through column addition
- **EWC** (Kirkpatrick et al., 2017): Elastic Weight Consolidation

### State of the Art

- **DynMoE** (ICLR 2025): Dynamic adjustment of expert count
- **Growth-based NAS**: Layer-by-layer construction
- **Continual Learning**: CoMA/CoFiMA with Fisher information
- **Multimodal Unified Models** (2024-2025): GPT-4o, Gemini

## 🎯 Roadmap

- [x] Automatic CNN support for images
- [x] Automatic data type detection
- [x] Factory pattern for modular experts
- [x] Refactored training scripts
- [x] Multi-modal support (Images/Text/Audio)
- [ ] Multi-GPU support (DataParallel/DistributedDataParallel)
- [ ] Benchmark vs baselines (static MoE, Progressive Networks)
- [ ] Inter-branch sharing mechanism
- [ ] Support for transformers and attention

## 📄 License

MIT

## 👥 Contact

ACOC Project - v0.3.0 (2026)

Author: Clément Valtier
