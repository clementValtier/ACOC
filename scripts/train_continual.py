"""
ACOC - Continual Learning Trainer V2
=====================================
Enhanced continual learning with:
- Native input dimensions (no forced resizing)
- Modality projections
- Experience replay
- Task-specific blocks

Train ACOC on multiple datasets sequentially to test continual learning capabilities.
"""

import sys
import random
from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from acoc import (
    ACOCModel,
    SystemConfig,
    ContinualACOCTrainer,
    ModalityProjector
)
from base_trainer import create_onehot_collate_fn


class ContinualLearningTrainer:
    """Enhanced continual learning trainer with projections and replay."""

    # Dataset configurations with NATIVE dimensions
    DATASET_CONFIGS = {
        'mnist': {
            'name': 'MNIST',
            'loader': torchvision.datasets.MNIST,
            'input_dim': 784,  # Native 28x28
            'image_size': (28, 28),
            'output_dim': 10,
            'image_channels': 1,
            'modality_type': 'image',
            'transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(torch.flatten)
            ])
        },
        'fashion': {
            'name': 'Fashion-MNIST',
            'loader': torchvision.datasets.FashionMNIST,
            'input_dim': 784,  # Native 28x28
            'image_size': (28, 28),
            'output_dim': 10,
            'image_channels': 1,
            'modality_type': 'image',
            'transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
                transforms.Lambda(torch.flatten)
            ])
        },
        'cifar10': {
            'name': 'CIFAR-10',
            'loader': torchvision.datasets.CIFAR10,
            'input_dim': 3072,  # Native 32x32x3
            'image_size': (32, 32),
            'output_dim': 10,
            'image_channels': 3,
            'modality_type': 'image',
            'transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                transforms.Lambda(torch.flatten)
            ])
        }
    }

    def __init__(
        self,
        datasets: list[str] | None = None,
        samples_per_dataset: int = 5000,
        num_cycles_per_dataset: int = 10,
        batch_size: int = 32,
        device: str = 'cpu',
        shuffle_order: bool = True,
        replay_buffer_size: int = 2000,
        replay_samples_per_task: int = 500
    ):
        """
        Args:
            datasets: List of dataset names to use
            samples_per_dataset: Number of samples per dataset
            num_cycles_per_dataset: Training cycles per dataset
            batch_size: Batch size
            device: Device to use
            shuffle_order: Whether to shuffle training order
            replay_buffer_size: Max examples in replay buffer
            replay_samples_per_task: Samples to store per task
        """
        self.datasets = datasets or list(self.DATASET_CONFIGS.keys())
        self.samples_per_dataset = samples_per_dataset
        self.num_cycles_per_dataset = num_cycles_per_dataset
        self.batch_size = batch_size
        self.device = device
        self.shuffle_order = shuffle_order
        self.replay_buffer_size = replay_buffer_size
        self.replay_samples_per_task = replay_samples_per_task

        # Validate datasets
        for ds in self.datasets:
            if ds not in self.DATASET_CONFIGS:
                raise ValueError(f"Unknown dataset: {ds}")

        # Training order
        self.training_order = self.datasets.copy()
        if self.shuffle_order:
            random.shuffle(self.training_order)

        # Determine unified dimension (use maximum input dim)
        self.unified_dim = max(
            self.DATASET_CONFIGS[ds]['input_dim']
            for ds in self.datasets
        )

        print("=" * 70)
        print("ACOC Continual Learning")
        print("=" * 70)
        print(f"Datasets: {', '.join(self.datasets)}")
        print(f"Training order: {' → '.join(self.training_order)}")
        print(f"Unified dimension: {self.unified_dim}")
        print(f"Samples per dataset: {self.samples_per_dataset}")
        print(f"Cycles per dataset: {self.num_cycles_per_dataset}")
        print(f"Replay buffer: {self.replay_buffer_size} (storing {self.replay_samples_per_task}/task)")
        print(f"Device: {self.device}")
        print("=" * 70)

    def _load_dataset(self, dataset_name: str, train: bool = True):
        """Load dataset with native dimensions."""
        config = self.DATASET_CONFIGS[dataset_name]

        # Load dataset
        dataset = config['loader'](
            root='./data',
            train=train,
            download=True,
            transform=config['transform']
        )

        # Limit samples
        if len(dataset) > self.samples_per_dataset:
            indices = random.sample(range(len(dataset)), self.samples_per_dataset)
            dataset = Subset(dataset, indices)

        # Collate function
        collate_fn = create_onehot_collate_fn(config['output_dim'])

        # DataLoader
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            collate_fn=collate_fn,
            num_workers=0
        )

        return loader

    def run(self):
        """Run continual learning experiment."""
        print("\n" + "=" * 70)
        print("INITIALIZATION")
        print("=" * 70)

        # Create model with unified dimension
        first_dataset = self.training_order[0]
        config = SystemConfig(
            device=self.device,
            input_dim=self.unified_dim,  # Use unified dim
            output_dim=self.DATASET_CONFIGS[first_dataset]['output_dim'],
            hidden_dim=512,
            use_cnn=True,
            image_channels=3,  # Max channels
            saturation_threshold=0.8,
            min_cycles_before_expand=5,
            expansion_cooldown=10
        )

        model = ACOCModel(config)
        print(f"✓ Model created: {model.get_total_params():,} parameters")

        # Create modality projector
        projector = ModalityProjector(
            unified_dim=self.unified_dim,
            device=model.device
        )
        print(f"✓ Modality projector created (unified_dim={self.unified_dim})")

        # Create continual trainer
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            learning_rate=0.001,
            replay_buffer_size=self.replay_buffer_size,
            replay_batch_ratio=0.3,  # 30% of batch from replay
            replay_frequency=1,  # Use replay every step
            projector=projector
        )
        print(f"✓ Continual trainer created")

        # Track results
        results = {
            'training_order': self.training_order,
            'accuracies_after_each': {},
            'final_accuracies': {},
            'forgetting': {}
        }

        # Store dataloaders for final evaluation
        all_test_loaders = {}

        print("\n" + "=" * 70)
        print("PHASE 1: Sequential Training with Replay")
        print("=" * 70)

        # Train on each dataset
        for idx, dataset_name in enumerate(self.training_order, 1):
            ds_config = self.DATASET_CONFIGS[dataset_name]

            print(f"\n{'=' * 70}")
            print(f"Task {idx}/{len(self.training_order)}: {ds_config['name']}")
            print(f"{'=' * 70}")

            # Start new task
            trainer.start_task(
                task_id=dataset_name,
                modality=dataset_name,  # Use dataset name as modality ID
                input_dim=ds_config['input_dim'],
                output_dim=ds_config['output_dim'],
                modality_type=ds_config['modality_type'],
                metadata={'image_size': ds_config['image_size']}
            )

            # Load data
            train_loader = self._load_dataset(dataset_name, train=True)
            test_loader = self._load_dataset(dataset_name, train=False)
            all_test_loaders[dataset_name] = test_loader

            print(f"  - Train samples: {len(train_loader.dataset)}")  # type: ignore[arg-type]
            print(f"  - Test samples: {len(test_loader.dataset)}")  # type: ignore[arg-type]

            # Populate replay buffer BEFORE training
            print(f"\n  Populating replay buffer...")
            trainer.populate_replay_buffer(
                data_loader=train_loader,
                num_samples=self.replay_samples_per_task
            )

            # Train with replay
            print(f"\n  Training with experience replay...")
            trainer.run(
                num_cycles=self.num_cycles_per_dataset,
                data_loader=train_loader,
                validation_data=test_loader,
                num_steps_per_cycle=100,
                verbose=True
            )

            # Evaluate on current task
            accuracy = self._evaluate_task(trainer, test_loader, dataset_name)
            results['accuracies_after_each'][dataset_name] = accuracy
            print(f"\n  ✓ Accuracy on {ds_config['name']}: {accuracy:.2f}%")

            # End task
            trainer.end_task()

            # Show intermediate performance on all previous tasks
            if idx > 1:
                print(f"\n  Intermediate evaluation on previous tasks:")
                for prev_task in self.training_order[:idx-1]:
                    prev_acc = self._evaluate_task(
                        trainer,
                        all_test_loaders[prev_task],
                        prev_task
                    )
                    forgetting = trainer.compute_forgetting(prev_task, prev_acc)
                    print(f"    {prev_task:15s}: {prev_acc:5.2f}% (forgetting: {forgetting:+5.2f}%)")

        # Phase 2: Final evaluation
        print(f"\n{'=' * 70}")
        print("PHASE 2: Final Evaluation on All Tasks")
        print(f"{'=' * 70}")

        final_results = trainer.evaluate_all_tasks(all_test_loaders)

        for dataset_name, accuracy in final_results.items():
            results['final_accuracies'][dataset_name] = accuracy
            forgetting = trainer.compute_forgetting(dataset_name, accuracy)
            results['forgetting'][dataset_name] = forgetting

        # Summary
        self._print_summary(results, trainer)

        # Save model
        save_path = f"acoc_continual_{'_'.join(self.datasets)}.pth"
        torch.save({
            'model_state': model.state_dict(),
            'projector_state': projector.state_dict(),
            'config': config,
            'results': results
        }, save_path)
        print(f"\n💾 Model saved: {save_path}")

        return results

    def _evaluate_task(
        self,
        trainer: ContinualACOCTrainer,
        test_loader: DataLoader,
        modality: str
    ) -> float:
        """Evaluate on a specific task."""
        model = trainer.model
        projector = trainer.projector

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(model.device)
                batch_y = batch_y.to(model.device)

                # Project
                batch_x = projector(batch_x, modality)

                # Forward
                outputs, _ = model(batch_x)

                # Predictions
                if batch_y.dim() == 2:
                    _, targets_idx = torch.max(batch_y, 1)
                else:
                    targets_idx = batch_y

                _, predicted = torch.max(outputs, 1)

                total += targets_idx.size(0)
                correct += (predicted == targets_idx).sum().item()

        model.train()
        accuracy = 100 * correct / total if total > 0 else 0.0
        return accuracy

    def _print_summary(self, results, trainer):
        """Print comprehensive summary."""
        print(f"\n{'=' * 70}")
        print("SUMMARY: Continual Learning Performance")
        print(f"{'=' * 70}")

        print(f"\nTraining order: {' → '.join([self.DATASET_CONFIGS[ds]['name'] for ds in self.training_order])}")
        print(f"\nFinal model: {trainer.model.get_total_params():,} parameters, {len(trainer.model.task_blocks)} blocks")

        print("\n  Dataset              | After Training | Final  | Forgetting")
        print("  " + "-" * 66)

        total_forgetting = 0.0
        for dataset_name in self.training_order:
            after = results['accuracies_after_each'][dataset_name]
            final = results['final_accuracies'][dataset_name]
            forgetting = results['forgetting'][dataset_name]
            total_forgetting += forgetting

            ds_display = self.DATASET_CONFIGS[dataset_name]['name']
            print(f"  {ds_display:20s} | {after:13.2f}% | {final:5.2f}% | {forgetting:+9.2f}%")

        avg_forgetting = total_forgetting / len(self.training_order)
        avg_final = sum(results['final_accuracies'].values()) / len(results['final_accuracies'])

        print(f"\n  Average final accuracy: {avg_final:.2f}%")
        print(f"  Average forgetting: {avg_forgetting:+.2f}%")

        # Replay buffer stats
        print(f"\n{trainer.replay_buffer.summary()}")

        # Projector stats
        print(f"\n{trainer.projector.summary()}")

        print(f"\n{'=' * 70}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='ACOC Continual Learning')
    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=['mnist', 'fashion', 'cifar10'],
        default=['mnist', 'fashion', 'cifar10'],
        help='Datasets to use'
    )
    parser.add_argument('--samples', type=int, default=5000, help='Samples per dataset')
    parser.add_argument('--cycles', type=int, default=10, help='Cycles per dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu', help='Device')
    parser.add_argument('--no-shuffle', action='store_true', help='Do not shuffle order')
    parser.add_argument('--replay-buffer-size', type=int, default=2000, help='Replay buffer size')
    parser.add_argument('--replay-per-task', type=int, default=500, help='Replay samples per task')

    args = parser.parse_args()

    trainer = ContinualLearningTrainer(
        datasets=args.datasets,
        samples_per_dataset=args.samples,
        num_cycles_per_dataset=args.cycles,
        batch_size=args.batch_size,
        device=args.device,
        shuffle_order=not args.no_shuffle,
        replay_buffer_size=args.replay_buffer_size,
        replay_samples_per_task=args.replay_per_task
    )

    trainer.run()


if __name__ == '__main__':
    main()
