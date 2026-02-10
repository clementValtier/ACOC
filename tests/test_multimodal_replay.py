"""
Tests for multi-modal replay scenarios.

Covers the critical path where replay buffer stores samples from different
modalities with different native dimensions (e.g. MNIST 784 + CIFAR 3072),
and the continual trainer must project each sample back through its
modality-specific projection layer.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from acoc import (
    ACOCModel,
    SystemConfig,
    ContinualACOCTrainer,
    ModalityProjector,
    ReplayBuffer,
)
from acoc.config.structures import ExpansionDecision


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def config(device):
    return SystemConfig(
        device=device,
        input_dim=128,
        output_dim=10,
        hidden_dim=64,
        use_cnn=False,
        use_cross_entropy=True,
    )


@pytest.fixture
def model(config):
    return ACOCModel(config)


# ---------------------------------------------------------------------------
# Replay buffer – mixed dimensions
# ---------------------------------------------------------------------------

class TestReplayBufferMixedDimensions:
    """Replay buffer must handle samples with different native dimensions."""

    def test_sample_mixed_dims_pads_to_max(self, device):
        """Sampling from buffer with heterogeneous dims returns a padded tensor."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # 20 samples at dim 784 (MNIST)
        for _ in range(20):
            buffer.add(torch.randn(784), torch.tensor(0), task_id='mnist', modality='mnist')

        # 20 samples at dim 3072 (CIFAR)
        for _ in range(20):
            buffer.add(torch.randn(3072), torch.tensor(1), task_id='cifar', modality='cifar')

        data, targets, tasks, modalities = buffer.sample(batch_size=20)

        # Padded to largest dim
        assert data.shape[1] == 3072
        assert data.shape[0] == 20

    def test_padded_values_are_zero(self, device):
        """Padding region must be exactly zero so truncation is lossless."""
        buffer = ReplayBuffer(capacity=50, device=device)

        small = torch.ones(100)   # dim 100
        big   = torch.ones(500)   # dim 500

        buffer.add(small, torch.tensor(0), task_id='a', modality='small')
        buffer.add(big,   torch.tensor(1), task_id='b', modality='big')

        data, _, _, modalities = buffer.sample(batch_size=2)

        for i, mod in enumerate(modalities):
            if mod == 'small':
                # tail must be zeros
                assert torch.all(data[i, 100:] == 0)
                # head must be ones
                assert torch.allclose(data[i, :100], torch.ones(100))

    def test_uniform_dims_no_padding(self, device):
        """When all samples share a dimension, plain stack (no padding) is used."""
        buffer = ReplayBuffer(capacity=50, device=device)

        for _ in range(30):
            buffer.add(torch.randn(784), torch.tensor(0), task_id='t', modality='m')

        data, _, _, _ = buffer.sample(batch_size=10)
        assert data.shape == (10, 784)


# ---------------------------------------------------------------------------
# Projection truncation after padding
# ---------------------------------------------------------------------------

class TestProjectionAfterPadding:
    """Projection layers must receive the native dim, not the padded one."""

    def test_truncate_and_project_single_sample(self, device):
        """Manually simulate the truncation + projection path."""
        projector = ModalityProjector(unified_dim=128, device=device)
        projector.register_modality('mnist', input_dim=784, modality_type='image')
        projector.register_modality('cifar', input_dim=3072, modality_type='image')

        # Simulate a padded MNIST sample (784 real + rest zeros)
        padded = torch.zeros(3072)
        padded[:784] = torch.randn(784)

        native_dim = projector.modalities['mnist'].input_dim
        truncated = padded[:native_dim]

        out = projector(truncated.unsqueeze(0), 'mnist')
        assert out.shape == (1, 128)

    def test_truncate_and_project_cifar_sample(self, device):
        """CIFAR sample (already at max dim) needs no truncation."""
        projector = ModalityProjector(unified_dim=128, device=device)
        projector.register_modality('cifar', input_dim=3072, modality_type='image')

        sample = torch.randn(3072)
        native_dim = projector.modalities['cifar'].input_dim
        truncated = sample[:native_dim]

        out = projector(truncated.unsqueeze(0), 'cifar')
        assert out.shape == (1, 128)

    def test_project_mixed_batch_individually(self, device):
        """Project a mixed-modality batch sample-by-sample (as the trainer does)."""
        projector = ModalityProjector(unified_dim=128, device=device)
        projector.register_modality('mnist', input_dim=784, modality_type='image')
        projector.register_modality('cifar', input_dim=3072, modality_type='image')

        buffer = ReplayBuffer(capacity=100, device=device)
        for _ in range(15):
            buffer.add(torch.randn(784), torch.tensor(0), task_id='mnist', modality='mnist')
        for _ in range(15):
            buffer.add(torch.randn(3072), torch.tensor(1), task_id='cifar', modality='cifar')

        data, _, _, modalities = buffer.sample(batch_size=20)

        projected = []
        for i in range(data.size(0)):
            mod = modalities[i]
            native_dim = projector.modalities[mod].input_dim
            x = data[i, :native_dim].unsqueeze(0)
            projected.append(projector(x, mod).squeeze(0))

        stacked = torch.stack(projected)
        assert stacked.shape == (20, 128)


# ---------------------------------------------------------------------------
# End-to-end: ContinualACOCTrainer with multi-modal replay
# ---------------------------------------------------------------------------

class TestContinualTrainerMultiModalReplay:
    """Full integration: train on two modalities, replay must not crash."""

    @pytest.fixture
    def trainer(self, model, config):
        return ContinualACOCTrainer(
            model=model,
            config=config,
            learning_rate=0.001,
            replay_buffer_size=200,
            replay_batch_ratio=0.5,
            replay_frequency=1,
            enable_replay=True,
            enable_projections=True,
        )

    @pytest.fixture
    def mnist_loader(self, device):
        x = torch.randn(80, 784, device=device)
        y = torch.randint(0, 10, (80,), device=device)
        return DataLoader(TensorDataset(x, y), batch_size=16)

    @pytest.fixture
    def cifar_loader(self, device):
        x = torch.randn(80, 3072, device=device)
        y = torch.randint(0, 10, (80,), device=device)
        return DataLoader(TensorDataset(x, y), batch_size=16)

    def test_train_first_task_no_replay(self, trainer, mnist_loader):
        """First task trains without replay (buffer empty)."""
        trainer.start_task('mnist', 'mnist', input_dim=784, output_dim=10)

        for batch_x, batch_y in mnist_loader:
            loss = trainer._training_step(batch_x, batch_y, use_replay=True)
            assert isinstance(loss, float)

    def test_populate_then_train_second_task_with_replay(
        self, trainer, mnist_loader, cifar_loader
    ):
        """Train MNIST, populate buffer, switch to CIFAR, replay must work."""
        # --- Task 1: MNIST ---
        trainer.start_task('mnist', 'mnist', input_dim=784, output_dim=10)

        for batch_x, batch_y in mnist_loader:
            trainer._training_step(batch_x, batch_y, use_replay=False)

        trainer.populate_replay_buffer(mnist_loader, num_samples=50)
        assert len(trainer.replay_buffer) == 50
        trainer.end_task()

        # --- Task 2: CIFAR ---
        trainer.start_task('cifar', 'cifar', input_dim=3072, output_dim=10)

        losses = []
        for batch_x, batch_y in cifar_loader:
            loss = trainer._training_step(batch_x, batch_y, use_replay=True)
            assert isinstance(loss, float)
            losses.append(loss)

        # At least some replay steps should have occurred
        assert trainer.replay_steps > 0
        assert len(losses) > 0

    def test_three_modalities_sequential(self, model, config, device):
        """Three modalities trained sequentially with replay across all."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            replay_buffer_size=300,
            replay_batch_ratio=0.4,
            replay_frequency=1,
            enable_replay=True,
            enable_projections=True,
        )

        dims = {'mnist': 784, 'cifar': 3072, 'audio': 1600}

        for task_name, dim in dims.items():
            x = torch.randn(40, dim, device=device)
            y = torch.randint(0, 10, (40,), device=device)
            loader = DataLoader(TensorDataset(x, y), batch_size=16)

            trainer.start_task(task_name, task_name, input_dim=dim, output_dim=10)

            for batch_x, batch_y in loader:
                loss = trainer._training_step(batch_x, batch_y, use_replay=True)
                assert isinstance(loss, float)

            trainer.populate_replay_buffer(loader, num_samples=40)
            trainer.end_task()

        # Buffer should contain samples from all 3 modalities
        dist = trainer.replay_buffer.get_task_distribution()
        assert len(dist) == 3

    def test_replay_does_not_corrupt_gradients(self, trainer, mnist_loader, cifar_loader, device):
        """Replay loss should produce finite gradients."""
        trainer.start_task('mnist', 'mnist', input_dim=784, output_dim=10)
        for batch_x, batch_y in mnist_loader:
            trainer._training_step(batch_x, batch_y, use_replay=False)
        trainer.populate_replay_buffer(mnist_loader, num_samples=50)
        trainer.end_task()

        trainer.start_task('cifar', 'cifar', input_dim=3072, output_dim=10)

        for batch_x, batch_y in cifar_loader:
            trainer._training_step(batch_x, batch_y, use_replay=True)

        # All model params should be finite
        for name, p in trainer.model.named_parameters():
            assert torch.isfinite(p).all(), f"Non-finite values in {name}"

    def test_expansion_phase_projects_data_for_fisher(self, trainer, mnist_loader):
        """expansion_phase must project raw data before Fisher computation on the router."""
        trainer.start_task('mnist', 'mnist', input_dim=784, output_dim=10)

        # Train a few steps so model has some state
        for batch_x, batch_y in mnist_loader:
            trainer._training_step(batch_x, batch_y, use_replay=False)

        # Force an expansion decision
        decision = ExpansionDecision(
            should_expand=True,
            expansion_type="width",
            target_block_id="block_0",
            confidence=0.9,
            reason="test"
        )

        # This would crash if raw 784-dim data were sent to a 128-dim router
        success = trainer.expansion_phase(decision, verbose=False, data_loader=mnist_loader)
        assert isinstance(success, bool)

    def test_expansion_phase_without_projections(self, model, config, device):
        """expansion_phase works normally when projections are disabled."""
        trainer = ContinualACOCTrainer(
            model=model, config=config,
            enable_replay=False, enable_projections=False,
        )
        x = torch.randn(40, 128, device=device)
        y = torch.randint(0, 10, (40,), device=device)
        loader = DataLoader(TensorDataset(x, y), batch_size=16)

        trainer.start_task('task', 'task', input_dim=128, output_dim=10)
        for bx, by in loader:
            trainer._training_step(bx, by, use_replay=False)

        decision = ExpansionDecision(
            should_expand=True,
            expansion_type="width",
            target_block_id="block_0",
            confidence=0.9,
            reason="test"
        )
        success = trainer.expansion_phase(decision, verbose=False, data_loader=loader)
        assert isinstance(success, bool)

    def test_replay_without_projections_same_dim(self, model, config, device):
        """Replay without projections works if all tasks share the same dim."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            replay_buffer_size=100,
            enable_replay=True,
            enable_projections=False,
        )

        # Both tasks at input_dim=128 (matches config.input_dim)
        for task_name in ['task_a', 'task_b']:
            x = torch.randn(40, 128, device=device)
            y = torch.randint(0, 10, (40,), device=device)
            loader = DataLoader(TensorDataset(x, y), batch_size=16)

            trainer.start_task(task_name, task_name, input_dim=128, output_dim=10)
            for batch_x, batch_y in loader:
                loss = trainer._training_step(batch_x, batch_y, use_replay=True)
                assert isinstance(loss, float)
            trainer.populate_replay_buffer(loader, num_samples=20)
            trainer.end_task()
