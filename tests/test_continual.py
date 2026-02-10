"""
Tests for continual learning components.
"""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from acoc import (
    ACOCModel,
    SystemConfig,
    ContinualACOCTrainer,
    ModalityProjector,
    ProjectionLayer,
    ReplayBuffer,
    ReplayExample
)


@pytest.fixture
def device():
    """Test device."""
    return torch.device('cpu')


@pytest.fixture
def config(device):
    """Test configuration."""
    return SystemConfig(
        device=device,
        input_dim=128,
        output_dim=10,
        hidden_dim=64,
        use_cnn=False,
        use_cross_entropy=True
    )


@pytest.fixture
def model(config):
    """Test model."""
    return ACOCModel(config)


class TestProjectionLayer:
    """Tests for ProjectionLayer."""

    def test_projection_layer_creation(self, device):
        """Test creating projection layer."""
        proj = ProjectionLayer(input_dim=784, output_dim=128, hidden_dim=256)
        proj.to(device)

        assert proj.down_proj.in_features == 784
        assert proj.down_proj.out_features == 256
        assert proj.up_proj.in_features == 256
        assert proj.up_proj.out_features == 128

    def test_projection_forward(self, device):
        """Test projection forward pass."""
        proj = ProjectionLayer(input_dim=784, output_dim=128)
        proj.to(device)

        x = torch.randn(32, 784).to(device)
        out = proj(x)

        assert out.shape == (32, 128)

    def test_projection_with_default_hidden(self, device):
        """Test projection with default hidden dimension."""
        proj = ProjectionLayer(input_dim=784, output_dim=128)  # hidden_dim=None
        proj.to(device)

        # Default hidden should be (input + output) // 2
        expected_hidden = (784 + 128) // 2
        assert proj.down_proj.out_features == expected_hidden  # 456


class TestModalityProjector:
    """Tests for ModalityProjector."""

    def test_projector_creation(self, device):
        """Test creating modality projector."""
        projector = ModalityProjector(unified_dim=128, device=device)

        assert projector.unified_dim == 128
        assert projector.device == device
        assert len(projector.projections) == 0

    def test_register_modality(self, device):
        """Test registering a modality."""
        projector = ModalityProjector(unified_dim=128, device=device)

        projector.register_modality(
            name='mnist',
            input_dim=784,
            modality_type='image'
        )

        assert 'mnist' in projector.list_modalities()
        info = projector.get_modality_info('mnist')
        assert info.input_dim == 784
        assert info.modality_type == 'image'

    def test_project_modality(self, device):
        """Test projecting data through modality."""
        projector = ModalityProjector(unified_dim=128, device=device)
        projector.register_modality(name='mnist', input_dim=784, modality_type='image')

        x = torch.randn(32, 784).to(device)
        out = projector(x, 'mnist')

        assert out.shape == (32, 128)

    def test_project_unknown_modality(self, device):
        """Test projecting through unregistered modality."""
        projector = ModalityProjector(unified_dim=128, device=device)

        x = torch.randn(32, 784).to(device)

        with pytest.raises(ValueError, match="Unknown modality"):
            projector(x, 'unknown')

    def test_multiple_modalities(self, device):
        """Test registering and using multiple modalities."""
        projector = ModalityProjector(unified_dim=128, device=device)

        # Register two modalities with different dims
        projector.register_modality(name='mnist', input_dim=784, modality_type='image')
        projector.register_modality(name='cifar', input_dim=3072, modality_type='image')

        # Project both
        x_mnist = torch.randn(16, 784).to(device)
        x_cifar = torch.randn(16, 3072).to(device)

        out_mnist = projector(x_mnist, 'mnist')
        out_cifar = projector(x_cifar, 'cifar')

        # Both should map to unified dimension
        assert out_mnist.shape == (16, 128)
        assert out_cifar.shape == (16, 128)


class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_buffer_creation(self, device):
        """Test creating replay buffer."""
        buffer = ReplayBuffer(capacity=100, device=device)

        assert buffer.capacity == 100
        assert len(buffer) == 0

    def test_add_example(self, device):
        """Test adding example to buffer."""
        buffer = ReplayBuffer(capacity=100, device=device)

        data = torch.randn(784).to(device)
        target = torch.tensor([1, 0, 0]).to(device)

        buffer.add(data, target, task_id='task1', modality='mnist')

        assert len(buffer) == 1

    def test_add_batch(self, device):
        """Test adding batch to buffer."""
        buffer = ReplayBuffer(capacity=100, device=device)

        data_batch = torch.randn(32, 784).to(device)
        target_batch = torch.randn(32, 10).to(device)

        buffer.add_batch(data_batch, target_batch, task_id='task1', modality='mnist')

        assert len(buffer) == 32

    def test_sample_from_buffer(self, device):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Add some examples
        for i in range(50):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task1', modality='mnist')

        # Sample
        data, targets, tasks, modalities = buffer.sample(batch_size=16)

        assert data.shape == (16, 784)
        assert targets.shape == (16, 10)
        assert len(tasks) == 16
        assert len(modalities) == 16

    def test_balanced_sampling(self, device):
        """Test balanced sampling across tasks."""
        buffer = ReplayBuffer(capacity=100, sampling_strategy='balanced', device=device)

        # Add examples from two tasks (same dimension for testing)
        for i in range(30):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task1', modality='mnist')

        for i in range(20):
            data = torch.randn(784).to(device)  # Same dimension as task1
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task2', modality='fashion')

        # Sample and check distribution
        data, targets, tasks, modalities = buffer.sample(batch_size=20)

        task1_count = sum(1 for t in tasks if t == 'task1')
        task2_count = sum(1 for t in tasks if t == 'task2')

        # Should be roughly balanced (allow some variance)
        assert task1_count >= 5
        assert task2_count >= 5

        # Check dimensions
        assert data.shape == (20, 784)
        assert targets.shape == (20, 10)

    def test_capacity_limit(self, device):
        """Test buffer respects capacity limit."""
        buffer = ReplayBuffer(capacity=10, device=device)

        # Add more than capacity
        for i in range(50):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task1', modality='mnist')

        # Should not exceed capacity
        assert len(buffer) == 10

    def test_task_distribution(self, device):
        """Test getting task distribution."""
        buffer = ReplayBuffer(capacity=100, device=device)

        # Add from two tasks
        for i in range(30):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task1', modality='mnist')

        for i in range(20):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            buffer.add(data, target, task_id='task2', modality='fashion')

        dist = buffer.get_task_distribution()

        assert 'task1' in dist
        assert 'task2' in dist
        assert dist['task1'] == 0.6  # 30/50
        assert dist['task2'] == 0.4  # 20/50


class TestContinualACOCTrainer:
    """Tests for ContinualACOCTrainer."""

    def test_trainer_creation_full(self, model, config):
        """Test creating trainer with all features enabled."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            enable_replay=True,
            enable_projections=True
        )

        assert trainer.enable_replay is True
        assert trainer.enable_projections is True
        assert trainer.replay_buffer is not None
        assert trainer.projector is not None

    def test_trainer_creation_replay_only(self, model, config):
        """Test creating trainer with replay only."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            enable_replay=True,
            enable_projections=False
        )

        assert trainer.enable_replay is True
        assert trainer.enable_projections is False
        assert trainer.replay_buffer is not None
        assert trainer.projector is None

    def test_trainer_creation_projections_only(self, model, config):
        """Test creating trainer with projections only."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            enable_replay=False,
            enable_projections=True
        )

        assert trainer.enable_replay is False
        assert trainer.enable_projections is True
        assert trainer.replay_buffer is None
        assert trainer.projector is not None

    def test_trainer_creation_disabled(self, model, config):
        """Test creating trainer with both features disabled."""
        trainer = ContinualACOCTrainer(
            model=model,
            config=config,
            enable_replay=False,
            enable_projections=False
        )

        assert trainer.enable_replay is False
        assert trainer.enable_projections is False
        assert trainer.replay_buffer is None
        assert trainer.projector is None

    def test_start_task(self, model, config):
        """Test starting a new task."""
        trainer = ContinualACOCTrainer(model, config)

        trainer.start_task(
            task_id='task1',
            modality='mnist',
            input_dim=784,
            output_dim=10
        )

        assert trainer.current_task == 'task1'
        assert trainer.current_modality == 'mnist'
        assert 'task1' in trainer.tasks_seen

    def test_end_task(self, model, config):
        """Test ending a task."""
        trainer = ContinualACOCTrainer(model, config)

        trainer.start_task('task1', 'mnist', 784, 10)
        trainer.end_task()

        assert trainer.current_task is None
        assert trainer.current_modality is None

    def test_populate_replay_buffer(self, model, config, device):
        """Test populating replay buffer."""
        trainer = ContinualACOCTrainer(
            model,
            config,
            replay_buffer_size=50,
            enable_replay=True
        )

        trainer.start_task('task1', 'mnist', 784, 10)

        # Create dummy data
        x = torch.randn(100, 784).to(device)
        y = torch.randn(100, 10).to(device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        # Populate
        trainer.populate_replay_buffer(loader, num_samples=30)

        assert len(trainer.replay_buffer) == 30

    def test_populate_replay_buffer_disabled(self, model, config, device):
        """Test populating replay buffer when disabled."""
        trainer = ContinualACOCTrainer(
            model,
            config,
            enable_replay=False
        )

        trainer.start_task('task1', 'mnist', 784, 10)

        # Create dummy data
        x = torch.randn(100, 784).to(device)
        y = torch.randn(100, 10).to(device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32)

        # Should not raise error, just skip
        trainer.populate_replay_buffer(loader, num_samples=30)

    def test_training_step_with_replay(self, model, config, device):
        """Test training step with replay enabled."""
        trainer = ContinualACOCTrainer(
            model,
            config,
            enable_replay=True,
            enable_projections=True
        )

        trainer.start_task('task1', 'mnist', 784, 10)

        # Add some examples to buffer
        for i in range(10):
            data = torch.randn(784).to(device)
            target = torch.randn(10).to(device)
            trainer.replay_buffer.add(data, target, 'task1', 'mnist')

        # Training step
        batch_x = torch.randn(16, 784).to(device)
        batch_y = torch.randn(16, 10).to(device)

        loss = trainer._training_step(batch_x, batch_y, use_replay=True)

        assert isinstance(loss, float)
        assert loss > 0

    def test_training_step_without_replay(self, model, config, device):
        """Test training step with replay disabled."""
        trainer = ContinualACOCTrainer(
            model,
            config,
            enable_replay=False,
            enable_projections=False
        )

        trainer.start_task('task1', 'mnist', 784, 10)

        # Training step (no projection, no replay)
        batch_x = torch.randn(16, 128).to(device)  # Already correct dim
        batch_y = torch.randn(16, 10).to(device)

        loss = trainer._training_step(batch_x, batch_y, use_replay=False)

        assert isinstance(loss, float)
        assert loss > 0

    def test_compute_forgetting(self, model, config):
        """Test computing forgetting metric."""
        trainer = ContinualACOCTrainer(model, config)

        # Simulate accuracy over time
        forgetting = trainer.compute_forgetting('task1', 90.0)
        assert forgetting == 0.0  # First time, no forgetting

        forgetting = trainer.compute_forgetting('task1', 85.0)
        assert forgetting == 5.0  # Forgot 5%

        forgetting = trainer.compute_forgetting('task1', 92.0)
        assert forgetting == 0.0  # Improved, so negative becomes 0
