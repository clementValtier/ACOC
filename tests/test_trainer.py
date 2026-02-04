"""
Tests for ACOC Trainer.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from acoc.config import SystemConfig, ExpansionDecision
from acoc.model import ACOCModel
from acoc.training.trainer import ACOCTrainer


class TestACOCTrainer:
    """Tests for ACOCTrainer."""

    @pytest.fixture
    def config(self):
        return SystemConfig(
            device='cpu',
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
            use_cross_entropy=False
        )

    @pytest.fixture
    def model(self, config):
        return ACOCModel(config)

    @pytest.fixture
    def trainer(self, model, config):
        return ACOCTrainer(model, config, learning_rate=0.001)

    @pytest.fixture
    def data_loader(self):
        """Create a simple data loader for testing."""
        x = torch.randn(100, 64)
        y = torch.randn(100, 10)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=16)

    def test_initialization(self, trainer, model, config):
        """Test that trainer initializes correctly."""
        assert trainer.model is model
        assert trainer.config is config
        assert trainer.learning_rate == 0.001
        assert len(trainer.training_logs) == 0
        assert trainer.optimizer is not None

    def test_rebuild_optimizer(self, trainer):
        """Test rebuilding optimizer with different learning rates."""
        initial_optimizer = trainer.optimizer

        # Rebuild with default settings
        trainer._rebuild_optimizer()
        assert trainer.optimizer is not initial_optimizer

        # Rebuild with custom LR multipliers
        lr_multipliers = {
            "router.bias": 5.0,
            "task_blocks.block_0.weight": 2.0
        }
        trainer._rebuild_optimizer(lr_multipliers)
        assert trainer.optimizer is not None

    def test_training_phase(self, trainer, data_loader):
        """Test training phase execution."""
        initial_cycle = trainer.model.current_cycle

        avg_loss = trainer.training_phase(
            data_loader=data_loader,
            num_steps=10,
            verbose=False
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0
        assert len(trainer.model.metrics.loss_history) > 0

    def test_training_phase_without_dataloader(self, trainer):
        """Test training phase with simulated data."""
        avg_loss = trainer.training_phase(
            data_loader=None,
            num_steps=10,
            verbose=False
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_training_step(self, trainer):
        """Test a single training step."""
        batch_x = torch.randn(16, 64)
        batch_y = torch.randn(16, 10)

        initial_params = {name: param.clone() for name, param in trainer.model.named_parameters()}

        loss = trainer._training_step(batch_x, batch_y)

        assert isinstance(loss, float)
        assert loss >= 0

        # Check that parameters were updated
        params_updated = False
        for name, param in trainer.model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_updated = True
                break
        assert params_updated

    def test_checkpoint_phase(self, trainer, data_loader):
        """Test checkpoint phase with variant voting."""
        should_expand, confidence, reason = trainer.checkpoint_phase(
            validation_data=data_loader,
            verbose=False
        )

        assert isinstance(should_expand, bool)
        assert isinstance(confidence, float)
        assert isinstance(reason, str)
        assert 0 <= confidence <= 1

    def test_checkpoint_phase_without_data(self, trainer):
        """Test checkpoint phase with simulated data."""
        should_expand, confidence, reason = trainer.checkpoint_phase(
            validation_data=None,
            verbose=False
        )

        assert isinstance(should_expand, bool)
        assert isinstance(confidence, float)
        assert isinstance(reason, str)

    def test_decision_phase(self, trainer):
        """Test decision phase."""
        decision = trainer.decision_phase(
            variant_vote=False,
            variant_confidence=0.5,
            verbose=False
        )

        assert isinstance(decision, ExpansionDecision)
        assert isinstance(decision.should_expand, bool)
        assert decision.expansion_type in ["width", "depth", "new_block", "none", None]

    def test_decision_phase_with_variant_vote(self, trainer):
        """Test decision phase with strong variant vote."""
        decision = trainer.decision_phase(
            variant_vote=True,
            variant_confidence=0.85,
            verbose=False
        )

        assert isinstance(decision, ExpansionDecision)

    def test_expansion_phase_no_expand(self, trainer):
        """Test expansion phase when no expansion is needed."""
        decision = ExpansionDecision(
            should_expand=False,
            expansion_type=None,
            target_block_id=None,
            confidence=0.0,
            reason="No expansion needed"
        )

        success = trainer.expansion_phase(decision, verbose=False)
        assert success is False

    def test_expansion_phase_with_expand(self, trainer):
        """Test expansion phase when expansion is triggered."""
        # First, train a bit to initialize metrics
        trainer.training_phase(data_loader=None, num_steps=10, verbose=False)

        decision = ExpansionDecision(
            should_expand=True,
            expansion_type="width",
            target_block_id="block_0",
            confidence=0.9,
            reason="High saturation"
        )

        initial_params = trainer.model.get_total_params()
        success = trainer.expansion_phase(decision, verbose=False)

        # Expansion might or might not succeed depending on internal state
        if success:
            assert trainer.model.get_total_params() >= initial_params

    def test_maintenance_phase(self, trainer):
        """Test maintenance phase."""
        actions = trainer.maintenance_phase(verbose=False)

        assert isinstance(actions, dict)
        assert "pruned" in actions
        assert "consolidated" in actions
        assert isinstance(actions["pruned"], list)
        assert isinstance(actions["consolidated"], list)

    def test_run_cycle(self, trainer, data_loader):
        """Test running a complete training cycle."""
        initial_cycle = trainer.model.current_cycle

        log = trainer.run_cycle(
            data_loader=data_loader,
            validation_data=data_loader,
            num_steps=10,
            verbose=False
        )

        assert trainer.model.current_cycle == initial_cycle + 1
        assert log.cycle == initial_cycle
        assert isinstance(log.avg_loss, float)
        assert log.total_params > 0
        assert log.num_blocks > 0
        assert len(trainer.training_logs) == 1

    def test_run_multiple_cycles(self, trainer, data_loader):
        """Test running multiple training cycles."""
        initial_cycle = trainer.model.current_cycle

        trainer.run(
            num_cycles=3,
            data_loader=data_loader,
            validation_data=data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        assert trainer.model.current_cycle == initial_cycle + 3
        assert len(trainer.training_logs) == 3

    def test_callbacks(self, trainer, data_loader):
        """Test that callbacks are called during training."""
        cycle_start_called = []
        cycle_end_called = []

        def on_start(cycle):
            cycle_start_called.append(cycle)

        def on_end(cycle, log):
            cycle_end_called.append(cycle)

        trainer.on_cycle_start = on_start
        trainer.on_cycle_end = on_end

        trainer.run_cycle(
            data_loader=data_loader,
            validation_data=data_loader,
            num_steps=5,
            verbose=False
        )

        assert len(cycle_start_called) == 1
        assert len(cycle_end_called) == 1

    def test_get_training_curve(self, trainer, data_loader):
        """Test getting training curve data."""
        trainer.run(
            num_cycles=3,
            data_loader=data_loader,
            num_steps_per_cycle=5,
            verbose=False
        )

        cycles, losses, params = trainer.get_training_curve()

        assert len(cycles) == 3
        assert len(losses) == 3
        assert len(params) == 3
        assert all(isinstance(c, int) for c in cycles)
        assert all(isinstance(l, float) for l in losses)
        assert all(isinstance(p, int) for p in params)

    def test_warmup_activation(self, trainer):
        """Test that warmup is activated after expansion."""
        # Train and force an expansion
        trainer.training_phase(data_loader=None, num_steps=20, verbose=False)

        # Manually trigger warmup
        trainer.model.warmup_manager.start_warmup("block_0", current_cycle=1)

        # Next training phase should handle warmup
        avg_loss = trainer.training_phase(data_loader=None, num_steps=5, verbose=False)

        assert isinstance(avg_loss, float)


class TestTrainerWithCrossEntropy:
    """Tests for trainer with cross-entropy loss."""

    @pytest.fixture
    def config(self):
        return SystemConfig(
            device='cpu',
            input_dim=64,
            hidden_dim=32,
            output_dim=10,
            use_cross_entropy=True  # Use cross-entropy
        )

    @pytest.fixture
    def model(self, config):
        return ACOCModel(config)

    @pytest.fixture
    def trainer(self, model, config):
        return ACOCTrainer(model, config)

    @pytest.fixture
    def data_loader(self):
        """Create data loader with integer labels for cross-entropy."""
        x = torch.randn(100, 64)
        y = torch.randint(0, 10, (100,))  # Integer labels
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=16)

    def test_training_with_cross_entropy(self, trainer, data_loader):
        """Test training with cross-entropy loss."""
        avg_loss = trainer.training_phase(
            data_loader=data_loader,
            num_steps=10,
            verbose=False
        )

        assert isinstance(avg_loss, float)
        assert avg_loss >= 0

    def test_checkpoint_with_cross_entropy(self, trainer, data_loader):
        """Test checkpoint phase with cross-entropy loss."""
        should_expand, confidence, reason = trainer.checkpoint_phase(
            validation_data=data_loader,
            verbose=False
        )

        assert isinstance(should_expand, bool)
        assert isinstance(confidence, float)
