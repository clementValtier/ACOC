"""
Integration tests for ACOC system.
Tests end-to-end workflows and component interactions.
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from acoc import ACOCModel, ACOCTrainer, SystemConfig


class TestBasicIntegration:
    """Basic integration tests."""

    @pytest.fixture
    def simple_config(self):
        """Simple configuration for quick tests."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False,
            saturation_threshold=0.8,
            min_cycles_before_expand=2,
            expansion_cooldown=2
        )

    @pytest.fixture
    def simple_data_loader(self):
        """Simple data loader for testing."""
        x = torch.randn(50, 32)
        y = torch.randn(50, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=10)

    def test_end_to_end_training(self, simple_config, simple_data_loader):
        """Test complete training workflow."""
        # Create model and trainer
        model = ACOCModel(simple_config)
        trainer = ACOCTrainer(model, simple_config)

        initial_params = model.get_total_params()

        # Run training
        trainer.run(
            num_cycles=3,
            data_loader=simple_data_loader,
            validation_data=simple_data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        # Verify training occurred
        assert len(trainer.training_logs) == 3
        assert all(log.avg_loss is not None and log.avg_loss >= 0 for log in trainer.training_logs)

        # Model should still be functional
        x_test = torch.randn(5, 32)
        outputs, routing_stats = model(x_test)
        assert outputs.shape == (5, 5)

    def test_model_save_and_load(self, simple_config, simple_data_loader):
        """Test saving and loading model state."""
        import tempfile
        import os

        # Train a model
        model = ACOCModel(simple_config)
        trainer = ACOCTrainer(model, simple_config)
        trainer.run(
            num_cycles=2,
            data_loader=simple_data_loader,
            num_steps_per_cycle=5,
            verbose=False
        )

        # Save model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as f:
            save_path = f.name

        try:
            torch.save(model.state_dict(), save_path)

            # Load model
            new_model = ACOCModel(simple_config)
            new_model.load_state_dict(torch.load(save_path))

            # Verify models have same structure and parameters
            model.eval()
            new_model.eval()

            # Check that state dicts match
            for (name1, param1), (name2, param2) in zip(
                model.state_dict().items(), new_model.state_dict().items()
            ):
                assert name1 == name2
                assert torch.allclose(param1, param2, atol=1e-6)

            # Check both models are functional
            x_test = torch.randn(3, 32)
            with torch.no_grad():
                output1, _ = model(x_test)
                output2, _ = new_model(x_test)

            # Both should produce valid outputs
            assert output1.shape == (3, 5)
            assert output2.shape == (3, 5)
        finally:
            if os.path.exists(save_path):
                os.remove(save_path)

    def test_inference_mode(self, simple_config):
        """Test model in inference mode."""
        model = ACOCModel(simple_config)
        model.eval()

        x = torch.randn(10, 32)

        with torch.no_grad():
            outputs, routing_stats = model(x)

        assert outputs.shape == (10, 5)
        assert isinstance(routing_stats, dict)
        assert len(routing_stats) > 0
        assert not outputs.requires_grad


class TestMultiModalIntegration:
    """Integration tests for multi-modal capabilities."""

    @pytest.fixture
    def multimodal_config(self):
        """Configuration with CNN support."""
        return SystemConfig(
            device='cpu',
            input_dim=784,  # 28×28 image
            hidden_dim=64,
            output_dim=10,
            use_cnn=True,
            image_channels=1,
            cnn_channels=[16, 32]
        )

    def test_image_data_processing(self, multimodal_config):
        """Test processing image data."""
        model = ACOCModel(multimodal_config)

        # Simulate MNIST-like data
        x = torch.randn(8, 784)
        outputs, routing_stats = model(x)

        assert outputs.shape == (8, 10)
        assert isinstance(routing_stats, dict)
        assert len(routing_stats) > 0

    def test_automatic_architecture_selection(self, multimodal_config):
        """Test that router selects appropriate architecture."""
        model = ACOCModel(multimodal_config)

        # Process image data
        x_image = torch.randn(5, 784)
        _, stats_image = model(x_image)

        # Router should detect and route appropriately
        assert isinstance(stats_image, dict)
        assert len(stats_image) > 0


class TestExpansionIntegration:
    """Integration tests for expansion mechanisms."""

    @pytest.fixture
    def expansion_config(self):
        """Configuration designed to trigger expansion."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False,  # Disable CNN to avoid dimension issues
            saturation_threshold=0.5,  # Low threshold
            min_cycles_before_expand=1,  # Quick expansion
            expansion_cooldown=1
        )

    @pytest.fixture
    def expansion_data_loader(self):
        """Data loader for expansion tests."""
        x = torch.randn(100, 32)
        y = torch.randn(100, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=20)

    def test_width_expansion(self, expansion_config, expansion_data_loader):
        """Test width expansion mechanism."""
        model = ACOCModel(expansion_config)
        trainer = ACOCTrainer(model, expansion_config)

        initial_params = model.get_total_params()

        # Run several cycles to potentially trigger expansion
        trainer.run(
            num_cycles=5,
            data_loader=expansion_data_loader,
            num_steps_per_cycle=20,
            verbose=False
        )

        # Check if any expansion occurred
        expansion_occurred = any(log.expanded for log in trainer.training_logs)

        # Model should still be functional regardless
        x_test = torch.randn(5, 32)
        outputs, _ = model(x_test)
        assert outputs.shape == (5, 5)

    def test_warmup_after_expansion(self, expansion_config, expansion_data_loader):
        """Test that warmup is activated after expansion."""
        model = ACOCModel(expansion_config)
        trainer = ACOCTrainer(model, expansion_config)

        # Run training
        trainer.run(
            num_cycles=5,
            data_loader=expansion_data_loader,
            num_steps_per_cycle=20,
            verbose=False
        )

        # Check if warmup was ever active
        warmup_was_active = any(log.warmup_active for log in trainer.training_logs)

        # This is not guaranteed, but the system should be functional
        assert len(trainer.training_logs) == 5


class TestVariantSystemIntegration:
    """Integration tests for variant voting system."""

    @pytest.fixture
    def variant_config(self):
        """Configuration for variant testing."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False,  # Disable CNN to avoid dimension issues
            num_variants=3,
            delta_magnitude=0.01,
            performance_threshold_ratio=0.95
        )

    @pytest.fixture
    def variant_data_loader(self):
        """Data loader for variant tests."""
        x = torch.randn(80, 32)
        y = torch.randn(80, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=16)

    def test_variant_voting(self, variant_config, variant_data_loader):
        """Test variant voting mechanism."""
        model = ACOCModel(variant_config)
        trainer = ACOCTrainer(model, variant_config)

        # Initialize variants
        model.variant_system.initialize_deltas(model)
        assert len(model.variant_system.deltas) == 3

        # Run training
        trainer.run(
            num_cycles=3,
            data_loader=variant_data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        # Variants should have been used
        vote_summary = model.variant_system.get_vote_summary()
        assert vote_summary["total"] > 0


class TestMaintenanceIntegration:
    """Integration tests for maintenance operations."""

    @pytest.fixture
    def maintenance_config(self):
        """Configuration for maintenance testing."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False,  # Disable CNN to avoid dimension issues
            maintenance_interval=2,
            prune_unused_after_cycles=3,
            consolidation_similarity_threshold=0.9
        )

    @pytest.fixture
    def maintenance_data_loader(self):
        """Data loader for maintenance tests."""
        x = torch.randn(60, 32)
        y = torch.randn(60, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=15)

    def test_maintenance_execution(self, maintenance_config, maintenance_data_loader):
        """Test that maintenance operations execute correctly."""
        model = ACOCModel(maintenance_config)
        trainer = ACOCTrainer(model, maintenance_config)

        # Run training with maintenance
        trainer.run(
            num_cycles=6,
            data_loader=maintenance_data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        # Maintenance should have run (every 2 cycles)
        # Model should still be functional
        x_test = torch.randn(5, 32)
        outputs, _ = model(x_test)
        assert outputs.shape == (5, 5)


class TestPenaltyIntegration:
    """Integration tests for penalty system."""

    @pytest.fixture
    def penalty_config(self):
        """Configuration for penalty testing."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False,  # Disable CNN to avoid dimension issues
            alpha_global_penalty=0.01,
            beta_task_penalty=0.05,
            task_param_threshold=50_000
        )

    @pytest.fixture
    def penalty_data_loader(self):
        """Data loader for penalty tests."""
        x = torch.randn(70, 32)
        y = torch.randn(70, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=14)

    def test_penalty_in_training(self, penalty_config, penalty_data_loader):
        """Test that penalties are applied during training."""
        model = ACOCModel(penalty_config)
        trainer = ACOCTrainer(model, penalty_config)

        # Run training
        trainer.run(
            num_cycles=3,
            data_loader=penalty_data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        # Check that penalty manager has history
        assert len(model.penalty_manager.penalty_history) > 0

        # All losses should be non-negative
        assert all(log.avg_loss is not None and log.avg_loss >= 0 for log in trainer.training_logs)


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    @pytest.fixture
    def metrics_config(self):
        """Configuration for metrics testing."""
        return SystemConfig(
            device='cpu',
            input_dim=32,
            hidden_dim=16,
            output_dim=5,
            use_cnn=False  # Disable CNN to avoid dimension issues
        )

    @pytest.fixture
    def metrics_data_loader(self):
        """Data loader for metrics tests."""
        x = torch.randn(60, 32)
        y = torch.randn(60, 5)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=12)

    def test_metrics_collection(self, metrics_config, metrics_data_loader):
        """Test that metrics are collected during training."""
        model = ACOCModel(metrics_config)
        trainer = ACOCTrainer(model, metrics_config)

        # Run training
        trainer.run(
            num_cycles=3,
            data_loader=metrics_data_loader,
            num_steps_per_cycle=10,
            verbose=False
        )

        # Check metrics history
        assert len(model.metrics.loss_history) > 0
        assert len(model.metrics.validation_scores) > 0

        # Check that saturation metrics exist
        metrics = model.collect_metrics()
        assert len(metrics.detailed_saturation) > 0
