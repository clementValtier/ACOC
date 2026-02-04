"""
Tests for the main ACOC model.
"""

import torch
import pytest
from acoc import ACOCModel, SystemConfig
from acoc.config import TaskType, ExpansionDecision


class TestACOCModel:
    """Tests for ACOCModel."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(
            device='cpu',
            input_dim=256,
            hidden_dim=128,
            output_dim=10,
            use_cnn=False,
            saturation_threshold=0.7,
            min_cycles_before_expand=2,
            expansion_cooldown=3
        )

    @pytest.fixture
    def model(self, config):
        """Create ACOC model for tests."""
        return ACOCModel(config)

    def test_initialization(self, model, config):
        """Tests model initialization."""
        assert model.config == config
        assert model.current_cycle == 0
        assert len(model.task_blocks) == 3  # TEXT, IMAGE, AUDIO

        # Check that base blocks exist
        assert "base_text" in model.task_blocks
        assert "base_image" in model.task_blocks
        assert "base_audio" in model.task_blocks

    def test_forward(self, model):
        """Tests forward pass."""
        x = torch.randn(32, 256)
        outputs, routing_stats = model(x)

        assert outputs.shape == (32, 10)
        assert not torch.isnan(outputs).any()
        assert isinstance(routing_stats, dict)
        assert sum(routing_stats.values()) == 32  # All samples routed

    def test_forward_multiple_batches(self, model):
        """Tests forward with multiple batches."""
        for _ in range(5):
            x = torch.randn(16, 256)
            outputs, routing_stats = model(x)
            assert outputs.shape == (16, 10)

    def test_compute_loss(self, model):
        """Tests loss computation."""
        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        loss = model.compute_loss(predictions, targets)
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_compute_loss_with_penalties(self, model):
        """Tests loss computation with penalties."""
        predictions = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))

        loss_with_penalty = model.compute_loss(predictions, targets, include_penalties=True)
        loss_without_penalty = model.compute_loss(predictions, targets, include_penalties=False)

        assert loss_with_penalty >= loss_without_penalty

    def test_collect_metrics(self, model):
        """Tests metrics collection."""
        # Forward pass to populate data
        x = torch.randn(32, 256)
        model(x)

        metrics = model.collect_metrics()
        assert metrics is not None
        assert len(metrics.detailed_saturation) > 0

    def test_reset_usage_counts(self, model):
        """Tests resetting usage counts."""
        # Forward to set usage
        x = torch.randn(32, 256)
        model(x)

        assert any(block.usage_count > 0 for block in model.task_blocks.values())

        model.reset_usage_counts()
        assert all(block.usage_count == 0 for block in model.task_blocks.values())

    def test_set_exploration(self, model):
        """Tests setting exploration mode."""
        model.set_exploration("base_text", prob=0.5)
        assert model._force_exploration_block == "base_text"
        assert model._exploration_prob == 0.5

        model.set_exploration(None, prob=0.0)
        assert model._force_exploration_block is None
        assert model._exploration_prob == 0.0

    def test_evaluate_expansion(self, model):
        """Tests expansion evaluation."""
        # Run some forward passes to collect metrics
        for _ in range(5):
            x = torch.randn(32, 256)
            outputs, _ = model(x)
            loss = model.compute_loss(outputs, torch.randint(0, 10, (32,)))
            loss.backward()
            model.metrics.add_loss(loss.item())

        model.current_cycle = 5
        decision = model.evaluate_expansion()

        assert isinstance(decision, ExpansionDecision)
        assert isinstance(decision.should_expand, bool)
        assert decision.confidence >= 0.0

    def test_get_total_params(self, model):
        """Tests total parameter counting."""
        total_params = model.get_total_params()
        assert total_params > 0

        # Verify it matches manual count
        manual_count = sum(p.numel() for p in model.parameters())
        assert total_params == manual_count

    def test_summary(self, model):
        """Tests model summary generation."""
        summary = model.summary()
        assert isinstance(summary, str)
        assert "ACOC Model Summary" in summary
        assert "base_text" in summary or "base_image" in summary

    def test_router_bias_initialization(self):
        """Tests that router bias is initialized on first forward."""
        config = SystemConfig(
            device='cpu',
            input_dim=784,  # MNIST-like
            hidden_dim=128,
            output_dim=10,
            use_cnn=True
        )
        model = ACOCModel(config)

        assert not model._router_bias_initialized

        # Forward with image-like data
        x = torch.randn(16, 784)
        model(x)

        assert model._router_bias_initialized

    def test_maintenance(self, model):
        """Tests running maintenance."""
        model.current_cycle = 25  # Ensure enough cycles have passed

        actions = model.run_maintenance()
        assert isinstance(actions, dict)
        assert "pruned" in actions
        assert "consolidated" in actions


class TestACOCModelWithCNN:
    """Tests for ACOC model with CNN enabled."""

    @pytest.fixture
    def config(self):
        """Test configuration with CNN."""
        return SystemConfig(
            device='cpu',
            input_dim=3072,  # 32x32x3
            hidden_dim=128,
            output_dim=10,
            use_cnn=True,
            cnn_channels=[32, 64],
            image_channels=3
        )

    @pytest.fixture
    def model(self, config):
        """Create ACOC model with CNN."""
        return ACOCModel(config)

    def test_forward_with_cnn(self, model):
        """Tests forward with CNN-based model."""
        x = torch.randn(32, 3072)  # Flattened images
        outputs, routing_stats = model(x)

        assert outputs.shape == (32, 10)
        assert not torch.isnan(outputs).any()

    def test_cnn_expert_used(self, model):
        """Tests that CNN expert is used for image block."""
        from acoc.experts import CNNExpert

        image_block = model.task_blocks["base_image"]
        assert len(image_block.layers) > 0
        assert isinstance(image_block.layers[0], CNNExpert)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
