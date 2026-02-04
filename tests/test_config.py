"""
Tests for configuration and data structures.
"""

import pytest
from acoc.config import (
    SystemConfig, TaskType, TaskBlock, ModelMetrics,
    SaturationMetrics, ExpansionDecision, TrainingLog
)


class TestSystemConfig:
    """Tests for SystemConfig."""

    def test_default_initialization(self):
        """Tests default config initialization."""
        config = SystemConfig()

        # Check that defaults are set (exact values may vary)
        assert config.input_dim > 0
        assert config.hidden_dim > 0
        assert config.output_dim > 0
        assert config.device in ['cpu', 'cuda', 'mps']

    def test_custom_initialization(self):
        """Tests custom config initialization."""
        config = SystemConfig(
            input_dim=256,
            hidden_dim=128,
            output_dim=5,
            device='cuda',
            saturation_threshold=0.75
        )

        assert config.input_dim == 256
        assert config.hidden_dim == 128
        assert config.output_dim == 5
        assert config.saturation_threshold == 0.75

    def test_cnn_config(self):
        """Tests CNN-specific configuration."""
        config = SystemConfig(
            use_cnn=True,
            cnn_channels=[32, 64, 128],
            image_channels=3
        )

        assert config.use_cnn is True
        assert config.cnn_channels == [32, 64, 128]
        assert config.image_channels == 3


class TestTaskType:
    """Tests for TaskType enum."""

    def test_task_types(self):
        """Tests task type values."""
        assert TaskType.TEXT.value == "text"
        assert TaskType.IMAGE.value == "image"
        assert TaskType.AUDIO.value == "audio"

    def test_task_type_comparison(self):
        """Tests task type comparisons."""
        assert TaskType.TEXT == TaskType.TEXT
        assert TaskType.TEXT != TaskType.IMAGE


class TestSaturationMetrics:
    """Tests for SaturationMetrics."""

    def test_initialization(self):
        """Tests metrics initialization."""
        metrics = SaturationMetrics()

        # Check default values from dataclass definition
        assert metrics.gradient_flow_ratio == 1.0
        assert metrics.activation_saturation == 0.0
        assert metrics.dead_neuron_ratio == 0.0
        assert metrics.activation_variance == 1.0
        assert metrics.combined_score == 0.0

    def test_custom_initialization(self):
        """Tests custom metrics initialization."""
        metrics = SaturationMetrics(
            gradient_flow_ratio=0.5,
            activation_saturation=0.6,
            dead_neuron_ratio=0.3,
            activation_variance=0.4
        )

        assert metrics.gradient_flow_ratio == 0.5
        assert metrics.activation_saturation == 0.6

    def test_compute_combined_score(self):
        """Tests combined score computation."""
        metrics = SaturationMetrics(
            gradient_flow_ratio=0.5,
            activation_saturation=0.6,
            dead_neuron_ratio=0.3,
            activation_variance=0.4
        )

        metrics.compute_combined_score()
        assert metrics.combined_score > 0.0
        assert 0.0 <= metrics.combined_score <= 1.0


class TestModelMetrics:
    """Tests for ModelMetrics."""

    def test_initialization(self):
        """Tests metrics initialization."""
        metrics = ModelMetrics()

        assert len(metrics.loss_history) == 0
        assert len(metrics.detailed_saturation) == 0
        assert len(metrics.expert_utilization) == 0

    def test_add_loss(self):
        """Tests adding loss values."""
        metrics = ModelMetrics()

        metrics.add_loss(1.5)
        metrics.add_loss(1.2)
        metrics.add_loss(0.9)

        assert len(metrics.loss_history) == 3
        assert metrics.loss_history[-1] == 0.9

    def test_add_validation_score(self):
        """Tests adding validation scores."""
        metrics = ModelMetrics()

        metrics.add_validation_score(0.85)
        metrics.add_validation_score(0.87)

        assert len(metrics.validation_scores) == 2

    def test_get_recent_loss_trend(self):
        """Tests recent loss trend calculation."""
        metrics = ModelMetrics()

        for i in range(15):
            metrics.add_loss(3.0 - i * 0.1)  # Decreasing loss

        trend = metrics.get_recent_loss_trend(window=10)
        assert trend is not None and trend > 0.0  # Positive trend means improvement (loss decreasing)


class TestExpansionDecision:
    """Tests for ExpansionDecision."""

    def test_initialization(self):
        """Tests decision initialization."""
        decision = ExpansionDecision(
            should_expand=True,
            expansion_type="width",
            target_block_id="block_1",
            confidence=0.85,
            reason="High saturation detected"
        )

        assert decision.should_expand is True
        assert decision.expansion_type == "width"
        assert decision.target_block_id == "block_1"
        assert decision.confidence == 0.85
        assert "saturation" in decision.reason.lower()


class TestTaskBlock:
    """Tests for TaskBlock."""

    def test_initialization(self):
        """Tests block initialization."""
        block = TaskBlock(
            id="test_block",
            task_type=TaskType.TEXT,
            num_params=10000,
            layers=[],
            creation_cycle=5
        )

        assert block.id == "test_block"
        assert block.task_type == TaskType.TEXT
        assert block.num_params == 10000
        assert block.creation_cycle == 5
        assert block.usage_count == 0
        assert block.last_used_cycle == 0

    def test_usage_tracking(self):
        """Tests usage tracking."""
        block = TaskBlock(
            id="test_block",
            task_type=TaskType.IMAGE,
            num_params=5000,
            layers=[],
            creation_cycle=0
        )

        block.usage_count = 100
        block.last_used_cycle = 10

        assert block.usage_count == 100
        assert block.last_used_cycle == 10


class TestTrainingLog:
    """Tests for TrainingLog."""

    def test_initialization(self):
        """Tests log initialization."""
        log = TrainingLog(
            cycle=10,
            avg_loss=0.5,
            total_params=100000,
            num_blocks=5,
            expanded=True,
            expansion_type="width",
            expansion_target="block_1",
            warmup_active=True,
            saturation_details={"block_1": 0.75}
        )

        assert log.cycle == 10
        assert log.avg_loss == 0.5
        assert log.total_params == 100000
        assert log.num_blocks == 5
        assert log.expanded is True
        assert log.warmup_active is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
