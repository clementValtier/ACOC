"""
Tests for model expansion logic.
"""

import torch
import pytest
from acoc.config import SystemConfig, TaskBlock, ModelMetrics, TaskType, SaturationMetrics
from acoc.management import ExpansionManager
from acoc.experts import MLPExpert


class TestExpansionManager:
    """Tests for ExpansionManager."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(
            saturation_threshold=0.6,
            min_cycles_before_expand=2,
            expansion_cooldown=3,
            expansion_ratio=0.1
        )

    @pytest.fixture
    def manager(self, config):
        """Expansion manager for tests."""
        return ExpansionManager(config)

    @pytest.fixture
    def task_blocks(self, config):
        """Task blocks for tests."""
        expert = MLPExpert(input_dim=256, hidden_dim=512, output_dim=256, name="test_expert", config=config)
        block = TaskBlock(
            id="test_block",
            task_type=TaskType.TEXT,
            num_params=expert.get_param_count(),
            layers=[expert],
            creation_cycle=0
        )
        return {"test_block": block}

    def test_initialization(self, config):
        """Tests manager initialization."""
        manager = ExpansionManager(config)
        assert manager.config == config
        assert manager.last_expansion_cycle == -config.expansion_cooldown
        assert len(manager.expansion_history) == 0

    def test_cooldown_prevents_expansion(self, manager, task_blocks):
        """Tests that cooldown prevents expansion."""
        metrics = ModelMetrics()

        # First cycle
        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=0)
        assert not decision.should_expand
        assert "Insufficient history" in decision.reason

    def test_saturation_triggers_expansion(self, manager, task_blocks):
        """Tests that saturation triggers expansion."""
        metrics = ModelMetrics()

        # Create high saturation metrics
        sat_metrics = SaturationMetrics(
            gradient_flow_ratio=0.3,  # Low
            activation_saturation=0.8,  # High
            dead_neuron_ratio=0.5,  # High
            activation_variance=0.1  # Low
        )
        sat_metrics.compute_combined_score()
        metrics.detailed_saturation["test_block"] = sat_metrics

        # Sufficient cycle
        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=5)
        assert decision.should_expand
        assert decision.expansion_type == "width"
        assert decision.target_block_id == "test_block"

    def test_stagnant_loss_triggers_new_block(self, manager, task_blocks):
        """Tests that stagnant loss triggers a new block."""
        metrics = ModelMetrics()

        # Stagnant loss (improvement < 1%)
        for i in range(15):
            metrics.add_loss(2.9 + i * 0.0001)  # Minimal improvement

        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=15)
        assert decision.should_expand
        assert decision.expansion_type == "new_block"

    def test_expand_width(self, manager, task_blocks):
        """Tests width expansion."""
        initial_params = task_blocks["test_block"].num_params

        success = manager._expand_width("test_block", task_blocks)
        assert success

        # Verify that parameter count increased
        new_params = task_blocks["test_block"].num_params
        assert new_params > initial_params

    def test_create_new_block(self, manager, task_blocks):
        """Tests creation of a new block."""
        initial_count = len(task_blocks)

        new_id = manager._create_new_block(
            task_blocks,
            current_cycle=5,
            device=torch.device('cpu')
        )

        assert new_id is not None
        assert len(task_blocks) == initial_count + 1
        assert new_id in task_blocks
        assert task_blocks[new_id].creation_cycle == 5

    def test_expansion_history_tracking(self, manager, task_blocks):
        """Tests that expansion history is properly tracked."""
        manager.execute_expansion(
            decision=type('obj', (object,), {
                'should_expand': True,
                'expansion_type': 'width',
                'target_block_id': 'test_block'
            })(),
            task_blocks=task_blocks,
            current_cycle=10,
            device=torch.device('cpu')
        )

        assert len(manager.expansion_history) == 1
        assert manager.last_expansion_cycle == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
