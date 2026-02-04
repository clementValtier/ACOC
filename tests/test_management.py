"""
Tests for ACOC management modules: penalty, pruning, and warmup.
"""

import pytest
import torch
from acoc.config import SystemConfig, TaskBlock, TaskType, ModelMetrics, SaturationMetrics
from acoc.management.penalty import PenaltyManager
from acoc.management.pruning import PruningManager
from acoc.management.warmup import WarmupManager


class TestPenaltyManager:
    """Tests for PenaltyManager."""

    @pytest.fixture
    def config(self):
        return SystemConfig(
            alpha_global_penalty=0.01,
            beta_task_penalty=0.05,
            task_param_threshold=1_000_000
        )

    @pytest.fixture
    def penalty_manager(self, config):
        return PenaltyManager(config)

    def test_initialization(self, penalty_manager):
        """Test that PenaltyManager initializes correctly."""
        assert penalty_manager.baseline_params == 100_000
        assert len(penalty_manager.penalty_history) == 0

    def test_compute_total_penalty(self, penalty_manager):
        """Test penalty computation with multiple blocks."""
        task_blocks = {
            "block_0": TaskBlock(
                id="block_0",
                task_type=TaskType.IMAGE,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=100,
                last_used_cycle=5
            ),
            "block_1": TaskBlock(
                id="block_1",
                task_type=TaskType.TEXT,
                num_params=1_500_000,  # Exceeds threshold
                layers=[],
                creation_cycle=0,
                usage_count=50,
                last_used_cycle=3
            )
        }

        router_params = 100_000
        total_penalty, global_penalty, task_penalties = penalty_manager.compute_total_penalty(
            task_blocks, router_params
        )

        assert total_penalty > 0
        assert global_penalty > 0
        assert "block_0" in task_penalties
        assert "block_1" in task_penalties
        assert task_penalties["block_1"] > task_penalties["block_0"]  # block_1 exceeds threshold
        assert len(penalty_manager.penalty_history) == 1

    def test_adjust_thresholds_stagnant(self, penalty_manager):
        """Test that penalties are relaxed when loss stagnates."""
        metrics = ModelMetrics()

        # Add stagnant losses
        for i in range(25):
            metrics.add_loss(1.0)

        initial_alpha = penalty_manager.config.alpha_global_penalty
        initial_beta = penalty_manager.config.beta_task_penalty

        adjusted = penalty_manager.adjust_thresholds(metrics)

        assert adjusted is True
        assert penalty_manager.config.alpha_global_penalty < initial_alpha
        assert penalty_manager.config.beta_task_penalty < initial_beta

    def test_adjust_thresholds_improving(self, penalty_manager):
        """Test that penalties are tightened when loss improves rapidly."""
        metrics = ModelMetrics()

        # Add rapidly improving losses
        for i in range(25):
            metrics.add_loss(2.0 - i * 0.05)

        initial_alpha = penalty_manager.config.alpha_global_penalty
        initial_beta = penalty_manager.config.beta_task_penalty

        adjusted = penalty_manager.adjust_thresholds(metrics)

        assert adjusted is True
        assert penalty_manager.config.alpha_global_penalty > initial_alpha
        assert penalty_manager.config.beta_task_penalty > initial_beta


class TestPruningManager:
    """Tests for PruningManager."""

    @pytest.fixture
    def config(self):
        return SystemConfig(
            prune_unused_after_cycles=20,
            consolidation_similarity_threshold=0.9
        )

    @pytest.fixture
    def pruning_manager(self, config):
        return PruningManager(config)

    def test_initialization(self, pruning_manager):
        """Test that PruningManager initializes correctly."""
        assert len(pruning_manager.pruning_history) == 0

    def test_identify_unused_blocks(self, pruning_manager):
        """Test identification of unused blocks."""
        task_blocks = {
            "block_0": TaskBlock(
                id="block_0",
                task_type=TaskType.IMAGE,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=2,  # Very low usage (< 35 * 0.1 = 3.5)
                last_used_cycle=0  # Not used recently
            ),
            "block_1": TaskBlock(
                id="block_1",
                task_type=TaskType.TEXT,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=100,  # High usage
                last_used_cycle=30
            )
        }

        current_cycle = 35
        unused = pruning_manager.identify_unused_blocks(task_blocks, current_cycle)

        assert "block_0" in unused
        assert "block_1" not in unused

    def test_find_similar_blocks(self, pruning_manager):
        """Test finding similar blocks."""
        task_blocks = {
            "block_0": TaskBlock(
                id="block_0",
                task_type=TaskType.IMAGE,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=50,
                last_used_cycle=10
            ),
            "block_1": TaskBlock(
                id="block_1",
                task_type=TaskType.IMAGE,
                num_params=480_000,  # Similar size
                layers=[],
                creation_cycle=0,
                usage_count=45,  # Similar usage
                last_used_cycle=12
            ),
            "block_2": TaskBlock(
                id="block_2",
                task_type=TaskType.TEXT,  # Different type
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=50,
                last_used_cycle=10
            )
        }

        similar = pruning_manager.find_similar_blocks(task_blocks)

        # Should find block_0 and block_1 as similar
        assert len(similar) > 0
        found_pair = any(
            (id1 in ["block_0", "block_1"] and id2 in ["block_0", "block_1"])
            for id1, id2, _ in similar
        )
        assert found_pair

    def test_prune_block(self, pruning_manager):
        """Test pruning a single block."""
        task_blocks = {
            "block_0": TaskBlock(
                id="block_0",
                task_type=TaskType.IMAGE,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=10,
                last_used_cycle=5
            )
        }

        success = pruning_manager.prune_block(task_blocks, "block_0", current_cycle=20)

        assert success is True
        assert "block_0" not in task_blocks
        assert len(pruning_manager.pruning_history) == 1

    def test_consolidate_blocks(self, pruning_manager):
        """Test consolidating two blocks."""
        task_blocks = {
            "block_0": TaskBlock(
                id="block_0",
                task_type=TaskType.IMAGE,
                num_params=500_000,
                layers=[],
                creation_cycle=0,
                usage_count=100,
                last_used_cycle=10
            ),
            "block_1": TaskBlock(
                id="block_1",
                task_type=TaskType.IMAGE,
                num_params=480_000,
                layers=[],
                creation_cycle=0,
                usage_count=50,
                last_used_cycle=12
            )
        }

        survivor = pruning_manager.consolidate_blocks(
            task_blocks, "block_0", "block_1", current_cycle=20
        )

        assert survivor is not None
        assert survivor in ["block_0", "block_1"]
        assert len(task_blocks) == 1
        assert survivor in task_blocks
        assert len(pruning_manager.pruning_history) == 1


class TestWarmupManager:
    """Tests for WarmupManager."""

    @pytest.fixture
    def config(self):
        return SystemConfig(
            warmup_steps=50,
            warmup_lr_multiplier=5.0,
            new_block_exploration_prob=0.1,
            max_warmup_cycles=10
        )

    @pytest.fixture
    def warmup_manager(self, config):
        return WarmupManager(config)

    def test_initialization(self, warmup_manager):
        """Test that WarmupManager initializes correctly."""
        assert len(warmup_manager.active_warmups) == 0
        assert not warmup_manager.is_warmup_active()

    def test_start_warmup(self, warmup_manager):
        """Test starting a warmup phase."""
        warmup_manager.start_warmup("block_0", current_cycle=5)

        assert warmup_manager.is_warmup_active()
        assert warmup_manager.is_warmup_active("block_0")
        assert "block_0" in warmup_manager.get_warmup_blocks()

    def test_step_warmup(self, warmup_manager):
        """Test incrementing warmup steps."""
        warmup_manager.start_warmup("block_0", current_cycle=5)

        initial_steps = warmup_manager.active_warmups["block_0"]["steps_done"]
        warmup_manager.step("block_0")

        assert warmup_manager.active_warmups["block_0"]["steps_done"] == initial_steps + 1

    def test_should_continue_warmup(self, warmup_manager):
        """Test warmup continuation logic."""
        warmup_manager.start_warmup("block_0", current_cycle=5)

        # Initially should continue
        assert warmup_manager.should_continue_warmup("block_0", current_cycle=6)

        # After max steps, should stop
        warmup_manager.active_warmups["block_0"]["steps_done"] = 50
        assert not warmup_manager.should_continue_warmup("block_0", current_cycle=7)

    def test_warmup_timeout(self, warmup_manager):
        """Test that warmup times out after max_warmup_cycles."""
        warmup_manager.start_warmup("block_0", current_cycle=5)

        # Should continue within timeout
        assert warmup_manager.should_continue_warmup("block_0", current_cycle=10)

        # Should stop after timeout
        assert not warmup_manager.should_continue_warmup("block_0", current_cycle=20)

    def test_end_warmup(self, warmup_manager):
        """Test ending a warmup phase."""
        warmup_manager.start_warmup("block_0", current_cycle=5)
        assert warmup_manager.is_warmup_active("block_0")

        warmup_manager.end_warmup("block_0")
        assert not warmup_manager.is_warmup_active("block_0")

    def test_check_and_cleanup(self, warmup_manager):
        """Test automatic cleanup of completed warmups."""
        warmup_manager.start_warmup("block_0", current_cycle=5)
        warmup_manager.active_warmups["block_0"]["steps_done"] = 50

        warmup_manager.check_and_cleanup(current_cycle=6)

        assert not warmup_manager.is_warmup_active("block_0")

    def test_get_lr_multiplier(self, warmup_manager):
        """Test learning rate multiplier for new parameters."""
        new_params = {"block_0.weight", "block_0.bias"}
        warmup_manager.start_warmup("block_0", current_cycle=5, new_params=new_params)

        # New parameters should get higher LR
        assert warmup_manager.get_lr_multiplier("block_0.weight") == 5.0

        # Old parameters should get normal LR
        assert warmup_manager.get_lr_multiplier("old_param.weight") == 1.0

    def test_get_exploration_prob(self, warmup_manager):
        """Test exploration probability for warmup blocks."""
        warmup_manager.start_warmup("block_0", current_cycle=5)

        # Warmup block should have exploration probability
        assert warmup_manager.get_exploration_prob("block_0") == 0.1

        # Non-warmup block should have zero exploration
        assert warmup_manager.get_exploration_prob("block_1") == 0.0
