"""
Tests for the variants and voting system.
"""

import torch
import torch.nn as nn
import pytest
from acoc.config import SystemConfig, ModelMetrics
from acoc.variants import VariantSystem


class SimpleModel(nn.Module):
    """Simple model for tests."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestVariantSystem:
    """Tests for VariantSystem."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(
            num_variants=5,
            delta_magnitude=0.01,
            performance_threshold_ratio=0.95
        )

    @pytest.fixture
    def variant_system(self, config):
        """Variant system for tests."""
        return VariantSystem(config, device=torch.device('cpu'))

    @pytest.fixture
    def model(self):
        """Simple model for tests."""
        return SimpleModel()

    def test_initialization(self, config):
        """Tests system initialization."""
        vs = VariantSystem(config, device=torch.device('cpu'))
        assert vs.config == config
        assert len(vs.deltas) == 0
        assert len(vs.score_history) == 0
        assert len(vs.vote_history) == 0

    def test_initialize_deltas(self, variant_system, model):
        """Tests delta initialization."""
        variant_system.initialize_deltas(model)

        assert len(variant_system.deltas) == 5
        for delta in variant_system.deltas:
            assert isinstance(delta, dict)
            assert len(delta) > 0  # At least a few parameters

    def test_deltas_are_small(self, variant_system, model):
        """Tests that deltas are small relative to the model."""
        variant_system.initialize_deltas(model)

        for delta in variant_system.deltas:
            for name, param_delta in delta.items():
                param = dict(model.named_parameters())[name]
                # Delta should be small compared to parameter
                ratio = param_delta.abs().mean() / (param.abs().mean() + 1e-8)
                assert ratio < 0.1  # Less than 10% on average

    def test_apply_delta(self, variant_system, model):
        """Tests applying a delta."""
        variant_system.initialize_deltas(model)

        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        variant_state = variant_system.apply_delta(model, delta_idx=0)

        # Original state should not be modified
        for name, param in model.state_dict().items():
            assert torch.equal(param, original_state[name])

        # Variant state should be different
        for name in variant_state:
            if name in variant_system.deltas[0]:
                assert not torch.equal(variant_state[name], original_state[name])

    def test_relative_threshold(self, variant_system):
        """Tests relative threshold calculation."""
        # No history
        threshold = variant_system._get_relative_threshold()
        assert threshold == 0.3  # Default threshold

        # With history
        variant_system.score_history.extend([0.8, 0.85, 0.9, 0.88, 0.92])
        threshold = variant_system._get_relative_threshold()
        expected = 0.95 * (sum([0.9, 0.88, 0.92, 0.85, 0.8]) / 5)
        assert abs(threshold - expected) < 0.01

    def test_evaluate_variants(self, variant_system, model):
        """Tests variant evaluation."""
        variant_system.initialize_deltas(model)

        def dummy_evaluate(m):
            """Dummy evaluation function."""
            return torch.randn(1).item()

        scored_variants = variant_system.evaluate_variants(model, dummy_evaluate)

        assert len(scored_variants) == 5
        # Check that it's sorted by descending score
        for i in range(len(scored_variants) - 1):
            assert scored_variants[i][1] >= scored_variants[i + 1][1]

    def test_vote_on_expansion(self, variant_system, model):
        """Tests voting on expansion."""
        variant_system.initialize_deltas(model)
        metrics = ModelMetrics()

        def dummy_evaluate(m):
            """Returns a low score to trigger expansion."""
            return 0.2

        should_expand, confidence, reason = variant_system.vote_on_expansion(
            model, dummy_evaluate, metrics
        )

        assert isinstance(should_expand, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reason, str)
        assert len(variant_system.score_history) > 0

    def test_merge_best_deltas(self, variant_system, model):
        """Tests merging best deltas."""
        variant_system.initialize_deltas(model)

        original_params = {name: param.clone() for name, param in model.named_parameters()}

        def dummy_evaluate(m):
            return torch.randn(1).item()

        variant_system.merge_best_deltas(model, dummy_evaluate, top_k=3)

        # Parameters should have changed
        changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, original_params[name]):
                changed = True
                break
        assert changed

    def test_vote_history_tracking(self, variant_system, model):
        """Tests that vote history is tracked."""
        variant_system.initialize_deltas(model)
        metrics = ModelMetrics()

        def dummy_evaluate(m):
            return 0.5

        variant_system.vote_on_expansion(model, dummy_evaluate, metrics)
        variant_system.vote_on_expansion(model, dummy_evaluate, metrics)

        assert len(variant_system.vote_history) == 2
        summary = variant_system.get_vote_summary()
        assert summary["total"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
