"""
Tests for monitoring modules (gradient flow and activations).
"""

import torch
import pytest
from acoc.monitoring import GradientFlowMonitor, ActivationMonitor


class TestGradientFlowMonitor:
    """Tests for GradientFlowMonitor."""

    def test_initialization(self):
        """Tests monitor initialization."""
        monitor = GradientFlowMonitor(threshold=1e-6, history_size=100)
        assert monitor.threshold == 1e-6
        assert monitor.history_size == 100
        assert len(monitor.gradient_history) == 0

    def test_register_layer(self):
        """Tests layer registration."""
        monitor = GradientFlowMonitor()
        monitor.register_layer("layer1")
        assert "layer1" in monitor.gradient_history

    def test_record_gradients(self):
        """Tests gradient recording."""
        monitor = GradientFlowMonitor(threshold=1e-3)
        gradients = torch.randn(10, 10)

        monitor.record_gradients("layer1", gradients)
        assert "layer1" in monitor.gradient_history
        assert len(monitor.gradient_history["layer1"]) == 1

        stats = monitor.gradient_history["layer1"][0]
        assert "mean" in stats
        assert "max" in stats
        assert "alive_ratio" in stats

    def test_get_flow_ratio(self):
        """Tests gradient flow ratio calculation."""
        monitor = GradientFlowMonitor(threshold=0.1)

        # Strong gradients
        strong_gradients = torch.ones(10, 10)
        monitor.record_gradients("strong", strong_gradients)
        strong_ratio = monitor.get_flow_ratio("strong")
        assert strong_ratio == 1.0  # All gradients > threshold

        # Weak gradients
        weak_gradients = torch.ones(10, 10) * 0.01
        monitor.record_gradients("weak", weak_gradients)
        weak_ratio = monitor.get_flow_ratio("weak")
        assert weak_ratio < 0.5  # Most gradients < threshold


class TestActivationMonitor:
    """Tests for ActivationMonitor."""

    def test_initialization(self):
        """Tests monitor initialization."""
        monitor = ActivationMonitor(
            saturation_threshold=0.95,
            dead_threshold=1e-6,
            history_size=100
        )
        assert monitor.saturation_threshold == 0.95
        assert monitor.dead_threshold == 1e-6
        assert monitor.history_size == 100

    def test_register_layer(self):
        """Tests layer registration."""
        monitor = ActivationMonitor()
        monitor.register_layer("layer1")
        assert "layer1" in monitor.activation_history

    def test_record_activations(self):
        """Tests activation recording."""
        monitor = ActivationMonitor()
        activations = torch.randn(32, 64)  # batch_size=32, neurons=64

        monitor.record_activations("layer1", activations)
        assert "layer1" in monitor.activation_history
        assert len(monitor.activation_history["layer1"]) == 1

        stats = monitor.activation_history["layer1"][0]
        assert "saturated_ratio" in stats
        assert "dead_ratio" in stats
        assert "variance" in stats

    def test_get_saturation_metrics(self):
        """Tests saturation metrics calculation."""
        monitor = ActivationMonitor(dead_threshold=0.01)

        # Normal activations
        normal_act = torch.randn(32, 64).abs()  # All positive
        monitor.record_activations("normal", normal_act)
        sat, dead, var = monitor.get_saturation_metrics("normal")
        assert 0.0 <= sat <= 1.0
        assert 0.0 <= dead <= 1.0
        assert var >= 0.0

        # Dead activations (all zeros)
        dead_act = torch.zeros(32, 64)
        monitor.record_activations("dead", dead_act)
        sat, dead, var = monitor.get_saturation_metrics("dead")
        assert dead > 0.5  # Most neurons are dead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
