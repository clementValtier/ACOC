"""
Tests pour les modules de monitoring (gradient flow et activations).
"""

import torch
import pytest
from acoc.monitoring import GradientFlowMonitor, ActivationMonitor


class TestGradientFlowMonitor:
    """Tests pour GradientFlowMonitor."""

    def test_initialization(self):
        """Test l'initialisation du monitor."""
        monitor = GradientFlowMonitor(threshold=1e-6, history_size=100)
        assert monitor.threshold == 1e-6
        assert monitor.history_size == 100
        assert len(monitor.gradient_history) == 0

    def test_register_layer(self):
        """Test l'enregistrement d'une couche."""
        monitor = GradientFlowMonitor()
        monitor.register_layer("layer1")
        assert "layer1" in monitor.gradient_history

    def test_record_gradients(self):
        """Test l'enregistrement des gradients."""
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
        """Test le calcul du ratio de gradient flow."""
        monitor = GradientFlowMonitor(threshold=0.1)

        # Gradients forts
        strong_gradients = torch.ones(10, 10)
        monitor.record_gradients("strong", strong_gradients)
        strong_ratio = monitor.get_flow_ratio("strong")
        assert strong_ratio == 1.0  # Tous les gradients > seuil

        # Gradients faibles
        weak_gradients = torch.ones(10, 10) * 0.01
        monitor.record_gradients("weak", weak_gradients)
        weak_ratio = monitor.get_flow_ratio("weak")
        assert weak_ratio < 0.5  # La plupart des gradients < seuil


class TestActivationMonitor:
    """Tests pour ActivationMonitor."""

    def test_initialization(self):
        """Test l'initialisation du monitor."""
        monitor = ActivationMonitor(
            saturation_threshold=0.95,
            dead_threshold=1e-6,
            history_size=100
        )
        assert monitor.saturation_threshold == 0.95
        assert monitor.dead_threshold == 1e-6
        assert monitor.history_size == 100

    def test_register_layer(self):
        """Test l'enregistrement d'une couche."""
        monitor = ActivationMonitor()
        monitor.register_layer("layer1")
        assert "layer1" in monitor.activation_history

    def test_record_activations(self):
        """Test l'enregistrement des activations."""
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
        """Test le calcul des métriques de saturation."""
        monitor = ActivationMonitor(dead_threshold=0.01)

        # Activations normales
        normal_act = torch.randn(32, 64).abs()  # Toutes positives
        monitor.record_activations("normal", normal_act)
        sat, dead, var = monitor.get_saturation_metrics("normal")
        assert 0.0 <= sat <= 1.0
        assert 0.0 <= dead <= 1.0
        assert var >= 0.0

        # Activations mortes (tous zéros)
        dead_act = torch.zeros(32, 64)
        monitor.record_activations("dead", dead_act)
        sat, dead, var = monitor.get_saturation_metrics("dead")
        assert dead > 0.5  # La plupart des neurones sont morts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
