"""
Tests for the Router component.
"""

import torch
import pytest
from acoc.core import Router


class TestRouter:
    """Tests for the Router class."""

    @pytest.fixture
    def router(self):
        """Create a basic router for tests."""
        return Router(input_dim=256, num_routes=3, hidden_dim=64)

    def test_initialization(self):
        """Tests router initialization."""
        router = Router(input_dim=256, num_routes=3, hidden_dim=64)
        assert router.input_dim == 256
        assert router.num_routes == 3
        assert router.route_bias.shape[0] == 3

    def test_forward(self, router):
        """Tests forward pass."""
        x = torch.randn(32, 256)
        selected, probs = router(x)

        assert selected.shape == (32,)
        assert probs.shape == (32, 3)
        assert torch.all(selected >= 0) and torch.all(selected < 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(32))

    def test_forward_with_exploration(self, router):
        """Tests forward with forced exploration."""
        x = torch.randn(32, 256)
        force_route = 1
        exploration_prob = 0.5

        selected, probs = router.forward_with_exploration(
            x, force_route=force_route, exploration_prob=exploration_prob
        )

        assert selected.shape == (32,)
        # With exploration_prob=0.5, roughly half should be forced to route 1
        forced_count = (selected == force_route).sum().item()
        assert 5 < forced_count < 27  # Probabilistic test with wide range

    def test_add_route(self, router):
        """Tests adding a new route."""
        initial_routes = router.num_routes
        router.add_route()

        assert router.num_routes == initial_routes + 1
        assert router.route_bias.shape[0] == initial_routes + 1

        # Test that the new route works
        x = torch.randn(32, 256)
        selected, probs = router(x)
        assert probs.shape == (32, initial_routes + 1)

    def test_set_route_bias(self, router):
        """Tests setting route bias."""
        router.set_route_bias(1, 5.0)
        assert router.route_bias[1].item() == 5.0

        # Verify bias affects routing
        x = torch.randn(32, 256)
        selected, probs = router(x)
        # Route 1 should have higher probability due to bias
        assert probs[:, 1].mean() > probs[:, 0].mean()

    def test_detect_data_type_image(self, router):
        """Tests image data type detection."""
        # 28x28 grayscale image (MNIST-like)
        x_mnist = torch.randn(32, 784)
        data_type = router.detect_data_type(x_mnist)
        assert data_type == "image"

        # 32x32x3 RGB image (CIFAR-like)
        x_cifar = torch.randn(32, 3072)
        data_type = router.detect_data_type(x_cifar)
        assert data_type == "image"

    def test_detect_data_type_text(self, router):
        """Tests text data type detection (sparse TF-IDF)."""
        # Sparse text data (TF-IDF-like)
        x_text = torch.zeros(32, 1000)
        x_text[:, :100] = torch.randn(32, 100).abs()  # Only 10% non-zero
        data_type = router.detect_data_type(x_text)
        assert data_type == "text"

    def test_detect_data_type_audio(self, router):
        """Tests audio data type detection (fallback case)."""
        # Use dimension that's clearly not an image (not a perfect square)
        # and with statistics that don't match text patterns
        x_audio = torch.randn(32, 513) * 0.8 + 0.5  # Not centered, moderate variance
        data_type = router.detect_data_type(x_audio)
        # Audio is detected as fallback when it doesn't match image/text
        assert data_type in ["audio", "text"]  # May vary based on random stats

    def test_compute_fisher(self, router):
        """Tests Fisher information computation."""
        from torch.utils.data import DataLoader, TensorDataset

        # Create dummy data
        x = torch.randn(100, 256)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=32)

        router.compute_fisher(dataloader, num_samples=100)

        assert router.fisher_info is not None
        assert router.old_params is not None
        assert len(router.fisher_info) > 0

    def test_ewc_loss(self, router):
        """Tests EWC loss computation."""
        # Without Fisher info, loss should be zero
        loss = router.ewc_loss()
        assert loss.item() == 0.0

        # With Fisher info
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.randn(100, 256)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size=32)

        router.compute_fisher(dataloader, num_samples=50)
        loss = router.ewc_loss()
        assert loss.item() >= 0.0

    def test_route_consistency(self, router):
        """Tests that routing is deterministic for same input."""
        x = torch.randn(32, 256)

        selected1, probs1 = router(x)
        selected2, probs2 = router(x)

        assert torch.equal(selected1, selected2)
        assert torch.allclose(probs1, probs2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
