"""
Tests for expert implementations (MLP, CNN, Factory).
"""

import torch
import pytest
from acoc.experts import MLPExpert, CNNExpert, ExpertFactory
from acoc.config import SystemConfig


class TestMLPExpert:
    """Tests for MLPExpert."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(input_dim=256, hidden_dim=128, output_dim=10)

    @pytest.fixture
    def mlp_expert(self, config):
        """Create MLP expert for tests."""
        return MLPExpert(
            input_dim=256,
            hidden_dim=128,
            output_dim=10,
            name="test_mlp",
            config=config
        )

    def test_initialization(self, mlp_expert):
        """Tests MLP initialization."""
        assert mlp_expert.input_dim == 256
        assert mlp_expert.hidden_dim == 128
        assert mlp_expert.output_dim == 10
        assert mlp_expert.name == "test_mlp"

    def test_forward(self, mlp_expert):
        """Tests forward pass."""
        x = torch.randn(32, 256)
        output = mlp_expert(x)

        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()

    def test_expand_width(self, mlp_expert):
        """Tests width expansion."""
        initial_params = mlp_expert.get_param_count()
        mlp_expert.expand_width(additional_neurons=32)
        new_params = mlp_expert.get_param_count()

        assert new_params > initial_params
        assert mlp_expert.hidden_dim == 128 + 32

        # Test that expanded model still works
        x = torch.randn(16, 256)
        output = mlp_expert(x)
        assert output.shape == (16, 10)

    def test_get_param_count(self, mlp_expert):
        """Tests parameter counting."""
        param_count = mlp_expert.get_param_count()
        assert param_count > 0

        # Manually count parameters
        manual_count = sum(p.numel() for p in mlp_expert.parameters())
        assert param_count == manual_count


class TestCNNExpert:
    """Tests for CNNExpert."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(
            input_dim=3072,  # 32x32x3 for CIFAR
            hidden_dim=128,
            output_dim=10,
            use_cnn=True,
            cnn_channels=[32, 64, 128],
            image_channels=3
        )

    @pytest.fixture
    def cnn_expert(self, config):
        """Create CNN expert for tests."""
        return CNNExpert(
            input_dim=3072,
            hidden_dim=128,
            output_dim=10,
            name="test_cnn",
            config=config
        )

    def test_initialization(self, cnn_expert):
        """Tests CNN initialization."""
        assert cnn_expert.input_dim == 3072
        assert cnn_expert.in_channels == 3
        assert cnn_expert.features is not None
        assert len(list(cnn_expert.features.children())) > 0

    def test_forward(self, cnn_expert):
        """Tests forward pass."""
        x = torch.randn(32, 3072)  # Flattened CIFAR-like images
        output = cnn_expert(x)

        assert output.shape == (32, 10)
        assert not torch.isnan(output).any()

    def test_forward_with_4d_input(self, cnn_expert):
        """Tests forward with 4D tensor input."""
        x = torch.randn(32, 3, 32, 32)  # NCHW format
        output = cnn_expert(x)

        assert output.shape == (32, 10)

    def test_expand_width(self, cnn_expert):
        """Tests width expansion for CNN."""
        initial_params = cnn_expert.get_param_count()
        cnn_expert.expand_width(additional_neurons=32)
        new_params = cnn_expert.get_param_count()

        assert new_params > initial_params

        # Test that expanded model still works
        x = torch.randn(16, 3072)
        output = cnn_expert(x)
        assert output.shape == (16, 10)

    def test_mnist_dimensions(self):
        """Tests CNN with MNIST-like dimensions."""
        config = SystemConfig(
            input_dim=784,  # 28x28 grayscale
            hidden_dim=128,
            output_dim=10,
            use_cnn=True,
            cnn_channels=[32, 64],
            image_channels=1
        )

        cnn = CNNExpert(
            input_dim=784,
            hidden_dim=128,
            output_dim=10,
            name="mnist_cnn",
            config=config
        )

        x = torch.randn(16, 784)
        output = cnn(x)
        assert output.shape == (16, 10)


class TestExpertFactory:
    """Tests for ExpertFactory."""

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return SystemConfig(
            input_dim=256,
            hidden_dim=128,
            output_dim=10,
            use_cnn=True,
            cnn_channels=[32, 64],
            image_channels=3
        )

    def test_create_mlp(self, config):
        """Tests creating MLP expert."""
        expert = ExpertFactory.create(
            expert_type="mlp",
            input_dim=256,
            hidden_dim=128,
            output_dim=10,
            name="test_mlp",
            config=config
        )

        assert isinstance(expert, MLPExpert)
        assert expert.expert_type == "mlp"

    def test_create_cnn(self, config):
        """Tests creating CNN expert."""
        expert = ExpertFactory.create(
            expert_type="cnn",
            input_dim=3072,
            hidden_dim=128,
            output_dim=10,
            name="test_cnn",
            config=config
        )

        assert isinstance(expert, CNNExpert)
        assert expert.expert_type == "cnn"

    def test_create_audio_mlp(self, config):
        """Tests creating audio MLP expert."""
        expert = ExpertFactory.create(
            expert_type="audio_mlp",
            input_dim=512,
            hidden_dim=128,
            output_dim=10,
            name="test_audio",
            config=config
        )

        assert isinstance(expert, MLPExpert)
        assert expert.expert_type == "audio_mlp"

    def test_invalid_expert_type(self, config):
        """Tests that invalid expert type falls back to MLP with warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            expert = ExpertFactory.create(
                expert_type="invalid_type",
                input_dim=256,
                hidden_dim=128,
                output_dim=10,
                name="test",
                config=config
            )

            # Should fallback to MLP
            assert isinstance(expert, MLPExpert)
            # Should have issued a warning
            assert len(w) == 1
            assert "Unknown expert type" in str(w[0].message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
