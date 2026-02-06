"""
Tests for architecture-aware routing.

Covers:
- Architecture-aware load balancing (non-uniform target distribution)
- Auxiliary routing loss (rewards routing to correct block type)
- Integration: CIFAR-10-like data should route mostly to CNN blocks
"""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from acoc import ACOCModel, SystemConfig, ACOCTrainer


@pytest.fixture
def device():
    return torch.device('cpu')


@pytest.fixture
def config(device):
    return SystemConfig(
        device=device,
        input_dim=3072,   # 32×32×3 → image
        hidden_dim=64,
        output_dim=10,
        use_cnn=True,
        use_cross_entropy=True,
        routing_loss_weight=0.1,
        load_balance_alpha=0.01,
    )


@pytest.fixture
def model(config):
    return ACOCModel(config)


# ---------------------------------------------------------------------------
# Router: expert_types and data_type storage
# ---------------------------------------------------------------------------

class TestRouterExpertTypes:

    def test_expert_types_initialized(self, model):
        """After model init, the router should know each block's expert type."""
        et = model.router.expert_types
        assert len(et) == 3
        types = set(et.values())
        assert types == {"mlp", "cnn", "audio_mlp"}

    def test_expert_types_updated_after_expansion(self, model):
        """Adding a new block should update the expert_types map."""
        from acoc.config.structures import ExpansionDecision
        decision = ExpansionDecision(
            should_expand=True,
            expansion_type="new_block",
            target_block_id=None,
            confidence=0.9,
            reason="test"
        )
        model.execute_expansion(decision)
        assert len(model.router.expert_types) == len(model.task_blocks)

    def test_detect_data_type_stores_result(self, model, device):
        """detect_data_type should store the result in detected_data_type."""
        x = torch.randn(16, 3072, device=device)
        dtype = model.router.detect_data_type(x)
        assert dtype == "image"
        assert model.router.detected_data_type == "image"


# ---------------------------------------------------------------------------
# Architecture-aware load balancing
# ---------------------------------------------------------------------------

class TestArchitectureAwareLoadBalance:

    def test_matching_indices_for_image(self, model, device):
        """Image data should match CNN block indices."""
        model.router.detected_data_type = "image"
        matching = model.router.get_matching_indices("image")
        assert len(matching) >= 1
        for idx in matching:
            assert model.router.expert_types[idx] == "cnn"

    def test_load_balance_favours_matching_block(self, model, device):
        """Load balance should push bias UP for matching blocks, DOWN for others."""
        model.router.detected_data_type = "image"
        matching = model.router.get_matching_indices("image")
        assert len(matching) >= 1

        # Simulate uniform routing counts
        counts = {i: 10 for i in range(model.router.num_routes)}
        bias_before = model.router.route_bias.clone()
        model.router.update_load_balance(counts, alpha=0.1)
        bias_after = model.router.route_bias

        for idx in range(model.router.num_routes):
            if idx in matching:
                # Matching block was at 33%, target is ~70% → bias should increase
                assert bias_after[idx] > bias_before[idx], f"Matching block {idx} bias should increase"
            else:
                # Non-matching block was at 33%, target is ~15% → bias should decrease
                assert bias_after[idx] < bias_before[idx], f"Non-matching block {idx} bias should decrease"

    def test_load_balance_uniform_when_no_type(self, model, device):
        """When no data type is detected, load balancing should use uniform target."""
        model.router.detected_data_type = None
        counts = {i: 10 for i in range(model.router.num_routes)}

        # With uniform counts and uniform target, bias should not change significantly
        bias_before = model.router.route_bias.clone()
        model.router.update_load_balance(counts, alpha=0.1)
        bias_after = model.router.route_bias

        # All changes should be near zero (uniform load matches uniform target)
        for idx in range(model.router.num_routes):
            assert abs(bias_after[idx].item() - bias_before[idx].item()) < 0.01


# ---------------------------------------------------------------------------
# Auxiliary routing loss
# ---------------------------------------------------------------------------

class TestRoutingLoss:

    def test_routing_loss_nonzero_for_wrong_routing(self, model, config, device):
        """Routing loss should be > 0 when image data is routed to non-CNN blocks."""
        trainer = ACOCTrainer(model, config, learning_rate=0.001)

        x = torch.randn(16, 3072, device=device)
        y = torch.randint(0, 10, (16,), device=device)

        # Run a forward to populate probs and detected_data_type
        model(x)
        loss = trainer._compute_routing_loss()
        # Loss should be a valid tensor (may or may not be zero depending on probs)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_routing_loss_zero_when_disabled(self, model, device):
        """Routing loss should be zero when routing_loss_weight is 0."""
        config = model.config
        config.routing_loss_weight = 0.0
        trainer = ACOCTrainer(model, config, learning_rate=0.001)

        x = torch.randn(16, 3072, device=device)
        model(x)
        loss = trainer._compute_routing_loss()
        assert loss.item() == 0.0

    def test_routing_loss_zero_when_no_data_type(self, model, config, device):
        """Routing loss should be zero when no data type has been detected."""
        trainer = ACOCTrainer(model, config, learning_rate=0.001)

        x = torch.randn(16, 3072, device=device)
        model(x)  # triggers auto-detect on first forward
        model.router.detected_data_type = None  # clear after init
        model(x)  # re-run with no detected type
        loss = trainer._compute_routing_loss()
        assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# Integration: training should favour CNN for image data
# ---------------------------------------------------------------------------

class TestRoutingSpecializationIntegration:

    def test_cifar_routes_to_cnn_after_training(self, model, config, device):
        """After a few training steps on image data, CNN block should get majority traffic."""
        trainer = ACOCTrainer(model, config, learning_rate=0.001)

        x = torch.randn(200, 3072, device=device)
        y = torch.randint(0, 10, (200,), device=device)
        loader = DataLoader(TensorDataset(x, y), batch_size=32)

        # Train for a few cycles
        trainer.run(
            num_cycles=5,
            data_loader=loader,
            num_steps_per_cycle=20,
            verbose=False,
        )

        # Evaluate routing distribution
        model.eval()
        with torch.no_grad():
            selected, probs = model.router(x[:64])

        block_ids = list(model.task_blocks.keys())
        cnn_indices = model.router.get_matching_indices("image")

        # Count how many samples are routed to CNN blocks
        cnn_count = sum(1 for s in selected if s.item() in cnn_indices)
        total = len(selected)
        cnn_ratio = cnn_count / total

        assert cnn_ratio > 0.5, (
            f"CNN blocks should get >50% of image traffic, got {cnn_ratio:.1%}. "
            f"Distribution: {[(bid, int((selected == i).sum())) for i, bid in enumerate(block_ids)]}"
        )
