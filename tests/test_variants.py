"""
Tests pour le système de variantes et de vote.
"""

import torch
import torch.nn as nn
import pytest
from acoc.config import SystemConfig, ModelMetrics
from acoc.variants import VariantSystem


class SimpleModel(nn.Module):
    """Modèle simple pour les tests."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestVariantSystem:
    """Tests pour VariantSystem."""

    @pytest.fixture
    def config(self):
        """Configuration de test."""
        return SystemConfig(
            num_variants=5,
            delta_magnitude=0.01,
            performance_threshold_ratio=0.95
        )

    @pytest.fixture
    def variant_system(self, config):
        """Système de variantes pour les tests."""
        return VariantSystem(config, device=torch.device('cpu'))

    @pytest.fixture
    def model(self):
        """Modèle simple pour les tests."""
        return SimpleModel()

    def test_initialization(self, config):
        """Test l'initialisation du système."""
        vs = VariantSystem(config, device=torch.device('cpu'))
        assert vs.config == config
        assert len(vs.deltas) == 0
        assert len(vs.score_history) == 0
        assert len(vs.vote_history) == 0

    def test_initialize_deltas(self, variant_system, model):
        """Test l'initialisation des deltas."""
        variant_system.initialize_deltas(model)

        assert len(variant_system.deltas) == 5
        for delta in variant_system.deltas:
            assert isinstance(delta, dict)
            assert len(delta) > 0  # Au moins quelques paramètres

    def test_deltas_are_small(self, variant_system, model):
        """Test que les deltas sont petits par rapport au modèle."""
        variant_system.initialize_deltas(model)

        for delta in variant_system.deltas:
            for name, param_delta in delta.items():
                param = dict(model.named_parameters())[name]
                # Le delta doit être petit comparé au paramètre
                ratio = param_delta.abs().mean() / (param.abs().mean() + 1e-8)
                assert ratio < 0.1  # Moins de 10% en moyenne

    def test_apply_delta(self, variant_system, model):
        """Test l'application d'un delta."""
        variant_system.initialize_deltas(model)

        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        variant_state = variant_system.apply_delta(model, delta_idx=0)

        # Le state original ne doit pas être modifié
        for name, param in model.state_dict().items():
            assert torch.equal(param, original_state[name])

        # Le variant_state doit être différent
        for name in variant_state:
            if name in variant_system.deltas[0]:
                assert not torch.equal(variant_state[name], original_state[name])

    def test_relative_threshold(self, variant_system):
        """Test le calcul du seuil relatif."""
        # Pas d'historique
        threshold = variant_system._get_relative_threshold()
        assert threshold == 0.3  # Seuil par défaut

        # Avec historique
        variant_system.score_history.extend([0.8, 0.85, 0.9, 0.88, 0.92])
        threshold = variant_system._get_relative_threshold()
        expected = 0.95 * (sum([0.9, 0.88, 0.92, 0.85, 0.8]) / 5)
        assert abs(threshold - expected) < 0.01

    def test_evaluate_variants(self, variant_system, model):
        """Test l'évaluation des variantes."""
        variant_system.initialize_deltas(model)

        def dummy_evaluate(m):
            """Fonction d'évaluation factice."""
            return torch.randn(1).item()

        scored_variants = variant_system.evaluate_variants(model, dummy_evaluate)

        assert len(scored_variants) == 5
        # Vérifie que c'est trié par score décroissant
        for i in range(len(scored_variants) - 1):
            assert scored_variants[i][1] >= scored_variants[i + 1][1]

    def test_vote_on_expansion(self, variant_system, model):
        """Test le vote sur l'expansion."""
        variant_system.initialize_deltas(model)
        metrics = ModelMetrics()

        def dummy_evaluate(m):
            """Retourne un score faible pour déclencher expansion."""
            return 0.2

        should_expand, confidence, reason = variant_system.vote_on_expansion(
            model, dummy_evaluate, metrics
        )

        assert isinstance(should_expand, bool)
        assert 0.0 <= confidence <= 1.0
        assert isinstance(reason, str)
        assert len(variant_system.score_history) > 0

    def test_merge_best_deltas(self, variant_system, model):
        """Test la fusion des meilleurs deltas."""
        variant_system.initialize_deltas(model)

        original_params = {name: param.clone() for name, param in model.named_parameters()}

        def dummy_evaluate(m):
            return torch.randn(1).item()

        variant_system.merge_best_deltas(model, dummy_evaluate, top_k=3)

        # Les paramètres doivent avoir changé
        changed = False
        for name, param in model.named_parameters():
            if not torch.equal(param, original_params[name]):
                changed = True
                break
        assert changed

    def test_vote_history_tracking(self, variant_system, model):
        """Test que l'historique des votes est suivi."""
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
