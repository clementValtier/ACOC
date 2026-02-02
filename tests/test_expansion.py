"""
Tests pour la logique d'expansion du modèle.
"""

import torch
import pytest
from acoc.config import SystemConfig, TaskBlock, ModelMetrics, TaskType, SaturationMetrics
from acoc.management import ExpansionManager
from acoc.experts import MLPExpert


class TestExpansionManager:
    """Tests pour ExpansionManager."""

    @pytest.fixture
    def config(self):
        """Configuration de test."""
        return SystemConfig(
            saturation_threshold=0.6,
            min_cycles_before_expand=2,
            expansion_cooldown=3,
            expansion_ratio=0.1
        )

    @pytest.fixture
    def manager(self, config):
        """Manager d'expansion pour les tests."""
        return ExpansionManager(config)

    @pytest.fixture
    def task_blocks(self, config):
        """Blocs de tâches pour les tests."""
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
        """Test l'initialisation du manager."""
        manager = ExpansionManager(config)
        assert manager.config == config
        assert manager.last_expansion_cycle == -config.expansion_cooldown
        assert len(manager.expansion_history) == 0

    def test_cooldown_prevents_expansion(self, manager, task_blocks):
        """Test que le cooldown empêche l'expansion."""
        metrics = ModelMetrics()

        # Premier cycle
        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=0)
        assert not decision.should_expand
        assert "Historique insuffisant" in decision.reason

    def test_saturation_triggers_expansion(self, manager, task_blocks):
        """Test que la saturation déclenche une expansion."""
        metrics = ModelMetrics()

        # Créer des métriques de saturation élevée
        sat_metrics = SaturationMetrics(
            gradient_flow_ratio=0.3,  # Faible
            activation_saturation=0.8,  # Élevée
            dead_neuron_ratio=0.5,  # Élevée
            activation_variance=0.1  # Faible
        )
        sat_metrics.compute_combined_score()
        metrics.detailed_saturation["test_block"] = sat_metrics

        # Cycle suffisant
        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=5)
        assert decision.should_expand
        assert decision.expansion_type == "width"
        assert decision.target_block_id == "test_block"

    def test_stagnant_loss_triggers_new_block(self, manager, task_blocks):
        """Test que la loss stagnante déclenche un nouveau bloc."""
        metrics = ModelMetrics()

        # Loss stagnante (amélioration < 1%)
        for i in range(15):
            metrics.add_loss(2.9 + i * 0.0001)  # Amélioration minimale

        decision = manager.evaluate_expansion_need(metrics, task_blocks, current_cycle=15)
        assert decision.should_expand
        assert decision.expansion_type == "new_block"

    def test_expand_width(self, manager, task_blocks):
        """Test l'expansion en largeur."""
        initial_params = task_blocks["test_block"].num_params

        success = manager._expand_width("test_block", task_blocks)
        assert success

        # Vérifier que le nombre de paramètres a augmenté
        new_params = task_blocks["test_block"].num_params
        assert new_params > initial_params

    def test_create_new_block(self, manager, task_blocks):
        """Test la création d'un nouveau bloc."""
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
        """Test que l'historique des expansions est bien suivi."""
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
