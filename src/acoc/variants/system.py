"""
ACOC - Système de Variantes (PyTorch)
=====================================
Gestion des 5 variantes avec deltas légers pour le vote et le model averaging.
Utilise un seuil RELATIF basé sur l'historique des performances.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from collections import deque

from ..config import SystemConfig, ModelMetrics


class VariantSystem:
    """
    Gère les 5 variantes avec poids légèrement différents.

    CHANGEMENT MAJEUR: Utilise un seuil de performance RELATIF
    basé sur l'historique récent, pas un seuil fixe.

    Au lieu de maintenir 5 modèles complets, on stocke:
    - 1 modèle de base
    - 5 "deltas" (petites perturbations des poids)
    """

    def __init__(self, config: SystemConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        self.deltas: List[Dict[str, torch.Tensor]] = []

        # Historique des scores pour le seuil relatif
        self.score_history: deque = deque(maxlen=20)

        # Statistiques des votes
        self.vote_history: List[Tuple[bool, float, str]] = []

    def initialize_deltas(self, model: nn.Module):
        """
        Crée N deltas aléatoires basés sur les poids du modèle.

        Les deltas sont de petites perturbations gaussiennes,
        proportionnelles à la magnitude de chaque poids.
        """
        self.deltas = []

        for _ in range(self.config.num_variants):
            delta = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Perturbation proportionnelle à l'écart-type des poids
                    scale = self.config.delta_magnitude * (param.std().item() + 1e-8)
                    delta[name] = torch.randn_like(param) * scale
            self.deltas.append(delta)

    def apply_delta(
        self,
        model: nn.Module,
        delta_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Retourne un state_dict avec le delta appliqué.
        Ne modifie PAS le modèle en place.
        """
        if delta_idx >= len(self.deltas):
            return model.state_dict()

        delta = self.deltas[delta_idx]
        new_state = {}

        for name, param in model.state_dict().items():
            if name in delta:
                new_state[name] = param + delta[name].to(param.device)
            else:
                new_state[name] = param

        return new_state

    def _get_relative_threshold(self) -> float:
        """
        Calcule un seuil de performance RELATIF basé sur l'historique.

        Logique:
        - Si on a peu d'historique, seuil bas (0.3)
        - Sinon, seuil = 95% de la moyenne récente
        """
        if len(self.score_history) < 3:
            return 0.3  # Seuil initial bas

        recent_scores = list(self.score_history)[-5:]
        avg_score = sum(recent_scores) / len(recent_scores)

        # Le seuil est ratio * moyenne récente
        threshold = self.config.performance_threshold_ratio * avg_score

        return threshold

    def evaluate_variants(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
    ) -> List[Tuple[int, float]]:
        """
        Évalue toutes les variantes.

        Args:
            model: Modèle de base
            evaluate_fn: Fonction (model) -> score

        Returns:
            Liste de (index, score) triée par score décroissant
        """
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        scored_variants = []

        for i in range(len(self.deltas)):
            # Appliquer le delta
            variant_state = self.apply_delta(model, i)
            model.load_state_dict(variant_state)

            # Évaluer
            score = evaluate_fn(model)
            scored_variants.append((i, score))

        # Restaurer les poids originaux
        model.load_state_dict(original_state)

        # Trier par score décroissant
        scored_variants.sort(key=lambda x: x[1], reverse=True)

        return scored_variants

    def vote_on_expansion(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
        metrics: ModelMetrics
    ) -> Tuple[bool, float, str]:
        """
        Chaque variante vote sur la nécessité d'expansion.

        CHANGEMENT: Utilise un seuil RELATIF basé sur l'historique,
        pas un seuil fixe de 0.7.

        Returns:
            (should_expand, confidence, reason)
        """
        scored_variants = self.evaluate_variants(model, evaluate_fn)

        # Ajouter le meilleur score à l'historique
        best_score = scored_variants[0][1] if scored_variants else 0.0
        self.score_history.append(best_score)

        # Calculer le seuil relatif
        threshold = self._get_relative_threshold()

        # Aussi utiliser le seuil des métriques si disponible
        metrics_threshold = metrics.get_relative_performance_threshold()
        # Prendre le max des deux (plus conservateur)
        final_threshold = max(threshold, metrics_threshold)

        # Compter les votes
        expansion_votes = []
        for idx, score in scored_variants:
            votes_expand = score < final_threshold
            expansion_votes.append(votes_expand)

        # Consensus: majorité simple
        expand_count = sum(expansion_votes)
        total = len(expansion_votes)
        should_expand = expand_count > total // 2
        confidence = expand_count / total if total > 0 else 0.0

        # Générer la raison détaillée
        avg_score = sum(s for _, s in scored_variants) / len(scored_variants) if scored_variants else 0
        worst_score = scored_variants[-1][1] if scored_variants else 0

        if should_expand:
            reason = (
                f"Vote majoritaire POUR expansion ({expand_count}/{total}). "
                f"Seuil relatif: {final_threshold:.3f}, "
                f"Score moyen: {avg_score:.3f}, min: {worst_score:.3f}"
            )
        else:
            reason = (
                f"Vote majoritaire CONTRE expansion ({total - expand_count}/{total}). "
                f"Seuil relatif: {final_threshold:.3f}, "
                f"Score moyen: {avg_score:.3f}, max: {best_score:.3f}"
            )

        # Sauvegarder dans l'historique
        self.vote_history.append((should_expand, confidence, reason))

        return should_expand, confidence, reason

    def merge_best_deltas(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
        top_k: Optional[int] = None
    ):
        """
        Sélectionne les top-k deltas et les fusionne dans le modèle.

        Modifie le modèle EN PLACE.
        """
        if top_k is None:
            top_k = self.config.top_k_merge

        # Évaluer et trier
        scored_variants = self.evaluate_variants(model, evaluate_fn)

        # Sélectionner les top-k
        top_indices = [idx for idx, _ in scored_variants[:top_k]]
        top_deltas = [self.deltas[i] for i in top_indices]

        if not top_deltas:
            return

        # Moyenne des deltas
        merged_delta = {}
        for name in top_deltas[0].keys():
            stacked = torch.stack([d[name] for d in top_deltas])
            merged_delta[name] = stacked.mean(dim=0)

        # Appliquer le delta fusionné
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in merged_delta:
                    param.add_(merged_delta[name].to(param.device))

    def weighted_merge(
        self,
        model: nn.Module,
        scores: List[Tuple[int, float]]
    ):
        """
        Fusion pondérée par les scores de performance.
        Les variantes avec de meilleurs scores contribuent plus.

        Modifie le modèle EN PLACE.
        """
        if not scores:
            return

        # Normaliser les scores en poids via softmax
        score_values = torch.tensor([s for _, s in scores])
        score_values = score_values - score_values.max()  # Stabilité
        weights = torch.softmax(score_values * 5, dim=0)  # Température = 0.2

        # Moyenne pondérée des deltas
        merged_delta = {}
        first_idx = scores[0][0]

        for name in self.deltas[first_idx].keys():
            weighted_sum = torch.zeros_like(self.deltas[first_idx][name])
            for (idx, _), w in zip(scores, weights):
                weighted_sum += w.item() * self.deltas[idx][name]
            merged_delta[name] = weighted_sum

        # Appliquer
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in merged_delta:
                    param.add_(merged_delta[name].to(param.device))

    def evolve_deltas(
        self,
        model: nn.Module,
        scores: List[Tuple[int, float]],
        mutation_rate: float = 0.1
    ):
        """
        Fait évoluer les deltas: garde les meilleurs, mute les pires.

        Approche évolutionnaire légère pour explorer l'espace des poids.
        """
        if len(scores) < 2:
            return

        # Indices triés par score
        sorted_indices = [idx for idx, _ in scores]

        # Les 2 pires héritent des 2 meilleurs (avec mutation)
        num_to_replace = min(2, len(sorted_indices) // 2)

        for i in range(num_to_replace):
            worst_idx = sorted_indices[-(i + 1)]
            best_idx = sorted_indices[i]

            # Copier le meilleur vers le pire
            for name in self.deltas[worst_idx].keys():
                self.deltas[worst_idx][name] = self.deltas[best_idx][name].clone()

                # Ajouter une mutation
                param = dict(model.named_parameters()).get(name)
                if param is not None:
                    mutation = torch.randn_like(self.deltas[worst_idx][name])
                    mutation *= mutation_rate * (param.std().item() + 1e-8)
                    self.deltas[worst_idx][name] += mutation

    def get_vote_summary(self) -> Dict:
        """Retourne un résumé des votes passés."""
        if not self.vote_history:
            return {"total": 0, "expand_votes": 0, "avg_confidence": 0}

        expand_count = sum(1 for v, _, _ in self.vote_history if v)
        avg_conf = sum(c for _, c, _ in self.vote_history) / len(self.vote_history)

        return {
            "total": len(self.vote_history),
            "expand_votes": expand_count,
            "avg_confidence": avg_conf,
            "current_threshold": self._get_relative_threshold()
        }
