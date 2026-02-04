"""
ACOC - Variant System (PyTorch)
===============================
Manages 5 variants with lightweight deltas for voting and model averaging.
Uses a RELATIVE threshold based on performance history.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from collections import deque

from ..config import SystemConfig, ModelMetrics


class VariantSystem:
    """
    Manages 5 variants with slightly different weights.

    MAJOR CHANGE: Uses a RELATIVE performance threshold
    based on recent history, not a fixed threshold.

    Instead of maintaining 5 complete models, stores:
    - 1 base model
    - 5 "deltas" (small weight perturbations)
    """

    def __init__(self, config: SystemConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cpu')
        self.deltas: List[Dict[str, torch.Tensor]] = []

        # History of scores for relative threshold calculation
        self.score_history: deque = deque(maxlen=20)

        # Vote statistics
        self.vote_history: List[Tuple[bool, float, str]] = []

    def initialize_deltas(self, model: nn.Module):
        """
        Creates N random deltas based on model weights.

        Deltas are small Gaussian perturbations,
        proportional to the magnitude of each weight.
        """
        self.deltas = []

        for _ in range(self.config.num_variants):
            delta = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Perturbation proportional to weight standard deviation
                    scale = self.config.delta_magnitude * (param.std().item() + 1e-8)
                    delta[name] = torch.randn_like(param) * scale
            self.deltas.append(delta)

    def apply_delta(
        self,
        model: nn.Module,
        delta_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a state_dict with delta applied.
        Does NOT modify the model in place.
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
        Computes a RELATIVE performance threshold based on history.

        Logic:
        - If insufficient history, use low threshold (0.3)
        - Otherwise, threshold = 95% of recent average
        """
        if len(self.score_history) < 3:
            return 0.3  # Initial low threshold

        recent_scores = list(self.score_history)[-5:]
        avg_score = sum(recent_scores) / len(recent_scores)

        # Threshold is ratio * recent average
        threshold = self.config.performance_threshold_ratio * avg_score

        return threshold

    def evaluate_variants(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
    ) -> List[Tuple[int, float]]:
        """
        Evaluates all variants.

        Args:
            model: Base model
            evaluate_fn: Function (model) -> score

        Returns:
            List of (index, score) sorted by descending score
        """
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        scored_variants = []

        for i in range(len(self.deltas)):
            # Apply delta
            variant_state = self.apply_delta(model, i)
            model.load_state_dict(variant_state)

            # Evaluate
            score = evaluate_fn(model)
            scored_variants.append((i, score))

        # Restore original weights
        model.load_state_dict(original_state)

        # Sort by descending score
        scored_variants.sort(key=lambda x: x[1], reverse=True)

        return scored_variants

    def vote_on_expansion(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
        metrics: ModelMetrics
    ) -> Tuple[bool, float, str]:
        """
        Each variant votes on whether expansion is needed.

        CHANGE: Uses a RELATIVE threshold based on history,
        not a fixed 0.7 threshold.

        Returns:
            (should_expand, confidence, reason)
        """
        scored_variants = self.evaluate_variants(model, evaluate_fn)

        # Add best score to history
        best_score = scored_variants[0][1] if scored_variants else 0.0
        self.score_history.append(best_score)

        # Calculate relative threshold
        threshold = self._get_relative_threshold()

        # Also use metrics threshold if available
        metrics_threshold = metrics.get_relative_performance_threshold()
        # Take max of both (more conservative)
        final_threshold = max(threshold, metrics_threshold)

        # Count votes
        expansion_votes = []
        for idx, score in scored_variants:
            votes_expand = score < final_threshold
            expansion_votes.append(votes_expand)

        # Consensus: simple majority
        expand_count = sum(expansion_votes)
        total = len(expansion_votes)
        should_expand = expand_count > total // 2
        confidence = expand_count / total if total > 0 else 0.0

        # Generate detailed reason
        avg_score = sum(s for _, s in scored_variants) / len(scored_variants) if scored_variants else 0
        worst_score = scored_variants[-1][1] if scored_variants else 0

        if should_expand:
            reason = (
                f"Majority vote FOR expansion ({expand_count}/{total}). "
                f"Relative threshold: {final_threshold:.3f}, "
                f"Average score: {avg_score:.3f}, min: {worst_score:.3f}"
            )
        else:
            reason = (
                f"Majority vote AGAINST expansion ({total - expand_count}/{total}). "
                f"Relative threshold: {final_threshold:.3f}, "
                f"Average score: {avg_score:.3f}, max: {best_score:.3f}"
            )

        # Save to history
        self.vote_history.append((should_expand, confidence, reason))

        return should_expand, confidence, reason

    def merge_best_deltas(
        self,
        model: nn.Module,
        evaluate_fn: Callable[[nn.Module], float],
        top_k: Optional[int] = None
    ):
        """
        Selects top-k deltas and merges them into the model.

        Modifies the model IN PLACE.
        """
        if top_k is None:
            top_k = self.config.top_k_merge

        # Evaluate and sort
        scored_variants = self.evaluate_variants(model, evaluate_fn)

        # Select top-k
        top_indices = [idx for idx, _ in scored_variants[:top_k]]
        top_deltas = [self.deltas[i] for i in top_indices]

        if not top_deltas:
            return

        # Average deltas
        merged_delta = {}
        for name in top_deltas[0].keys():
            stacked = torch.stack([d[name] for d in top_deltas])
            merged_delta[name] = stacked.mean(dim=0)

        # Apply merged delta
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
        Weighted merge based on performance scores.
        Variants with better scores contribute more.

        Modifies the model IN PLACE.
        """
        if not scores:
            return

        # Normalize scores to weights via softmax
        score_values = torch.tensor([s for _, s in scores])
        score_values = score_values - score_values.max()  # Numerical stability
        weights = torch.softmax(score_values * 5, dim=0)  # Temperature = 0.2

        # Weighted average of deltas
        merged_delta = {}
        first_idx = scores[0][0]

        for name in self.deltas[first_idx].keys():
            weighted_sum = torch.zeros_like(self.deltas[first_idx][name])
            for (idx, _), w in zip(scores, weights):
                weighted_sum += w.item() * self.deltas[idx][name]
            merged_delta[name] = weighted_sum

        # Apply
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
        Evolves deltas: keeps best, mutates worst.

        Lightweight evolutionary approach to explore weight space.
        """
        if len(scores) < 2:
            return

        # Indices sorted by score
        sorted_indices = [idx for idx, _ in scores]

        # Worst 2 inherit from best 2 (with mutation)
        num_to_replace = min(2, len(sorted_indices) // 2)

        for i in range(num_to_replace):
            worst_idx = sorted_indices[-(i + 1)]
            best_idx = sorted_indices[i]

            # Copy best to worst
            for name in self.deltas[worst_idx].keys():
                self.deltas[worst_idx][name] = self.deltas[best_idx][name].clone()

                # Add mutation
                param = dict(model.named_parameters()).get(name)
                if param is not None:
                    mutation = torch.randn_like(self.deltas[worst_idx][name])
                    mutation *= mutation_rate * (param.std().item() + 1e-8)
                    self.deltas[worst_idx][name] += mutation

    def get_vote_summary(self) -> Dict:
        """Returns a summary of past votes."""
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
