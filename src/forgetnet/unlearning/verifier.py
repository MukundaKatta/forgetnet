"""UnlearningVerifier - checks if knowledge is truly forgotten."""

from __future__ import annotations

import numpy as np


class UnlearningVerifier:
    """Verifies that unlearning actually removed target knowledge.

    Uses multiple verification strategies:
    1. Direct probing: ask the fact directly
    2. Indirect probing: ask related questions that require the fact
    3. Adversarial probing: try to extract via prompt manipulation
    """

    def __init__(self, confidence_threshold: float = 0.15) -> None:
        self.confidence_threshold = confidence_threshold

    def verify_direct(
        self,
        weights: np.ndarray,
        fact_embedding: np.ndarray,
    ) -> tuple[bool, float]:
        """Direct verification: is the fact still accessible?"""
        logit = float(np.dot(weights, fact_embedding))
        conf = 1.0 / (1.0 + np.exp(-logit))
        forgotten = conf < self.confidence_threshold
        return forgotten, conf

    def verify_indirect(
        self,
        weights: np.ndarray,
        indirect_embeddings: list[np.ndarray],
    ) -> tuple[bool, float]:
        """Indirect verification via related facts."""
        if not indirect_embeddings:
            return True, 0.0
        confidences = []
        for emb in indirect_embeddings:
            logit = float(np.dot(weights, emb))
            conf = 1.0 / (1.0 + np.exp(-logit))
            confidences.append(conf)
        avg_conf = float(np.mean(confidences))
        forgotten = avg_conf < self.confidence_threshold
        return forgotten, avg_conf

    def verify_adversarial(
        self,
        weights: np.ndarray,
        fact_embedding: np.ndarray,
        perturbation_scale: float = 0.1,
        num_perturbations: int = 10,
        seed: int = 42,
    ) -> tuple[bool, float]:
        """Adversarial verification: try perturbed queries."""
        rng = np.random.RandomState(seed)
        max_conf = 0.0
        for _ in range(num_perturbations):
            perturbed = fact_embedding + rng.normal(0, perturbation_scale, fact_embedding.shape)
            logit = float(np.dot(weights, perturbed))
            conf = 1.0 / (1.0 + np.exp(-logit))
            max_conf = max(max_conf, conf)
        forgotten = max_conf < self.confidence_threshold
        return forgotten, max_conf

    def full_verification(
        self,
        weights: np.ndarray,
        fact_embedding: np.ndarray,
        indirect_embeddings: list[np.ndarray] | None = None,
    ) -> dict[str, tuple[bool, float]]:
        """Run all verification methods."""
        results: dict[str, tuple[bool, float]] = {}
        results["direct"] = self.verify_direct(weights, fact_embedding)
        results["indirect"] = self.verify_indirect(
            weights, indirect_embeddings or []
        )
        results["adversarial"] = self.verify_adversarial(weights, fact_embedding)
        return results
