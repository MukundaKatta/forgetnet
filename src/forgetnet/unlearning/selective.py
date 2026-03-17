"""SelectiveForgetter - targets specific facts for removal."""

from __future__ import annotations

import numpy as np

from forgetnet.models import UnlearningResult
from forgetnet.unlearning.eraser import KnowledgeEraser
from forgetnet.unlearning.verifier import UnlearningVerifier


class SelectiveForgetter:
    """Selectively forgets specific facts while preserving others.

    Uses iterative gradient ascent with regularization to minimize
    collateral damage to non-target knowledge.
    """

    def __init__(
        self,
        learning_rate: float = 0.005,
        max_steps: int = 200,
        regularization: float = 0.1,
        confidence_threshold: float = 0.1,
    ) -> None:
        self.eraser = KnowledgeEraser(
            learning_rate=learning_rate,
            max_steps=max_steps,
            convergence_threshold=confidence_threshold,
        )
        self.verifier = UnlearningVerifier(
            confidence_threshold=confidence_threshold,
        )
        self.regularization = regularization

    def forget(
        self,
        target_facts: list[str],
        target_embeddings: list[np.ndarray],
        weights: np.ndarray,
        preserve_embeddings: list[np.ndarray] | None = None,
    ) -> tuple[list[UnlearningResult], np.ndarray]:
        """Forget multiple facts selectively.

        Args:
            target_facts: Facts to forget.
            target_embeddings: Embeddings of facts to forget.
            weights: Current model weights.
            preserve_embeddings: Embeddings of facts to preserve.
        """
        results: list[UnlearningResult] = []
        w = weights.copy()
        original_w = weights.copy()
        for fact, emb in zip(target_facts, target_embeddings):
            result, w = self.eraser.erase(
                target_fact=fact,
                fact_embedding=emb,
                weights=w,
                related_embeddings=preserve_embeddings,
            )
            # Regularization: don't drift too far from original
            drift = w - original_w
            w = w - self.regularization * drift
            results.append(result)
        return results, w

    def forget_and_verify(
        self,
        target_facts: list[str],
        target_embeddings: list[np.ndarray],
        weights: np.ndarray,
        preserve_embeddings: list[np.ndarray] | None = None,
    ) -> tuple[list[UnlearningResult], dict[str, dict], np.ndarray]:
        """Forget facts and verify they are truly gone."""
        results, new_weights = self.forget(
            target_facts, target_embeddings, weights, preserve_embeddings
        )
        verifications: dict[str, dict] = {}
        for fact, emb in zip(target_facts, target_embeddings):
            v = self.verifier.full_verification(
                new_weights, emb, preserve_embeddings
            )
            verifications[fact] = {k: {"forgotten": f, "confidence": c} for k, (f, c) in v.items()}
        return results, verifications, new_weights
