"""KnowledgeEraser - implements gradient ascent unlearning."""

from __future__ import annotations

import numpy as np

from forgetnet.models import UnlearningResult


class KnowledgeEraser:
    """Erases specific knowledge via gradient ascent on target facts.

    Gradient ascent unlearning works by maximizing the loss on target
    facts, effectively pushing the model away from those outputs.
    This is a simulation of the approach using numpy weight vectors.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_steps: int = 100,
        convergence_threshold: float = 0.1,
    ) -> None:
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.convergence_threshold = convergence_threshold

    def _simulate_model_confidence(
        self,
        weights: np.ndarray,
        fact_embedding: np.ndarray,
    ) -> float:
        """Simulate model confidence as sigmoid of dot product."""
        logit = float(np.dot(weights, fact_embedding))
        return 1.0 / (1.0 + np.exp(-logit))

    def erase(
        self,
        target_fact: str,
        fact_embedding: np.ndarray,
        weights: np.ndarray,
        related_embeddings: list[np.ndarray] | None = None,
    ) -> tuple[UnlearningResult, np.ndarray]:
        """Perform gradient ascent unlearning on a single fact.

        Returns updated weights and an UnlearningResult.
        """
        pre_conf = self._simulate_model_confidence(weights, fact_embedding)
        w = weights.copy()
        steps = 0
        for step in range(self.max_steps):
            conf = self._simulate_model_confidence(w, fact_embedding)
            if conf < self.convergence_threshold:
                break
            # Gradient ascent: increase loss = move away from correct output
            # d/dw sigmoid(w.f) * f = sigmoid * (1-sigmoid) * f
            grad = conf * (1 - conf) * fact_embedding
            w += self.learning_rate * grad  # ascent, not descent
            steps = step + 1
        post_conf = self._simulate_model_confidence(w, fact_embedding)
        # Measure collateral damage on related facts
        collateral = 0.0
        if related_embeddings:
            pre_related = [
                self._simulate_model_confidence(weights, e) for e in related_embeddings
            ]
            post_related = [
                self._simulate_model_confidence(w, e) for e in related_embeddings
            ]
            damages = [
                max(pre - post, 0.0) for pre, post in zip(pre_related, post_related)
            ]
            collateral = float(np.mean(damages)) if damages else 0.0
        return (
            UnlearningResult(
                target_fact=target_fact,
                pre_unlearn_confidence=float(pre_conf),
                post_unlearn_confidence=float(post_conf),
                collateral_damage=collateral,
                num_gradient_steps=steps,
                success=post_conf < self.convergence_threshold,
            ),
            w,
        )
