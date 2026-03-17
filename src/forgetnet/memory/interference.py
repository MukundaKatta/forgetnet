"""InterferenceDetector - tests proactive and retroactive interference."""

from __future__ import annotations

import numpy as np

from forgetnet.models import InterferenceResult


class InterferenceDetector:
    """Detects proactive and retroactive interference in AI memory.

    Proactive interference: old memories interfere with new learning.
    Retroactive interference: new learning interferes with old memories.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.RandomState(seed)

    def test_proactive_interference(
        self,
        baseline_scores: list[float],
        post_learning_scores: list[float],
    ) -> InterferenceResult:
        """Measure proactive interference.

        Compare retention of new facts when old similar facts exist (post)
        vs. without prior knowledge (baseline).
        """
        baseline_mean = float(np.mean(baseline_scores))
        post_mean = float(np.mean(post_learning_scores))
        magnitude = max(baseline_mean - post_mean, 0.0)
        return InterferenceResult(
            baseline_retention=baseline_mean,
            post_interference_retention=post_mean,
            interference_type="proactive",
            interference_magnitude=magnitude,
            num_interfering_items=len(baseline_scores),
        )

    def test_retroactive_interference(
        self,
        baseline_scores: list[float],
        post_new_learning_scores: list[float],
    ) -> InterferenceResult:
        """Measure retroactive interference.

        Compare retention of old facts before vs. after learning new facts.
        """
        baseline_mean = float(np.mean(baseline_scores))
        post_mean = float(np.mean(post_new_learning_scores))
        magnitude = max(baseline_mean - post_mean, 0.0)
        return InterferenceResult(
            baseline_retention=baseline_mean,
            post_interference_retention=post_mean,
            interference_type="retroactive",
            interference_magnitude=magnitude,
            num_interfering_items=len(post_new_learning_scores),
        )

    def simulate_interference(
        self,
        num_original: int = 10,
        num_interfering: int = 10,
        similarity: float = 0.7,
        interference_type: str = "retroactive",
    ) -> InterferenceResult:
        """Simulate interference effects.

        Higher similarity between original and interfering items
        causes stronger interference.
        """
        # Baseline retention without interference
        baseline = self.rng.uniform(0.7, 1.0, num_original)
        # Interference effect scales with similarity
        interference_strength = similarity * 0.4
        noise = self.rng.normal(0, 0.05, num_original)
        post_scores = np.clip(baseline - interference_strength + noise, 0, 1)
        return InterferenceResult(
            baseline_retention=float(np.mean(baseline)),
            post_interference_retention=float(np.mean(post_scores)),
            interference_type=interference_type,
            interference_magnitude=float(
                np.mean(baseline) - np.mean(post_scores)
            ),
            num_interfering_items=num_interfering,
        )
