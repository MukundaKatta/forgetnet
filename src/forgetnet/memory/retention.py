"""RetentionTester - measures how LLMs retain information across context."""

from __future__ import annotations

import numpy as np

from forgetnet.models import (
    ForgettingCurvePoint,
    MemoryProbe,
    RetentionResult,
    RetentionType,
)


class RetentionTester:
    """Tests retention of information across varying context distances.

    Simulates how well a model retains facts as the distance between
    the fact and the query increases (measured in token positions).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        seed: int = 42,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.rng = np.random.RandomState(seed)

    def create_probes(
        self,
        facts: list[tuple[str, str]],
        category: str = "general",
    ) -> list[MemoryProbe]:
        """Create memory probes from (fact, expected_answer) pairs."""
        probes = []
        for i, (fact, answer) in enumerate(facts):
            probes.append(
                MemoryProbe(
                    probe_id=i,
                    fact=fact,
                    expected_answer=answer,
                    context_position=i * 100,
                    category=category,
                )
            )
        return probes

    def _compute_similarity(self, expected: str, actual: str) -> float:
        """Simple character-level similarity (Jaccard on character n-grams)."""
        if not expected or not actual:
            return 0.0
        n = 3
        def ngrams(s: str) -> set[str]:
            s = s.lower().strip()
            return {s[i : i + n] for i in range(max(len(s) - n + 1, 1))}

        e_set = ngrams(expected)
        a_set = ngrams(actual)
        if not e_set or not a_set:
            return 1.0 if expected.lower().strip() == actual.lower().strip() else 0.0
        intersection = e_set & a_set
        union = e_set | a_set
        return len(intersection) / len(union)

    def test_retention(
        self,
        probes: list[MemoryProbe],
        responses: list[str],
        query_positions: list[int] | None = None,
    ) -> list[RetentionResult]:
        """Test retention by comparing responses to expected answers.

        Args:
            probes: Memory probes with expected answers.
            responses: Model responses for each probe.
            query_positions: Where in context each query was placed.
        """
        if query_positions is None:
            query_positions = [p.context_position + 500 for p in probes]
        results = []
        for probe, response, q_pos in zip(probes, responses, query_positions):
            score = self._compute_similarity(probe.expected_answer, response)
            recalled = score >= self.similarity_threshold
            distance = abs(q_pos - probe.context_position)
            rtype = RetentionType.EXACT if score > 0.95 else (
                RetentionType.SEMANTIC if score > 0.5 else RetentionType.PARTIAL
            )
            results.append(
                RetentionResult(
                    probe=probe,
                    recalled=recalled,
                    retention_score=score,
                    context_distance=distance,
                    retention_type=rtype,
                )
            )
        return results

    def simulate_retention_decay(
        self,
        num_facts: int = 20,
        max_distance: int = 5000,
        num_steps: int = 10,
        decay_rate: float = 0.001,
    ) -> list[ForgettingCurvePoint]:
        """Simulate retention decay over context distance."""
        distances = np.linspace(0, max_distance, num_steps)
        points = []
        for d in distances:
            # Exponential decay model
            retention = np.exp(-decay_rate * d)
            # Add noise
            noise = self.rng.normal(0, 0.02)
            retention = np.clip(retention + noise, 0, 1)
            points.append(
                ForgettingCurvePoint(
                    time_step=float(d),
                    retention_rate=float(retention),
                    num_probes=num_facts,
                )
            )
        return points
