"""Tests for forgetnet."""

import numpy as np
import pytest

from forgetnet.memory.retention import RetentionTester
from forgetnet.memory.forgetting import ForgettingCurveAnalyzer
from forgetnet.memory.interference import InterferenceDetector
from forgetnet.unlearning.eraser import KnowledgeEraser
from forgetnet.unlearning.verifier import UnlearningVerifier
from forgetnet.unlearning.selective import SelectiveForgetter
from forgetnet.models import ForgettingCurvePoint, MemoryProbe


class TestRetentionTester:
    def test_create_probes(self):
        tester = RetentionTester()
        facts = [("Paris is the capital of France", "Paris")]
        probes = tester.create_probes(facts)
        assert len(probes) == 1
        assert probes[0].expected_answer == "Paris"

    def test_retention_scoring(self):
        tester = RetentionTester()
        probes = [
            MemoryProbe(probe_id=0, fact="test", expected_answer="Paris", context_position=0)
        ]
        results = tester.test_retention(probes, ["Paris"], [500])
        assert results[0].retention_score > 0.9

    def test_simulate_decay(self):
        tester = RetentionTester()
        points = tester.simulate_retention_decay(num_steps=5)
        assert len(points) == 5
        # Retention should generally decrease
        assert points[0].retention_rate > points[-1].retention_rate


class TestForgettingCurveAnalyzer:
    def test_ebbinghaus_formula(self):
        t = np.array([0.0, 1.0, 2.0])
        r = ForgettingCurveAnalyzer.ebbinghaus(t, b=1.0, s=1.0)
        assert abs(r[0] - 1.0) < 1e-6
        assert abs(r[1] - np.exp(-1)) < 1e-6

    def test_fit_ebbinghaus(self):
        analyzer = ForgettingCurveAnalyzer()
        # Generate perfect exponential data
        points = [
            ForgettingCurvePoint(
                time_step=float(t), retention_rate=float(np.exp(-t / 500)), num_probes=10
            )
            for t in range(0, 2001, 200)
        ]
        curve = analyzer.fit_ebbinghaus(points)
        assert curve.r_squared > 0.9
        assert curve.initial_strength > 0.8

    def test_half_life(self):
        analyzer = ForgettingCurveAnalyzer()
        points = [
            ForgettingCurvePoint(time_step=float(t), retention_rate=float(np.exp(-t / 100)))
            for t in range(0, 501, 50)
        ]
        curve = analyzer.fit_ebbinghaus(points)
        hl = analyzer.half_life(curve)
        # Half-life should be ~S * ln(2) = ~69.3
        assert 30 < hl < 150

    def test_power_law_fit(self):
        analyzer = ForgettingCurveAnalyzer()
        points = [
            ForgettingCurvePoint(
                time_step=float(t), retention_rate=float((1.0 + t) ** (-0.5))
            )
            for t in range(0, 11)
        ]
        curve = analyzer.fit_power_law(points)
        assert curve.model_type == "power_law"
        assert curve.r_squared > 0.9

    def test_predict_retention(self):
        analyzer = ForgettingCurveAnalyzer()
        points = [
            ForgettingCurvePoint(time_step=float(t), retention_rate=float(np.exp(-t / 100)))
            for t in range(0, 501, 50)
        ]
        curve = analyzer.fit_ebbinghaus(points)
        pred = analyzer.predict_retention(curve, 0.0)
        assert 0.8 < pred <= 1.1


class TestInterferenceDetector:
    def test_proactive(self):
        detector = InterferenceDetector()
        baseline = [0.9, 0.85, 0.88]
        post = [0.6, 0.55, 0.58]
        result = detector.test_proactive_interference(baseline, post)
        assert result.interference_type == "proactive"
        assert result.interference_magnitude > 0.2

    def test_retroactive(self):
        detector = InterferenceDetector()
        baseline = [0.9, 0.85, 0.88]
        post = [0.7, 0.65, 0.68]
        result = detector.test_retroactive_interference(baseline, post)
        assert result.interference_type == "retroactive"
        assert result.interference_magnitude > 0.1

    def test_simulate(self):
        detector = InterferenceDetector()
        result = detector.simulate_interference(similarity=0.9)
        assert result.interference_magnitude > 0


class TestKnowledgeEraser:
    def test_erase(self):
        rng = np.random.RandomState(42)
        weights = rng.randn(8).astype(np.float32)
        emb = rng.randn(8).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        eraser = KnowledgeEraser(learning_rate=0.1, max_steps=200)
        result, new_w = eraser.erase("test_fact", emb, weights)
        assert result.post_unlearn_confidence < result.pre_unlearn_confidence


class TestUnlearningVerifier:
    def test_verify_direct(self):
        verifier = UnlearningVerifier(confidence_threshold=0.5)
        weights = np.zeros(8)
        emb = np.ones(8)
        forgotten, conf = verifier.verify_direct(weights, emb)
        assert conf == 0.5  # sigmoid(0) = 0.5
        assert not forgotten  # 0.5 is not < 0.5


class TestSelectiveForgetter:
    def test_forget_and_verify(self):
        rng = np.random.RandomState(42)
        dim = 8
        weights = rng.randn(dim).astype(np.float32) * 2
        target_embs = [rng.randn(dim).astype(np.float32) for _ in range(2)]
        target_embs = [e / np.linalg.norm(e) for e in target_embs]
        preserve = [rng.randn(dim).astype(np.float32) for _ in range(3)]
        preserve = [e / np.linalg.norm(e) for e in preserve]
        forgetter = SelectiveForgetter(learning_rate=0.05, max_steps=300)
        results, verif, new_w = forgetter.forget_and_verify(
            ["fact_0", "fact_1"], target_embs, weights, preserve
        )
        assert len(results) == 2
        assert "fact_0" in verif
