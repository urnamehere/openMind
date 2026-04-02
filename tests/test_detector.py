"""Tests for openmind.detection: SignalQualityDetector and verdict synthesis."""

import pytest

from openmind.detection.detectors import (
    DetectionFlag,
    DetectionResult,
    SignalQualityDetector,
    SignalVerdict,
)
from openmind.reward.signal import RewardSignal


# ---------------------------------------------------------------------------
# DetectionFlag enum
# ---------------------------------------------------------------------------


class TestDetectionFlag:
    """Basic enum checks."""

    def test_clean_exists(self):
        assert DetectionFlag.CLEAN.value == "clean"

    def test_all_flags_exist(self):
        expected = {
            "clean", "low_confidence", "conflicting_dimensions",
            "reward_gaming", "distribution_shift", "novel_territory",
            "stale_reference", "self_reinforcing", "outlier", "sparse_signal",
        }
        actual = {f.value for f in DetectionFlag}
        assert expected == actual


# ---------------------------------------------------------------------------
# SignalVerdict enum
# ---------------------------------------------------------------------------


class TestSignalVerdict:
    def test_four_verdicts(self):
        assert {v.value for v in SignalVerdict} == {
            "train", "shelve", "ask", "discard",
        }


# ---------------------------------------------------------------------------
# DetectionResult dataclass
# ---------------------------------------------------------------------------


class TestDetectionResult:
    def test_defaults(self):
        result = DetectionResult(
            experience_id="test_1",
            timestamp=0.0,
            verdict=SignalVerdict.TRAIN,
            flags=[DetectionFlag.CLEAN],
            confidence=0.9,
        )
        assert result.training_weight == 1.0
        assert result.suggested_inquiry is None
        assert result.shelve_reason is None

    def test_shelve_result(self):
        result = DetectionResult(
            experience_id="test_2",
            timestamp=0.0,
            verdict=SignalVerdict.SHELVE,
            flags=[DetectionFlag.SELF_REINFORCING],
            confidence=0.3,
            shelve_reason="loop detected",
            revisit_after_days=7.0,
        )
        assert result.verdict == SignalVerdict.SHELVE
        assert result.revisit_after_days == 7.0


# ---------------------------------------------------------------------------
# SignalQualityDetector — individual detector methods
# ---------------------------------------------------------------------------


class TestDimensionalConflictDetector:
    def setup_method(self):
        self.detector = SignalQualityDetector()

    def test_no_conflict_when_aligned(self):
        signal = RewardSignal(task_success=0.8, coherence=0.7, user_satisfaction=0.6)
        result = self.detector._detect_dimensional_conflict(signal)
        assert not result["detected"]

    def test_conflict_satisfied_but_wrong(self):
        signal = RewardSignal(user_satisfaction=0.8, groundedness=-0.5)
        result = self.detector._detect_dimensional_conflict(signal)
        assert result["detected"]
        assert any(c["type"] == "satisfied_but_wrong" for c in result["conflicts"])

    def test_conflict_correct_but_disliked(self):
        signal = RewardSignal(task_success=0.8, user_satisfaction=-0.5)
        result = self.detector._detect_dimensional_conflict(signal)
        assert result["detected"]
        assert any(c["type"] == "correct_but_disliked" for c in result["conflicts"])

    def test_conflict_eloquent_but_useless(self):
        signal = RewardSignal(coherence=0.9, task_success=-0.5)
        result = self.detector._detect_dimensional_conflict(signal)
        assert result["detected"]

    def test_no_conflict_single_dimension(self):
        signal = RewardSignal(task_success=0.8)
        result = self.detector._detect_dimensional_conflict(signal)
        assert not result["detected"]


class TestSignalSparsityDetector:
    def setup_method(self):
        self.detector = SignalQualityDetector()

    def test_sparse_with_one_dimension(self):
        signal = RewardSignal(coherence=0.5)
        result = self.detector._detect_signal_sparsity(signal)
        assert result["detected"]
        assert result["coverage"] == 0.2

    def test_not_sparse_with_three_dimensions(self):
        signal = RewardSignal(
            task_success=0.7, coherence=0.6, groundedness=0.5,
        )
        result = self.detector._detect_signal_sparsity(signal)
        assert not result["detected"]
        assert result["coverage"] == 0.6

    def test_critical_missing_detected(self):
        signal = RewardSignal(coherence=0.5, novelty_value=0.3)
        result = self.detector._detect_signal_sparsity(signal)
        assert "task_success" in result["critical_missing"]
        assert "groundedness" in result["critical_missing"]


class TestNoveltyDetector:
    def setup_method(self):
        self.detector = SignalQualityDetector()
        # Pre-populate baselines so "math" is known
        self.detector.domain_baselines["math"] = {"mean": 0.5, "std": 0.2, "n": 20}

    def test_novel_domain_detected(self):

        class FakeExp:
            domain_tags = ["quantum_computing"]

        result = self.detector._detect_novelty(FakeExp())
        assert result["detected"]
        assert "quantum_computing" in result["novel_domains"]

    def test_known_domain_not_novel(self):

        class FakeExp:
            domain_tags = ["math"]

        result = self.detector._detect_novelty(FakeExp())
        assert not result["detected"]


class TestOutlierDetector:
    def setup_method(self):
        self.detector = SignalQualityDetector()
        self.detector.domain_baselines["coding"] = {
            "mean": 0.5, "std": 0.1, "n": 50,
        }

    def test_outlier_detected(self):

        class FakeExp:
            domain_tags = ["coding"]

        signal = RewardSignal(task_success=-0.9, coherence=-0.8)
        result = self.detector._detect_outlier(FakeExp(), signal)
        assert result["detected"]

    def test_normal_not_outlier(self):

        class FakeExp:
            domain_tags = ["coding"]

        signal = RewardSignal(task_success=0.5, coherence=0.5)
        result = self.detector._detect_outlier(FakeExp(), signal)
        assert not result["detected"]

    def test_no_baseline_not_outlier(self):

        class FakeExp:
            domain_tags = ["unknown_domain"]

        signal = RewardSignal(task_success=0.9)
        result = self.detector._detect_outlier(FakeExp(), signal)
        assert not result["detected"]


# ---------------------------------------------------------------------------
# Full evaluate() integration
# ---------------------------------------------------------------------------


class TestEvaluateIntegration:
    """End-to-end tests through the evaluate() method."""

    def setup_method(self):
        self.detector = SignalQualityDetector()

    def test_clean_signal_trains(self):

        class FakeExp:
            experience_id = "exp_1"
            domain_tags = ["general"]
            output = "Hello world"

        signal = RewardSignal(
            task_success=0.8, coherence=0.7, user_satisfaction=0.6,
            groundedness=0.6, novelty_value=0.3,
        )
        result = self.detector.evaluate(FakeExp(), signal)
        assert isinstance(result, DetectionResult)
        assert result.verdict == SignalVerdict.TRAIN
        assert DetectionFlag.CLEAN in result.flags

    def test_conflicting_signal_gets_flagged(self):

        class FakeExp:
            experience_id = "exp_2"
            domain_tags = ["general"]
            output = "Some output"

        signal = RewardSignal(user_satisfaction=0.9, groundedness=-0.8)
        result = self.detector.evaluate(FakeExp(), signal)
        assert DetectionFlag.CONFLICTING_DIMENSIONS in result.flags

    def test_sparse_signal_flagged(self):

        class FakeExp:
            experience_id = "exp_3"
            domain_tags = ["general"]
            output = "Output"

        signal = RewardSignal(novelty_value=0.5)
        result = self.detector.evaluate(FakeExp(), signal)
        assert DetectionFlag.SPARSE_SIGNAL in result.flags

    def test_result_has_all_fields(self):

        class FakeExp:
            experience_id = "exp_4"
            domain_tags = ["general"]
            output = "Output text"

        signal = RewardSignal(task_success=0.5, coherence=0.5)
        result = self.detector.evaluate(FakeExp(), signal)
        assert result.experience_id == "exp_4"
        assert result.timestamp > 0
        assert isinstance(result.flags, list)
        assert isinstance(result.confidence, float)
