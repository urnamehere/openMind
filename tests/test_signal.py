"""Tests for openmind.reward.signal.RewardSignal."""

import time

from openmind.reward.signal import RewardSignal


class TestRewardSignalConfidence:
    """Confidence combines coverage and agreement."""

    def test_full_coverage_same_sign(self):
        sig = RewardSignal(
            task_success=0.8,
            coherence=0.7,
            user_satisfaction=0.9,
            groundedness=0.6,
            novelty_value=0.5,
        )
        # All 5 dimensions measured (coverage=1.0), all positive (agreement=1.0)
        assert sig.confidence == 1.0

    def test_partial_coverage(self):
        sig = RewardSignal(task_success=0.8)
        # 1/5 measured -> coverage=0.2, single signal -> agreement=1.0
        expected = 0.2 * 0.6 + 1.0 * 0.4
        assert abs(sig.confidence - expected) < 1e-9

    def test_no_dimensions_measured(self):
        sig = RewardSignal()
        # coverage=0, agreement=1.0 (vacuously)
        expected = 0.0 * 0.6 + 1.0 * 0.4
        assert abs(sig.confidence - expected) < 1e-9

    def test_mixed_signs_lower_agreement(self):
        sig = RewardSignal(task_success=0.9, coherence=-0.8)
        # 2/5 coverage, 1 pos + 1 neg -> agreement = 1/2 = 0.5
        expected = (2 / 5) * 0.6 + 0.5 * 0.4
        assert abs(sig.confidence - expected) < 1e-9


class TestRewardSignalAggregate:
    """Weighted aggregate of measured dimensions."""

    def test_single_dimension(self):
        sig = RewardSignal(task_success=0.6)
        # Only task_success measured; re-normalised weight = 1.0
        assert abs(sig.aggregate - 0.6) < 1e-9

    def test_all_ones(self):
        sig = RewardSignal(
            task_success=1.0,
            coherence=1.0,
            user_satisfaction=1.0,
            groundedness=1.0,
            novelty_value=1.0,
        )
        assert abs(sig.aggregate - 1.0) < 1e-9

    def test_all_negative_ones(self):
        sig = RewardSignal(
            task_success=-1.0,
            coherence=-1.0,
            user_satisfaction=-1.0,
            groundedness=-1.0,
            novelty_value=-1.0,
        )
        assert abs(sig.aggregate - (-1.0)) < 1e-9

    def test_empty_returns_zero(self):
        sig = RewardSignal()
        assert sig.aggregate == 0.0


class TestRewardSignalConflicts:
    """Conflict detection: strong positive and strong negative simultaneously."""

    def test_no_conflict_all_positive(self):
        sig = RewardSignal(task_success=0.9, coherence=0.7)
        assert not sig._has_conflicts()

    def test_conflict_detected(self):
        sig = RewardSignal(task_success=0.8, coherence=-0.7)
        assert sig._has_conflicts()

    def test_mild_disagreement_no_conflict(self):
        # Values not strong enough to trigger (threshold is +/- 0.5)
        sig = RewardSignal(task_success=0.3, coherence=-0.3)
        assert not sig._has_conflicts()


class TestRewardSignalMeasuredValues:
    """_get_measured_values returns only non-None dimensions."""

    def test_all_none(self):
        sig = RewardSignal()
        assert sig._get_measured_values() == {}

    def test_some_measured(self):
        sig = RewardSignal(task_success=0.5, groundedness=0.3)
        measured = sig._get_measured_values()
        assert "task_success" in measured
        assert "groundedness" in measured
        assert len(measured) == 2


class TestRewardSignalClamping:
    """Values outside [-1, 1] are clamped."""

    def test_clamp_above(self):
        sig = RewardSignal(task_success=5.0)
        assert sig.task_success == 1.0

    def test_clamp_below(self):
        sig = RewardSignal(coherence=-3.0)
        assert sig.coherence == -1.0


class TestRewardSignalSerialization:
    """Round-trip through to_dict / from_dict."""

    def test_round_trip(self):
        original = RewardSignal(
            task_success=0.7,
            coherence=0.5,
            signal_sources={"task_success": "self_eval"},
        )
        data = original.to_dict()
        restored = RewardSignal.from_dict(data)
        assert restored.task_success == original.task_success
        assert restored.coherence == original.coherence
        assert restored.signal_sources == original.signal_sources


class TestRewardSignalAmbiguity:
    """is_ambiguous combines low confidence and conflict checks."""

    def test_not_ambiguous_when_confident(self):
        sig = RewardSignal(
            task_success=0.8,
            coherence=0.7,
            user_satisfaction=0.9,
            groundedness=0.6,
            novelty_value=0.5,
        )
        assert not sig.is_ambiguous

    def test_ambiguous_with_conflicts(self):
        sig = RewardSignal(
            task_success=0.9,
            coherence=-0.8,
            user_satisfaction=0.7,
            groundedness=-0.6,
            novelty_value=0.5,
        )
        assert sig.is_ambiguous
