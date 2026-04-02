"""Tests for openmind.core.experience.Experience and ExperienceBuffer."""

import os
import tempfile

import pytest

from openmind.core.experience import Experience, ExperienceBuffer


# ---------------------------------------------------------------------------
# Experience dataclass
# ---------------------------------------------------------------------------


class TestExperience:
    """Basic construction and validation of Experience."""

    def test_create_valid(self):
        exp = Experience(
            timestamp="2026-04-02T12:00:00Z",
            input_context="What is 2+2?",
            output="4",
            reward_signal=0.9,
            domain_tags=["math"],
            confidence=0.95,
        )
        assert exp.reward_signal == 0.9
        assert exp.confidence == 0.95
        assert exp.training_weight == 1.0
        assert len(exp.experience_id) > 0

    def test_reward_out_of_range_raises(self):
        with pytest.raises(ValueError, match="reward_signal"):
            Experience(
                timestamp="2026-04-02T12:00:00Z",
                input_context="x",
                output="y",
                reward_signal=1.5,
                domain_tags=[],
                confidence=0.5,
            )

    def test_negative_reward_out_of_range_raises(self):
        with pytest.raises(ValueError, match="reward_signal"):
            Experience(
                timestamp="2026-04-02T12:00:00Z",
                input_context="x",
                output="y",
                reward_signal=-1.1,
                domain_tags=[],
                confidence=0.5,
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            Experience(
                timestamp="2026-04-02T12:00:00Z",
                input_context="x",
                output="y",
                reward_signal=0.5,
                domain_tags=[],
                confidence=2.0,
            )

    def test_negative_training_weight_raises(self):
        with pytest.raises(ValueError, match="training_weight"):
            Experience(
                timestamp="2026-04-02T12:00:00Z",
                input_context="x",
                output="y",
                reward_signal=0.5,
                domain_tags=[],
                confidence=0.5,
                training_weight=-0.1,
            )

    def test_round_trip_dict(self):
        exp = Experience(
            timestamp="2026-04-02T12:00:00Z",
            input_context="prompt",
            output="response",
            reward_signal=0.7,
            domain_tags=["science"],
            confidence=0.8,
        )
        data = exp.to_dict()
        restored = Experience.from_dict(data)
        assert restored.experience_id == exp.experience_id
        assert restored.reward_signal == exp.reward_signal
        assert restored.domain_tags == exp.domain_tags


# ---------------------------------------------------------------------------
# ExperienceBuffer
# ---------------------------------------------------------------------------


def _make_experience(reward: float, tag: str = "test", ts: str = "2026-04-02T12:00:00Z") -> Experience:
    return Experience(
        timestamp=ts,
        input_context="input",
        output="output",
        reward_signal=reward,
        domain_tags=[tag],
        confidence=0.8,
    )


class TestExperienceBuffer:
    """Record, retrieve, and sample from the buffer."""

    def test_record_and_len(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        assert len(buf) == 0

        exp = _make_experience(0.5)
        eid = buf.record(exp)
        assert eid == exp.experience_id
        assert len(buf) == 1

    def test_record_multiple(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        for r in [0.1, 0.5, 0.9]:
            buf.record(_make_experience(r))
        assert len(buf) == 3

    def test_get_training_batch_filters_by_reward(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        for r in [0.1, 0.2, 0.5, 0.8, 0.95]:
            buf.record(_make_experience(r))

        batch = buf.get_training_batch(min_reward=0.5)
        rewards = [e.reward_signal for e in batch]
        assert all(r >= 0.5 for r in rewards)
        assert len(batch) == 3  # 0.5, 0.8, 0.95

    def test_training_batch_sorted_descending(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        for r in [0.3, 0.9, 0.6]:
            buf.record(_make_experience(r))

        batch = buf.get_training_batch(min_reward=0.3)
        rewards = [e.reward_signal for e in batch]
        assert rewards == sorted(rewards, reverse=True)

    def test_get_replay_sample_returns_subset(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        for i in range(20):
            buf.record(_make_experience(0.5))

        sample = buf.get_replay_sample(n=5)
        assert len(sample) == 5

    def test_get_replay_sample_returns_all_when_small(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        for i in range(3):
            buf.record(_make_experience(0.5))

        sample = buf.get_replay_sample(n=10)
        assert len(sample) == 3

    def test_get_recent(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        buf.record(_make_experience(0.5, ts="2026-01-01T00:00:00Z"))
        buf.record(_make_experience(0.6, ts="2026-04-01T00:00:00Z"))
        buf.record(_make_experience(0.7, ts="2026-04-02T00:00:00Z"))

        recent = buf.get_recent(n=2)
        assert len(recent) == 2
        # Most recent first
        assert recent[0].timestamp >= recent[1].timestamp

    def test_empty_buffer_returns_empty_lists(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        assert buf.get_training_batch() == []
        assert buf.get_replay_sample() == []
        assert buf.get_recent() == []

    def test_get_by_domain(self, tmp_path):
        buf = ExperienceBuffer(tmp_path / "exp.jsonl")
        buf.record(_make_experience(0.5, tag="math", ts="2026-04-01T00:00:00Z"))
        buf.record(_make_experience(0.6, tag="science", ts="2026-04-01T00:00:00Z"))
        buf.record(_make_experience(0.7, tag="math", ts="2026-04-01T00:00:00Z"))

        results = buf.get_by_domain(["math"], days_back=30)
        assert all("math" in e.domain_tags for e in results)
        assert len(results) == 2
