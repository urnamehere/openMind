"""Tests for the curiosity-driven exploration system."""

import time

import pytest

from openmind.curiosity.interest import DomainInterest, InterestModel
from openmind.curiosity.explorer import CuriosityEngine, Exploration
from openmind.utils.config import CuriosityConfig


# ---------------------------------------------------------------------------
# DomainInterest
# ---------------------------------------------------------------------------


class TestDomainInterest:
    def test_new_domain_has_high_novelty(self):
        di = DomainInterest(domain="quantum", last_interaction=time.time())
        assert di.novelty_score == 1.0

    def test_curiosity_score_bounded(self):
        di = DomainInterest(domain="test", last_interaction=time.time())
        assert 0.0 <= di.curiosity_score <= 1.0

    def test_high_mastery_reduces_curiosity(self):
        active = DomainInterest(
            domain="test",
            mastery_score=0.95,
            novelty_score=0.0,
            learning_velocity=0.0,
            last_interaction=time.time(),
        )
        low_mastery = DomainInterest(
            domain="test",
            mastery_score=0.1,
            novelty_score=0.5,
            learning_velocity=0.02,
            last_interaction=time.time(),
        )
        assert low_mastery.curiosity_score > active.curiosity_score

    def test_learning_velocity_drives_curiosity(self):
        progressing = DomainInterest(
            domain="test",
            learning_velocity=0.1,
            novelty_score=0.5,
            gap_score=0.3,
            last_interaction=time.time(),
        )
        stalled = DomainInterest(
            domain="test",
            learning_velocity=0.0,
            novelty_score=0.5,
            gap_score=0.3,
            last_interaction=time.time(),
        )
        assert progressing.curiosity_score > stalled.curiosity_score

    def test_interest_levels(self):
        # Dormant: very low curiosity
        dormant = DomainInterest(
            domain="test",
            mastery_score=0.99,
            novelty_score=0.0,
            learning_velocity=-0.01,
            last_interaction=time.time() - 86400 * 30,
        )
        assert dormant.interest_level in ("dormant", "aware")

    def test_round_trip_serialization(self):
        di = DomainInterest(
            domain="coding",
            total_experiences=50,
            mastery_score=0.6,
            connected_domains=["math"],
        )
        data = di.to_dict()
        restored = DomainInterest.from_dict(data)
        assert restored.domain == "coding"
        assert restored.total_experiences == 50
        assert restored.connected_domains == ["math"]

    def test_knowledge_gaps_increase_curiosity(self):
        with_gaps = DomainInterest(
            domain="test",
            gap_score=0.8,
            shelved_count=5,
            novelty_score=0.5,
            last_interaction=time.time(),
        )
        no_gaps = DomainInterest(
            domain="test",
            gap_score=0.0,
            shelved_count=0,
            novelty_score=0.5,
            last_interaction=time.time(),
        )
        assert with_gaps.curiosity_score > no_gaps.curiosity_score


# ---------------------------------------------------------------------------
# InterestModel
# ---------------------------------------------------------------------------


class TestInterestModel:
    def test_update_creates_domain(self):
        model = InterestModel()
        model.update_from_experience(["coding"], reward=0.8)
        assert model.get_domain("coding") is not None
        assert model.get_domain("coding").total_experiences == 1

    def test_positive_reward_increases_mastery(self):
        model = InterestModel()
        model.update_from_experience(["math"], reward=0.9)
        model.update_from_experience(["math"], reward=0.8)
        di = model.get_domain("math")
        assert di.mastery_score > 0.0

    def test_shelved_increases_gap_score(self):
        model = InterestModel()
        model.update_from_experience(["physics"], reward=0.0, was_shelved=True)
        di = model.get_domain("physics")
        assert di.gap_score > 0.0
        assert di.shelved_count == 1

    def test_most_curious_ranking(self):
        model = InterestModel()
        # Create a domain with active learning
        for _ in range(10):
            model.update_from_experience(["active_domain"], reward=0.7)
        # Create a stale domain
        model.update_from_experience(["stale_domain"], reward=0.1)

        ranked = model.get_most_curious(2)
        assert len(ranked) == 2
        assert ranked[0][0] == "active_domain"

    def test_discover_connection(self):
        model = InterestModel()
        model.update_from_experience(["coding"], reward=0.5)
        model.update_from_experience(["math"], reward=0.5)
        model.discover_connection("coding", "math")

        coding = model.get_domain("coding")
        math = model.get_domain("math")
        assert "math" in coding.connected_domains
        assert "coding" in math.connected_domains

    def test_knowledge_reduces_gaps(self):
        model = InterestModel()
        model.update_from_experience(["science"], reward=0.0, was_shelved=True)
        model.update_from_experience(["science"], reward=0.0, was_shelved=True)
        gap_before = model.get_domain("science").gap_score

        model.update_from_knowledge(["science"])
        gap_after = model.get_domain("science").gap_score
        assert gap_after < gap_before

    def test_summary(self):
        model = InterestModel()
        model.update_from_experience(["coding"], reward=0.8)
        summary = model.summary()
        assert summary["total_domains"] == 1
        assert len(summary["most_curious"]) == 1

    def test_empty_summary(self):
        model = InterestModel()
        summary = model.summary()
        assert summary["total_domains"] == 0


# ---------------------------------------------------------------------------
# CuriosityEngine
# ---------------------------------------------------------------------------


class TestCuriosityEngine:
    def test_no_backend_returns_empty(self):
        model = InterestModel()
        engine = CuriosityEngine(interest_model=model, backend=None)
        results = engine.explore()
        assert results == []

    def test_get_curious_about_structure(self):
        model = InterestModel()
        model.update_from_experience(["coding"], reward=0.8)
        model.update_from_experience(["math"], reward=0.6)

        engine = CuriosityEngine(interest_model=model, backend=None)
        report = engine.get_curious_about()

        assert "top_interests" in report
        assert "knowledge_gaps" in report
        assert "making_progress_in" in report
        assert "would_like_to_explore" in report

    def test_exploration_dataclass(self):
        exp = Exploration(
            exploration_id="test1",
            timestamp=time.time(),
            mode="deep_dive",
            domain="coding",
            query="What are common pitfalls?",
            response="Here are some...",
            reward=0.7,
        )
        data = exp.to_dict()
        assert data["mode"] == "deep_dive"
        assert data["domain"] == "coding"

    def test_explore_with_mock_backend(self):

        class MockBackend:
            is_ready = True

            def generate(self, prompt, context=None):
                return "This is a detailed exploration response about the topic with lots of useful information and insights that spans many words to be considered substantive."

            def evaluate_output(self, input_ctx, output):
                return {"task_success": 0.7, "coherence": 0.8}

        model = InterestModel()
        # Build up enough curiosity
        for _ in range(5):
            model.update_from_experience(["coding"], reward=0.7)
            model.update_from_experience(["coding"], reward=0.0, was_shelved=True)

        engine = CuriosityEngine(
            interest_model=model,
            backend=MockBackend(),
            budget_per_cycle=1,
            min_curiosity_threshold=0.1,
        )

        results = engine.explore()
        assert len(results) >= 1
        assert results[0].domain == "coding"
        assert results[0].reward > 0

    def test_budget_respected(self):

        class MockBackend:
            is_ready = True

            def generate(self, prompt, context=None):
                return "Response " * 50

            def evaluate_output(self, input_ctx, output):
                return {"task_success": 0.5, "coherence": 0.5}

        model = InterestModel()
        for domain in ["a", "b", "c", "d", "e"]:
            for _ in range(10):
                model.update_from_experience([domain], reward=0.6)

        engine = CuriosityEngine(
            interest_model=model,
            backend=MockBackend(),
            budget_per_cycle=2,
            min_curiosity_threshold=0.1,
        )

        results = engine.explore()
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# CuriosityConfig
# ---------------------------------------------------------------------------


class TestCuriosityConfig:
    def test_defaults(self):
        cfg = CuriosityConfig()
        assert cfg.enabled is True
        assert cfg.budget_per_cycle == 3
        assert cfg.min_curiosity_threshold == 0.3
