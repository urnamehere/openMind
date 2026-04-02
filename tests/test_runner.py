"""Tests for the continuous runner and adaptive scheduler."""

import time

import pytest

from openmind.runner import AdaptiveScheduler, OpenMindRunner, RunnerState


# ---------------------------------------------------------------------------
# AdaptiveScheduler
# ---------------------------------------------------------------------------


class TestAdaptiveScheduler:
    def test_initial_state(self):
        sched = AdaptiveScheduler()
        assert sched.consolidation_interval == 300.0
        assert sched.reflection_interval == 1800.0
        assert sched.exploration_interval == 120.0

    def test_should_consolidate_after_interval(self):
        sched = AdaptiveScheduler(base_consolidation_interval=1.0)
        sched.last_consolidation = time.time() - 2.0
        assert sched.should_consolidate(experience_count=5, ambiguity_count=0)

    def test_should_not_consolidate_too_soon(self):
        sched = AdaptiveScheduler(base_consolidation_interval=300.0)
        sched.last_consolidation = time.time()
        assert not sched.should_consolidate(experience_count=5, ambiguity_count=0)

    def test_consolidate_immediately_on_buffer_pressure(self):
        sched = AdaptiveScheduler(base_consolidation_interval=9999.0)
        sched.last_consolidation = time.time()
        # 51 experiences should trigger immediate consolidation
        assert sched.should_consolidate(experience_count=51, ambiguity_count=0)

    def test_consolidate_immediately_on_ambiguity_pressure(self):
        sched = AdaptiveScheduler(base_consolidation_interval=9999.0)
        sched.last_consolidation = time.time()
        assert sched.should_consolidate(experience_count=0, ambiguity_count=21)

    def test_should_reflect_requires_knowledge(self):
        sched = AdaptiveScheduler(base_reflection_interval=1.0)
        sched.last_reflection = time.time() - 2.0
        assert not sched.should_reflect(knowledge_count=0)
        assert sched.should_reflect(knowledge_count=5)

    def test_should_explore_with_high_curiosity(self):
        sched = AdaptiveScheduler(base_exploration_interval=10.0)
        sched.last_exploration = time.time() - 5.0
        # High curiosity (0.9) reduces interval by factor of 0.1
        assert sched.should_explore(max_curiosity=0.9)

    def test_should_not_explore_with_low_curiosity(self):
        sched = AdaptiveScheduler(base_exploration_interval=100.0)
        sched.last_exploration = time.time() - 30.0
        # Low curiosity doesn't compress enough
        assert not sched.should_explore(max_curiosity=0.1)

    def test_adaptation_reduces_interval_on_high_yield(self):
        sched = AdaptiveScheduler(base_consolidation_interval=300.0)
        original = sched.consolidation_interval

        # Simulate several productive consolidations
        for _ in range(5):
            sched.record_consolidation({"consolidation": {"distilled": 3}})

        assert sched.consolidation_interval < original

    def test_adaptation_increases_interval_on_low_yield(self):
        sched = AdaptiveScheduler(base_consolidation_interval=300.0)
        original = sched.consolidation_interval

        # Simulate several unproductive consolidations
        for _ in range(5):
            sched.record_consolidation({"consolidation": {"distilled": 0}})

        assert sched.consolidation_interval > original

    def test_interval_stays_within_bounds(self):
        sched = AdaptiveScheduler(
            base_consolidation_interval=300.0,
            min_consolidation_interval=60.0,
            max_consolidation_interval=3600.0,
        )

        # Push toward minimum with lots of productive cycles
        for _ in range(50):
            sched.record_consolidation({"consolidation": {"distilled": 5}})
        assert sched.consolidation_interval >= 60.0

        # Reset and push toward maximum
        sched.consolidation_interval = 300.0
        for _ in range(50):
            sched.record_consolidation({"consolidation": {"distilled": 0}})
        assert sched.consolidation_interval <= 3600.0

    def test_exploration_adapts_on_reward(self):
        sched = AdaptiveScheduler(base_exploration_interval=120.0)
        original = sched.exploration_interval

        # Productive explorations
        for _ in range(5):
            sched.record_exploration({
                "explorations": [{"reward": 0.8}, {"reward": 0.7}]
            })

        assert sched.exploration_interval < original

    def test_get_status(self):
        sched = AdaptiveScheduler()
        status = sched.get_status()
        assert "consolidation_interval" in status
        assert "adaptation_cycles" in status

    def test_record_interaction(self):
        sched = AdaptiveScheduler()
        sched.record_interaction()
        assert sched.last_interaction > 0


# ---------------------------------------------------------------------------
# RunnerState
# ---------------------------------------------------------------------------


class TestRunnerState:
    def test_all_states(self):
        expected = {"idle", "chatting", "exploring", "consolidating",
                    "reflecting", "shutting_down"}
        actual = {s.value for s in RunnerState}
        assert actual == expected


# ---------------------------------------------------------------------------
# OpenMindRunner (unit tests, no actual running)
# ---------------------------------------------------------------------------


class TestOpenMindRunnerInit:
    """Test runner construction without starting the loop."""

    def test_creates_with_defaults(self):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()

        # Create a minimal wrapper (Claude backend, no API key = not ready)
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from openmind import ContinualWrapper
            mind = ContinualWrapper(data_dir=tmpdir)
            runner = OpenMindRunner(mind=mind)
            assert runner.state == RunnerState.IDLE
            assert runner.uptime == 0.0
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_get_runner_status(self):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from openmind import ContinualWrapper
            mind = ContinualWrapper(data_dir=tmpdir)
            runner = OpenMindRunner(mind=mind)
            status = runner.get_runner_status()
            assert "state" in status
            assert "scheduler" in status
            assert "mind" in status
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_idle_think_can_be_disabled(self):
        import tempfile, os
        tmpdir = tempfile.mkdtemp()
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            from openmind import ContinualWrapper
            mind = ContinualWrapper(data_dir=tmpdir)
            runner = OpenMindRunner(mind=mind, idle_think_enabled=False)
            assert runner._idle_think_enabled is False
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key
