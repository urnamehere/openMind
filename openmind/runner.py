"""Continuous runner -- always-on hybrid loop for openMind.

The runner operates as a living system that is always doing *something*:
exploring when idle, responding when prompted, consolidating when enough
experience has accumulated, and reflecting on a longer cycle. The
scheduling is adaptive -- the system tunes its own learning intervals
based on how productive each activity is.

Architecture:
- Main thread handles user I/O (stdin/stdout or pluggable transport)
- Background thread runs the autonomous loop
- User input preempts autonomous activity
- AdaptiveScheduler self-tunes all timing

State machine::

    ┌──────────┐   user input   ┌───────────┐
    │ EXPLORING ├───────────────►│ CHATTING   │
    └─────┬────┘                └─────┬─────┘
          │                           │
          │ timer                     │ done
          ▼                           ▼
    ┌──────────────┐           ┌──────────┐
    │ CONSOLIDATING│◄──────────┤ EXPLORING │
    └──────┬───────┘           └──────────┘
           │
           │ timer
           ▼
    ┌──────────┐
    │ REFLECTING│
    └──────────┘
"""

from __future__ import annotations

import json
import logging
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openmind.wrapper import ContinualWrapper

logger = logging.getLogger(__name__)


# ======================================================================
# Adaptive Scheduler
# ======================================================================


class AdaptiveScheduler:
    """Self-tuning scheduler that adjusts cycle intervals based on learning efficiency.

    The scheduler tracks how productive each activity (consolidation,
    reflection, exploration) has been and adjusts intervals accordingly:

    - High learning efficiency → consolidate more often
    - Many accumulated experiences → consolidate sooner
    - High curiosity scores → explore more often
    - Plateauing learning → explore more, consolidate less
    - Stale knowledge → reflect sooner

    All intervals have min/max bounds to prevent runaway behavior.
    """

    def __init__(
        self,
        base_consolidation_interval: float = 300.0,   # 5 minutes
        base_reflection_interval: float = 1800.0,      # 30 minutes
        base_exploration_interval: float = 120.0,       # 2 minutes
        min_consolidation_interval: float = 60.0,       # 1 minute floor
        max_consolidation_interval: float = 3600.0,     # 1 hour ceiling
        min_exploration_interval: float = 30.0,          # 30 second floor
        max_exploration_interval: float = 600.0,         # 10 minute ceiling
    ) -> None:
        # Current intervals (these adapt)
        self.consolidation_interval = base_consolidation_interval
        self.reflection_interval = base_reflection_interval
        self.exploration_interval = base_exploration_interval

        # Bounds
        self._min_consolidation = min_consolidation_interval
        self._max_consolidation = max_consolidation_interval
        self._min_exploration = min_exploration_interval
        self._max_exploration = max_exploration_interval

        # Timestamps of last activity
        self.last_consolidation = time.time()
        self.last_reflection = time.time()
        self.last_exploration = time.time()
        self.last_interaction = time.time()

        # Efficiency tracking
        self._consolidation_yields: List[float] = []  # knowledge gained per cycle
        self._exploration_yields: List[float] = []      # reward per exploration
        self._experience_rates: List[float] = []         # experiences per minute

        # Adaptation state
        self._adaptation_cycle = 0

    def should_consolidate(self, experience_count: int, ambiguity_count: int) -> bool:
        """Should we consolidate now?"""
        elapsed = time.time() - self.last_consolidation

        # Immediate triggers (override timer)
        if experience_count > 50:  # buffer getting full
            return True
        if ambiguity_count > 20:   # too many unresolved
            return True

        # Experience-pressure adjustment: more experiences = consolidate sooner
        pressure = min(experience_count / 30.0, 2.0)  # up to 2x faster
        adjusted_interval = self.consolidation_interval / max(pressure, 0.5)

        return elapsed >= adjusted_interval

    def should_reflect(self, knowledge_count: int) -> bool:
        """Should we reflect now?"""
        elapsed = time.time() - self.last_reflection

        # Need some knowledge to review
        if knowledge_count < 1:
            return False

        return elapsed >= self.reflection_interval

    def should_explore(self, max_curiosity: float) -> bool:
        """Should we explore now?"""
        elapsed = time.time() - self.last_exploration

        # High curiosity = explore sooner
        curiosity_factor = max(0.3, 1.0 - max_curiosity)
        adjusted_interval = self.exploration_interval * curiosity_factor

        return elapsed >= adjusted_interval

    def record_consolidation(self, stats: Dict[str, Any]) -> None:
        """Record consolidation results and adapt interval."""
        self.last_consolidation = time.time()

        # Measure yield: how much knowledge was gained
        consolidation = stats.get("consolidation", {})
        distilled = consolidation.get("distilled", 0)
        promoted = 0
        if isinstance(consolidation.get("promotion"), dict):
            promoted = consolidation["promotion"].get("promoted", 0)
        knowledge_yield = distilled + promoted

        self._consolidation_yields.append(knowledge_yield)
        self._adapt_consolidation_interval()
        self._adaptation_cycle += 1

    def record_reflection(self, stats: Dict[str, Any]) -> None:
        """Record reflection results and adapt interval."""
        self.last_reflection = time.time()

    def record_exploration(self, stats: Dict[str, Any]) -> None:
        """Record exploration results and adapt interval."""
        self.last_exploration = time.time()

        explorations = stats.get("explorations", [])
        if isinstance(explorations, list):
            rewards = [e.get("reward", 0) for e in explorations if isinstance(e, dict)]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0
            self._exploration_yields.append(avg_reward)

        self._adapt_exploration_interval()

    def record_interaction(self) -> None:
        """Record a user interaction."""
        now = time.time()
        if self.last_interaction > 0:
            gap = now - self.last_interaction
            if gap < 600:  # only track gaps < 10 min (active usage)
                rate = 60.0 / max(gap, 1.0)  # interactions per minute
                self._experience_rates.append(rate)
                self._experience_rates = self._experience_rates[-20:]
        self.last_interaction = now

    def get_status(self) -> Dict[str, Any]:
        """Return current scheduler state."""
        now = time.time()
        return {
            "consolidation_interval": f"{self.consolidation_interval:.0f}s",
            "reflection_interval": f"{self.reflection_interval:.0f}s",
            "exploration_interval": f"{self.exploration_interval:.0f}s",
            "time_since_consolidation": f"{now - self.last_consolidation:.0f}s",
            "time_since_reflection": f"{now - self.last_reflection:.0f}s",
            "time_since_exploration": f"{now - self.last_exploration:.0f}s",
            "adaptation_cycles": self._adaptation_cycle,
            "avg_consolidation_yield": (
                f"{sum(self._consolidation_yields[-5:]) / max(len(self._consolidation_yields[-5:]), 1):.1f}"
            ),
            "avg_exploration_reward": (
                f"{sum(self._exploration_yields[-5:]) / max(len(self._exploration_yields[-5:]), 1):.2f}"
            ),
        }

    # === Adaptation logic ===

    def _adapt_consolidation_interval(self) -> None:
        """Tune consolidation frequency based on yield."""
        recent = self._consolidation_yields[-5:]
        if len(recent) < 2:
            return

        avg_yield = sum(recent) / len(recent)
        trend = recent[-1] - recent[0] if len(recent) >= 2 else 0

        if avg_yield > 1.5:
            # Very productive → consolidate more often (reduce interval)
            self.consolidation_interval *= 0.85
        elif avg_yield > 0.5:
            # Moderately productive → slight decrease
            self.consolidation_interval *= 0.95
        elif avg_yield < 0.1 and trend <= 0:
            # Not productive and not improving → consolidate less often
            self.consolidation_interval *= 1.15
        # else: keep current interval

        self.consolidation_interval = max(
            self._min_consolidation,
            min(self._max_consolidation, self.consolidation_interval),
        )

        logger.debug(
            "Adapted consolidation interval to %.0fs (avg yield: %.1f)",
            self.consolidation_interval,
            avg_yield,
        )

    def _adapt_exploration_interval(self) -> None:
        """Tune exploration frequency based on reward yield."""
        recent = self._exploration_yields[-5:]
        if len(recent) < 2:
            return

        avg_reward = sum(recent) / len(recent)

        if avg_reward > 0.5:
            # Explorations are valuable → explore more often
            self.exploration_interval *= 0.85
        elif avg_reward < 0.2:
            # Explorations aren't producing much → slow down
            self.exploration_interval *= 1.2

        self.exploration_interval = max(
            self._min_exploration,
            min(self._max_exploration, self.exploration_interval),
        )


# ======================================================================
# Runner State
# ======================================================================


class RunnerState(Enum):
    """What the system is currently doing."""
    IDLE = "idle"
    CHATTING = "chatting"
    EXPLORING = "exploring"
    CONSOLIDATING = "consolidating"
    REFLECTING = "reflecting"
    SHUTTING_DOWN = "shutting_down"


# ======================================================================
# OpenMind Runner
# ======================================================================


class OpenMindRunner:
    """Always-on hybrid runner for openMind.

    Combines interactive chat with autonomous learning. The system is
    always doing something: responding to users, exploring topics it's
    curious about, consolidating knowledge, or reviewing beliefs.

    The scheduler adapts all timing autonomously based on learning
    efficiency -- if consolidation is producing a lot of knowledge,
    it happens more often; if exploration isn't yielding much, it
    slows down.

    Parameters:
        mind: A configured ContinualWrapper instance.
        scheduler: Optional custom scheduler (uses AdaptiveScheduler by default).
        input_fn: Callable that returns user input (default: stdin).
        output_fn: Callable to display output (default: stdout).
        idle_think_enabled: Whether the system explores when idle.
    """

    def __init__(
        self,
        mind: ContinualWrapper,
        scheduler: Optional[AdaptiveScheduler] = None,
        input_fn: Optional[Callable[[], Optional[str]]] = None,
        output_fn: Optional[Callable[[str], None]] = None,
        idle_think_enabled: bool = True,
    ) -> None:
        self.mind = mind
        self.scheduler = scheduler or AdaptiveScheduler()
        self._input_fn = input_fn
        self._output_fn = output_fn or self._default_output
        self._idle_think_enabled = idle_think_enabled

        self._state = RunnerState.IDLE
        self._running = False
        self._input_queue: queue.Queue = queue.Queue()
        self._response_queue: queue.Queue = queue.Queue()

        self._stats = {
            "started_at": 0.0,
            "total_chats": 0,
            "total_consolidations": 0,
            "total_reflections": 0,
            "total_explorations": 0,
        }

        # Thread management
        self._auto_thread: Optional[threading.Thread] = None
        self._input_thread: Optional[threading.Thread] = None

    def run(self) -> None:
        """Start the continuous runner (blocking).

        This is the main entry point. It starts the autonomous background
        loop and the interactive input loop, then blocks until shutdown.
        """
        self._running = True
        self._stats["started_at"] = time.time()

        self._output_fn(self._startup_banner())

        # Start the autonomous background loop
        self._auto_thread = threading.Thread(
            target=self._autonomous_loop, daemon=True, name="openmind-auto"
        )
        self._auto_thread.start()

        # Run the interactive loop on the main thread
        try:
            self._interactive_loop()
        except KeyboardInterrupt:
            self._output_fn("\n[openMind] Shutting down gracefully...")
        finally:
            self.shutdown()

    def submit_input(self, user_input: str) -> str:
        """Submit input programmatically (for API/server integration).

        Thread-safe. Can be called from any thread.

        Returns:
            The model's response.
        """
        self._input_queue.put(user_input)
        response = self._response_queue.get(timeout=120)
        return response

    def shutdown(self) -> None:
        """Stop the runner gracefully."""
        self._state = RunnerState.SHUTTING_DOWN
        self._running = False
        self._output_fn("[openMind] Running final consolidation...")

        try:
            stats = self.mind.consolidate()
            logger.info("Shutdown consolidation: %s", stats)
        except Exception as e:
            logger.error("Shutdown consolidation failed: %s", e)

        self._output_fn("[openMind] Goodbye. All knowledge has been saved.")

    @property
    def state(self) -> RunnerState:
        return self._state

    @property
    def uptime(self) -> float:
        """Seconds since the runner started."""
        if self._stats["started_at"] == 0:
            return 0.0
        return time.time() - self._stats["started_at"]

    def get_runner_status(self) -> Dict[str, Any]:
        """Full status including runner, scheduler, and mind state."""
        return {
            "state": self._state.value,
            "uptime_hours": f"{self.uptime / 3600:.1f}",
            "stats": dict(self._stats),
            "scheduler": self.scheduler.get_status(),
            "mind": self.mind.get_status(),
        }

    # === Interactive loop (main thread) ===

    def _interactive_loop(self) -> None:
        """Handle user input on the main thread."""
        input_fn = self._input_fn or self._default_input

        while self._running:
            try:
                user_input = input_fn()
                if user_input is None:
                    time.sleep(0.1)
                    continue

                user_input = user_input.strip()
                if not user_input:
                    continue

                # Handle meta-commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Process chat
                self._handle_chat(user_input)

            except EOFError:
                break
            except KeyboardInterrupt:
                break

    def _handle_chat(self, user_input: str) -> None:
        """Process a user message."""
        prev_state = self._state
        self._state = RunnerState.CHATTING
        self.scheduler.record_interaction()

        try:
            response = self.mind.chat(user_input)
            self._stats["total_chats"] += 1

            self._output_fn(f"\n{response}\n")

            # Also push to response queue for programmatic access
            try:
                self._response_queue.put_nowait(response)
            except queue.Full:
                pass

        except Exception as e:
            error_msg = f"[openMind] Error: {e}"
            self._output_fn(error_msg)
            logger.error("Chat failed: %s", e)
        finally:
            self._state = RunnerState.IDLE

    def _handle_command(self, command: str) -> None:
        """Handle slash commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd == "/status":
            status = self.get_runner_status()
            self._output_fn(f"\n{json.dumps(status, indent=2, default=str)}\n")

        elif cmd == "/curious":
            report = self.mind.get_curious_about()
            self._output_fn(f"\n{json.dumps(report, indent=2, default=str)}\n")

        elif cmd == "/explore":
            self._output_fn("[openMind] Exploring now...")
            self._state = RunnerState.EXPLORING
            stats = self.mind.explore()
            self._stats["total_explorations"] += 1
            self.scheduler.record_exploration(stats)
            explorations = stats.get("explorations", [])
            if isinstance(explorations, list) and explorations:
                for e in explorations:
                    if isinstance(e, dict):
                        self._output_fn(
                            f"  [{e.get('mode', '?')}] {e.get('domain', '?')}: "
                            f"{e.get('insight') or e.get('response', '')[:100]}"
                        )
            else:
                self._output_fn("  No explorations this cycle.")
            self._state = RunnerState.IDLE

        elif cmd == "/consolidate":
            self._output_fn("[openMind] Consolidating knowledge...")
            self._run_consolidation()

        elif cmd == "/reflect":
            self._output_fn("[openMind] Reflecting on knowledge...")
            self._run_reflection()

        elif cmd == "/schedule":
            sched = self.scheduler.get_status()
            self._output_fn(f"\n{json.dumps(sched, indent=2)}\n")

        elif cmd == "/quit" or cmd == "/exit":
            self._running = False

        elif cmd == "/help":
            self._output_fn(self._help_text())

        else:
            self._output_fn(f"[openMind] Unknown command: {cmd}. Try /help")

    # === Autonomous loop (background thread) ===

    def _autonomous_loop(self) -> None:
        """Background loop that handles consolidation, reflection, and exploration."""
        while self._running:
            try:
                # Check for programmatic input
                try:
                    user_input = self._input_queue.get_nowait()
                    self._handle_chat(user_input)
                    continue
                except queue.Empty:
                    pass

                # Don't run autonomous tasks while chatting
                if self._state == RunnerState.CHATTING:
                    time.sleep(0.5)
                    continue

                # Priority 1: Consolidation (if needed)
                mind_status = self.mind.get_status()
                exp_count = mind_status.get("experience_buffer_size", 0)
                amb_count = mind_status.get("ambiguity_buffer_size", 0)
                knowledge_count = mind_status.get("knowledge_count", 0)

                if self.scheduler.should_consolidate(exp_count, amb_count):
                    self._run_consolidation()
                    continue

                # Priority 2: Reflection (less frequent)
                if self.scheduler.should_reflect(knowledge_count):
                    self._run_reflection()
                    continue

                # Priority 3: Exploration (when idle and curious)
                if self._idle_think_enabled:
                    curiosity = mind_status.get("curiosity", {})
                    most_curious = curiosity.get("most_curious", [])
                    max_curiosity = 0.0
                    if most_curious:
                        try:
                            max_curiosity = float(most_curious[0].get("score", 0))
                        except (ValueError, IndexError):
                            pass

                    if self.scheduler.should_explore(max_curiosity):
                        self._run_exploration()
                        continue

                # Nothing to do -- brief sleep
                time.sleep(2.0)

            except Exception as e:
                logger.error("Autonomous loop error: %s", e, exc_info=True)
                time.sleep(5.0)

    def _run_consolidation(self) -> None:
        """Execute a consolidation cycle."""
        self._state = RunnerState.CONSOLIDATING
        self._output_fn(
            "[openMind] 💤 Consolidating knowledge... "
            f"(interval: {self.scheduler.consolidation_interval:.0f}s)"
        )

        try:
            stats = self.mind.consolidate()
            self._stats["total_consolidations"] += 1
            self.scheduler.record_consolidation(stats)

            # Report what happened
            consolidation = stats.get("consolidation", {})
            distilled = consolidation.get("distilled", 0)
            if distilled:
                self._output_fn(
                    f"[openMind] Consolidated: {distilled} new knowledge entries"
                )
            ambiguity = stats.get("ambiguity_resolution", {})
            resolved = ambiguity.get("resolved", 0)
            if resolved:
                self._output_fn(
                    f"[openMind] Resolved {resolved} ambiguous experiences"
                )

        except Exception as e:
            logger.error("Consolidation failed: %s", e)
        finally:
            self._state = RunnerState.IDLE

    def _run_reflection(self) -> None:
        """Execute a reflection cycle."""
        self._state = RunnerState.REFLECTING
        self._output_fn(
            "[openMind] 🔍 Reflecting on accumulated knowledge..."
        )

        try:
            stats = self.mind.reflect()
            self._stats["total_reflections"] += 1
            self.scheduler.record_reflection(stats)

            reflection = stats.get("reflection", {})
            reviewed = reflection.get("reviewed_count", 0)
            if reviewed:
                self._output_fn(
                    f"[openMind] Reviewed {reviewed} knowledge entries"
                )

        except Exception as e:
            logger.error("Reflection failed: %s", e)
        finally:
            self._state = RunnerState.IDLE

    def _run_exploration(self) -> None:
        """Execute an exploration cycle."""
        self._state = RunnerState.EXPLORING

        try:
            stats = self.mind.explore()
            self._stats["total_explorations"] += 1
            self.scheduler.record_exploration(stats)

            explorations = stats.get("explorations", [])
            if isinstance(explorations, list) and explorations:
                for e in explorations:
                    if isinstance(e, dict) and e.get("insight"):
                        self._output_fn(
                            f"[openMind] 💡 Discovered ({e.get('domain', '?')}): "
                            f"{e['insight'][:120]}"
                        )

        except Exception as e:
            logger.error("Exploration failed: %s", e)
        finally:
            self._state = RunnerState.IDLE

    # === I/O helpers ===

    @staticmethod
    def _default_input() -> Optional[str]:
        """Default input: read from stdin."""
        try:
            return input("you> ")
        except EOFError:
            return None

    @staticmethod
    def _default_output(text: str) -> None:
        """Default output: print to stdout."""
        print(text, flush=True)

    def _startup_banner(self) -> str:
        status = self.mind.get_status()
        backend = status.get("backend", "unknown")
        ready = status.get("backend_ready", False)
        knowledge = status.get("knowledge_count", 0)
        curiosity = status.get("curiosity", {})
        top = curiosity.get("most_curious", [])

        lines = [
            "",
            "═══════════════════════════════════════════════",
            "  openMind - Continuous Learning System",
            "═══════════════════════════════════════════════",
            f"  Backend: {backend} ({'ready' if ready else 'not connected'})",
            f"  Knowledge entries: {knowledge}",
        ]

        if top:
            interests = ", ".join(
                f"{t['domain']} ({t['level']})" for t in top[:3]
            )
            lines.append(f"  Curious about: {interests}")

        lines.extend([
            "",
            "  Type a message to chat. The system learns",
            "  continuously in the background.",
            "",
            "  Commands: /status /curious /explore",
            "            /consolidate /reflect /schedule",
            "            /help /quit",
            "═══════════════════════════════════════════════",
            "",
        ])
        return "\n".join(lines)

    @staticmethod
    def _help_text() -> str:
        return "\n".join([
            "",
            "  openMind Commands:",
            "  ─────────────────",
            "  /status       - Full system status",
            "  /curious      - What the system is curious about",
            "  /explore      - Trigger exploration now",
            "  /consolidate  - Trigger knowledge consolidation now",
            "  /reflect      - Trigger belief review now",
            "  /schedule     - Show adaptive scheduler state",
            "  /help         - This help message",
            "  /quit         - Shutdown gracefully",
            "",
            "  Just type normally to chat. The system learns",
            "  from every interaction and explores on its own",
            "  when idle.",
            "",
        ])


# ======================================================================
# Convenience entry point
# ======================================================================


def run(
    base_model: Optional[str] = None,
    config_path: Optional[str] = None,
    data_dir: str = "./openmind_data",
    **kwargs: Any,
) -> None:
    """Start openMind in continuous mode.

    This is the simplest way to run the system::

        from openmind.runner import run
        run()  # Claude API backend, interactive + autonomous

        # Or with a local model:
        run(base_model="mistralai/Mistral-7B-v0.3")

    Parameters:
        base_model: HuggingFace model path (uses local backend if set).
        config_path: Path to a YAML config file.
        data_dir: Where to store persistent data.
        **kwargs: Passed to OpenMindRunner constructor.
    """
    mind = ContinualWrapper(
        base_model=base_model,
        config_path=config_path,
        data_dir=data_dir,
    )
    runner = OpenMindRunner(mind=mind, **kwargs)
    runner.run()
