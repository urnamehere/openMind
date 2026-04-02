"""Temporal reward tracking for the openMind continuous learning system.

Reward signals are not static: user corrections arrive late, execution results
trickle in after the fact, and the system's own understanding of quality
evolves over training cycles.  This module tracks how reward signals change
over time, detects delayed feedback, and flags contradictions when already-
trained-on signals are later revised.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .signal import RewardSignal

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------

@dataclass
class TemporalRewardEvent:
    """A time-aware wrapper around a reward signal for a single experience.

    Tracks the initial signal, any subsequent revisions, and whether the
    signal has already been consumed by a training cycle.

    Attributes:
        experience_id:        Unique identifier linking to the source experience.
        created_at:           ISO-8601 UTC timestamp of creation.
        initial_signal:       The first reward signal captured, stored as a dict
                              (output of ``RewardSignal.to_dict()``).
        revisions:            Ordered list of ``{"timestamp": ..., "signal": ...,
                              "reason": ...}`` dicts recording every revision.
        trained_on:           Whether this signal has already been used in a
                              training cycle.
        contradiction_detected: True if a revision contradicts a signal that
                              was already trained on.
    """

    experience_id: str
    created_at: str
    initial_signal: Dict[str, Any]
    revisions: List[Dict[str, Any]] = field(default_factory=list)
    trained_on: bool = False
    contradiction_detected: bool = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_signal(self) -> Dict[str, Any]:
        """Return the most recent signal dict (latest revision or initial)."""
        if self.revisions:
            return self.revisions[-1]["signal"]
        return self.initial_signal

    @property
    def stability(self) -> float:
        """Measure how stable this signal has been over time.

        Returns a float in [0.0, 1.0] where 1.0 means perfectly stable
        (no revisions or all revisions agree) and 0.0 means maximally
        unstable.
        """
        if not self.revisions:
            return 1.0

        # Compare the aggregate of each revision to the initial
        initial_agg = self.initial_signal.get("aggregate", 0.0)
        if initial_agg is None:
            initial_agg = 0.0

        drifts: List[float] = []
        for rev in self.revisions:
            rev_agg = rev["signal"].get("aggregate", 0.0)
            if rev_agg is None:
                rev_agg = 0.0
            drifts.append(abs(rev_agg - initial_agg))

        if not drifts:
            return 1.0

        # Average drift, mapped from [0, 2] (max possible drift) to [0, 1]
        avg_drift = sum(drifts) / len(drifts)
        stability = max(0.0, 1.0 - avg_drift / 2.0)

        # Penalise for sheer number of revisions (instability indicator)
        revision_penalty = min(len(self.revisions) * 0.05, 0.3)
        return max(0.0, stability - revision_penalty)

    @property
    def time_since_last_revision(self) -> Optional[float]:
        """Seconds since the most recent revision, or None if never revised."""
        if not self.revisions:
            return None
        last_ts = self.revisions[-1].get("timestamp")
        if last_ts is None:
            return None
        try:
            last_dt = datetime.fromisoformat(last_ts)
            return (datetime.now(timezone.utc) - last_dt).total_seconds()
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalRewardEvent:
        return cls(**data)


# -----------------------------------------------------------------------
# Tracker
# -----------------------------------------------------------------------

class TemporalRewardTracker:
    """Manages reward signals over time with JSONL persistence.

    Responsibilities:
    - Register new reward signals for experiences.
    - Revise existing signals when new evidence arrives.
    - Detect delayed feedback that changes the reward picture.
    - Flag contradictions when trained-on signals are revised.
    - Export training-ready signals filtered by stability and recency.
    - Compute per-domain reliability scores.

    Parameters:
        storage_path: Path to the backing JSONL file.
    """

    # How much an aggregate must shift to count as a contradiction
    CONTRADICTION_THRESHOLD: float = 0.5

    # Minimum stability to be considered training-ready
    TRAINING_STABILITY_THRESHOLD: float = 0.6

    def __init__(self, storage_path: str | Path) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, experience_id: str, signal: RewardSignal) -> TemporalRewardEvent:
        """Register a new reward signal for an experience.

        Args:
            experience_id: Unique identifier of the experience.
            signal: The initial :class:`RewardSignal`.

        Returns:
            The newly created :class:`TemporalRewardEvent`.
        """
        now = datetime.now(timezone.utc).isoformat()
        event = TemporalRewardEvent(
            experience_id=experience_id,
            created_at=now,
            initial_signal=signal.to_dict(),
        )
        self._append(event)
        logger.debug("Registered temporal reward for experience %s", experience_id)
        return event

    def revise(
        self,
        experience_id: str,
        new_signal: RewardSignal,
        reason: str = "delayed_feedback",
    ) -> Optional[TemporalRewardEvent]:
        """Revise the reward signal for an existing experience.

        If the experience was already trained on and the revision is a
        significant contradiction, ``contradiction_detected`` is set and a
        correction record is generated.

        Args:
            experience_id: The experience to revise.
            new_signal: The updated :class:`RewardSignal`.
            reason: Human-readable reason for the revision.

        Returns:
            The updated event, or None if the experience_id was not found.
        """
        events = self._load_all()
        target: Optional[TemporalRewardEvent] = None
        target_idx: Optional[int] = None

        for idx, ev in enumerate(events):
            if ev.experience_id == experience_id:
                target = ev
                target_idx = idx
                break

        if target is None:
            logger.warning("revise() called for unknown experience %s", experience_id)
            return None

        now = datetime.now(timezone.utc).isoformat()
        revision = {
            "timestamp": now,
            "signal": new_signal.to_dict(),
            "reason": reason,
        }
        target.revisions.append(revision)

        # Check for contradiction with trained-on signal
        if target.trained_on:
            old_agg = target.initial_signal.get("aggregate", 0.0) or 0.0
            new_agg = new_signal.aggregate
            if abs(new_agg - old_agg) >= self.CONTRADICTION_THRESHOLD:
                target.contradiction_detected = True
                correction = self._generate_correction(target, new_signal)
                logger.warning(
                    "Contradiction detected for trained experience %s: "
                    "old_agg=%.3f new_agg=%.3f. Correction: %s",
                    experience_id, old_agg, new_agg, correction,
                )

        events[target_idx] = target  # type: ignore[index]
        self._rewrite(events)
        return target

    def detect_delayed_feedback(
        self,
        experience_id: str,
        user_response: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Check whether delayed evidence changes the reward for an experience.

        This is called when new information arrives after the initial signal
        was recorded (e.g. user comes back hours later with feedback, or a
        long-running code execution finally completes).

        Args:
            experience_id: The experience to check.
            user_response: Late-arriving user feedback text.
            execution_result: Late-arriving execution outcome dict.

        Returns:
            A dict summarising what changed, or None if no update was needed.
        """
        events = self._load_all()
        target: Optional[TemporalRewardEvent] = None
        for ev in events:
            if ev.experience_id == experience_id:
                target = ev
                break

        if target is None:
            logger.warning(
                "detect_delayed_feedback() called for unknown experience %s",
                experience_id,
            )
            return None

        current = target.current_signal
        updates: Dict[str, Any] = {}

        # Check explicit feedback in user_response
        if user_response is not None:
            satisfaction = self._extract_satisfaction_from_text(user_response)
            if satisfaction is not None:
                old_sat = current.get("user_satisfaction")
                if old_sat is None or abs(satisfaction - (old_sat or 0.0)) > 0.2:
                    updates["user_satisfaction"] = satisfaction
                    updates["user_satisfaction_source"] = "delayed_explicit_feedback"

        # Check execution result
        if execution_result is not None:
            exec_score = self._score_execution(execution_result)
            if exec_score is not None:
                old_task = current.get("task_success")
                if old_task is None or abs(exec_score - (old_task or 0.0)) > 0.2:
                    updates["task_success"] = exec_score
                    updates["task_success_source"] = "delayed_execution_result"

        if not updates:
            return None

        # Build a new signal from current + updates
        dim_names = RewardSignal._dimension_names()
        dim_kwargs: Dict[str, Optional[float]] = {}
        for dim in dim_names:
            if dim in updates:
                dim_kwargs[dim] = updates[dim]
            else:
                dim_kwargs[dim] = current.get(dim)

        sources = dict(current.get("signal_sources", {}))
        for dim in dim_names:
            source_key = f"{dim}_source"
            if source_key in updates:
                sources[dim] = updates[source_key]

        new_signal = RewardSignal(**dim_kwargs, signal_sources=sources)
        self.revise(experience_id, new_signal, reason="delayed_feedback")

        return {
            "experience_id": experience_id,
            "updates": updates,
            "new_aggregate": new_signal.aggregate,
            "old_aggregate": current.get("aggregate", 0.0),
        }

    def get_training_ready(
        self,
        min_stability: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
    ) -> List[TemporalRewardEvent]:
        """Return events suitable for inclusion in a training cycle.

        Filters for events that:
        - Have not yet been trained on.
        - Meet the minimum stability threshold.
        - Are not flagged as contradictions.
        - Optionally fall within a recency window.

        Args:
            min_stability: Override the default stability threshold.
            max_age_seconds: If set, only include events created within this
                many seconds of now.

        Returns:
            List of training-ready events.
        """
        threshold = min_stability if min_stability is not None else self.TRAINING_STABILITY_THRESHOLD
        now = datetime.now(timezone.utc)
        results: List[TemporalRewardEvent] = []

        for ev in self._load_all():
            if ev.trained_on:
                continue
            if ev.contradiction_detected:
                continue
            if ev.stability < threshold:
                continue
            if max_age_seconds is not None:
                try:
                    created = datetime.fromisoformat(ev.created_at)
                    age = (now - created).total_seconds()
                    if age > max_age_seconds:
                        continue
                except (ValueError, TypeError):
                    continue
            results.append(ev)

        logger.info(
            "Training-ready events: %d (stability >= %.2f)", len(results), threshold
        )
        return results

    def mark_trained(self, experience_ids: List[str]) -> int:
        """Mark a batch of events as trained-on.

        Args:
            experience_ids: IDs to mark.

        Returns:
            Number of events actually updated.
        """
        id_set = set(experience_ids)
        events = self._load_all()
        count = 0
        for ev in events:
            if ev.experience_id in id_set and not ev.trained_on:
                ev.trained_on = True
                count += 1
        if count > 0:
            self._rewrite(events)
        return count

    def compute_domain_reliability(
        self,
        domain_tag: str,
        experience_domain_map: Dict[str, List[str]],
    ) -> Dict[str, float]:
        """Compute reliability metrics for a specific domain.

        Analyses all tracked events whose experience_id maps to the given
        domain tag (via the provided mapping) and returns aggregate stats.

        Args:
            domain_tag: The domain to evaluate.
            experience_domain_map: Maps experience_id -> list of domain tags.

        Returns:
            Dict with keys: avg_stability, avg_aggregate, contradiction_rate,
            revision_rate, event_count.
        """
        events = self._load_all()
        domain_events = [
            ev for ev in events
            if domain_tag in experience_domain_map.get(ev.experience_id, [])
        ]

        if not domain_events:
            return {
                "avg_stability": 0.0,
                "avg_aggregate": 0.0,
                "contradiction_rate": 0.0,
                "revision_rate": 0.0,
                "event_count": 0,
            }

        n = len(domain_events)
        stabilities = [ev.stability for ev in domain_events]
        aggregates = [
            ev.current_signal.get("aggregate", 0.0) or 0.0
            for ev in domain_events
        ]
        contradictions = sum(1 for ev in domain_events if ev.contradiction_detected)
        revised = sum(1 for ev in domain_events if ev.revisions)

        return {
            "avg_stability": sum(stabilities) / n,
            "avg_aggregate": sum(aggregates) / n,
            "contradiction_rate": contradictions / n,
            "revision_rate": revised / n,
            "event_count": n,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_correction(
        self,
        event: TemporalRewardEvent,
        new_signal: RewardSignal,
    ) -> Dict[str, Any]:
        """Generate a correction record when a trained-on signal is contradicted.

        This record can be used by the training system to create a
        counter-example or to increase the priority of the corrected signal
        in the next training cycle.

        Returns:
            A dict describing the correction needed.
        """
        old_agg = event.initial_signal.get("aggregate", 0.0) or 0.0
        new_agg = new_signal.aggregate
        direction = "positive" if new_agg > old_agg else "negative"

        # Compute per-dimension deltas
        dim_deltas: Dict[str, float] = {}
        for dim in RewardSignal._dimension_names():
            old_val = event.initial_signal.get(dim)
            new_val = getattr(new_signal, dim)
            if old_val is not None and new_val is not None:
                dim_deltas[dim] = new_val - old_val

        correction = {
            "experience_id": event.experience_id,
            "correction_type": "contradiction_after_training",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_aggregate": old_agg,
            "new_aggregate": new_agg,
            "shift_direction": direction,
            "shift_magnitude": abs(new_agg - old_agg),
            "dimension_deltas": dim_deltas,
            "recommended_training_weight": min(2.0, 1.0 + abs(new_agg - old_agg)),
        }

        logger.info(
            "Generated correction for %s: %s shift of %.3f",
            event.experience_id, direction, abs(new_agg - old_agg),
        )
        return correction

    @staticmethod
    def _extract_satisfaction_from_text(text: str) -> Optional[float]:
        """Simple keyword-based satisfaction extraction for delayed feedback.

        Returns a float in [-1, 1] or None if no signal detected.
        """
        import re

        lower = text.lower().strip()

        positive_strong = [
            r"\bthanks?\b", r"\bthank you\b", r"\bperfect\b",
            r"\bexcellent\b", r"\bgreat\b", r"\bawesome\b",
        ]
        negative_strong = [
            r"\bwrong\b", r"\bincorrect\b", r"\bbad\b",
            r"\bterrible\b", r"\bhallucin", r"\buseless\b",
        ]
        positive_mild = [r"\bgood\b", r"\bhelpful\b", r"\bworks\b", r"\byes\b"]
        negative_mild = [r"\bnot quite\b", r"\bnot really\b", r"\bno\b"]

        for pattern in positive_strong:
            if re.search(pattern, lower):
                return 0.9
        for pattern in negative_strong:
            if re.search(pattern, lower):
                return -0.9
        for pattern in positive_mild:
            if re.search(pattern, lower):
                return 0.5
        for pattern in negative_mild:
            if re.search(pattern, lower):
                return -0.4
        return None

    @staticmethod
    def _score_execution(execution_result: Dict[str, Any]) -> Optional[float]:
        """Score task success from a late-arriving execution result."""
        if "success" not in execution_result:
            return None
        if execution_result["success"]:
            passed = execution_result.get("tests_passed")
            total = execution_result.get("tests_total")
            if passed is not None and total is not None and total > 0:
                ratio = passed / total
                return -0.5 + 1.5 * ratio
            return 0.9
        else:
            stderr = execution_result.get("stderr", "")
            return -0.8 if stderr and len(stderr) > 200 else -0.5

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _append(self, event: TemporalRewardEvent) -> None:
        """Append a single event to the JSONL file."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event.to_dict()) + "\n")

    def _load_all(self) -> List[TemporalRewardEvent]:
        """Read all events from the backing JSONL file."""
        if not self._path.exists():
            return []
        events: List[TemporalRewardEvent] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    events.append(TemporalRewardEvent.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping corrupt temporal event at %s:%d -- %s",
                        self._path, lineno, exc,
                    )
        return events

    def _rewrite(self, events: List[TemporalRewardEvent]) -> None:
        """Overwrite the backing JSONL file with the given events."""
        with open(self._path, "w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event.to_dict()) + "\n")
