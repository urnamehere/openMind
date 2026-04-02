"""Ambiguity detection, buffering, and resolution for continuous learning.

When the system encounters experiences that produce mixed reward signals,
contradictory outputs, or touch novel domains, those experiences are shelved
in the AmbiguityBuffer rather than immediately used for training.  The
AmbiguityResolver periodically revisits shelved items, attempting to resolve
them through accumulated evidence, weight-change analysis, or expiration.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Data model
# -----------------------------------------------------------------------

@dataclass
class AmbiguousExperience:
    """An experience that the system could not confidently learn from.

    Attributes:
        experience_id: Unique identifier (mirrors the source Experience id).
        timestamp: ISO-8601 UTC timestamp when the ambiguity was detected.
        input_context: The prompt / context presented to the model.
        output: The model's generated response.
        domain_tags: Knowledge-domain tags associated with the experience.
        ambiguity_type: Classification of why the experience is ambiguous.
            One of ``"mixed_signals"``, ``"low_confidence"``,
            ``"novel_domain"``, ``"contradiction"``, or ``"unknown"``.
        ambiguity_score: Scalar in [0.0, 1.0] indicating severity (1.0 = most
            ambiguous).
        partial_signals: Raw reward dimensions and metadata captured at
            detection time.
        review_count: How many resolution review cycles have inspected this
            entry.
        last_reviewed: ISO-8601 timestamp of the most recent review, or None.
        resolution: One of ``None`` (unresolved), ``"clarified_by_experience"``,
            ``"clarified_by_weight_changes"``, ``"expired"``, or
            ``"promoted"``.
        related_experience_ids: IDs of other experiences that share context or
            domain overlap with this one.
    """

    experience_id: str
    timestamp: str
    input_context: str
    output: str
    domain_tags: List[str]
    ambiguity_type: str
    ambiguity_score: float
    partial_signals: Dict[str, Any]
    review_count: int = 0
    last_reviewed: Optional[str] = None
    resolution: Optional[str] = None
    related_experience_ids: List[str] = field(default_factory=list)

    # ---- validation ----

    def __post_init__(self) -> None:
        valid_types = {
            "mixed_signals",
            "low_confidence",
            "novel_domain",
            "contradiction",
            "unknown",
        }
        if self.ambiguity_type not in valid_types:
            raise ValueError(
                f"ambiguity_type must be one of {valid_types}, "
                f"got {self.ambiguity_type!r}"
            )
        if not (0.0 <= self.ambiguity_score <= 1.0):
            raise ValueError(
                f"ambiguity_score must be in [0.0, 1.0], "
                f"got {self.ambiguity_score}"
            )

    # ---- serialisation ----

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AmbiguousExperience:
        """Reconstruct from a dictionary."""
        return cls(**data)


# -----------------------------------------------------------------------
# Persistence & detection
# -----------------------------------------------------------------------

class AmbiguityBuffer:
    """Append-only buffer for ambiguous experiences, persisted as JSONL.

    Parameters:
        storage_path: Path to the backing JSONL file.
    """

    # Thresholds ----------------------------------------------------------
    CONFIDENCE_THRESHOLD: float = 0.45
    CONTRADICTION_THRESHOLD: float = 0.6
    NOVELTY_DOMAIN_MIN_EXPERIENCES: int = 3

    def __init__(self, storage_path: str | Path) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ---- public API -----------------------------------------------------

    def shelve(self, ambiguous_experience: AmbiguousExperience) -> str:
        """Persist an ambiguous experience and return its id."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ambiguous_experience.to_dict()) + "\n")
        logger.info(
            "Shelved ambiguous experience %s (type=%s, score=%.2f)",
            ambiguous_experience.experience_id,
            ambiguous_experience.ambiguity_type,
            ambiguous_experience.ambiguity_score,
        )
        return ambiguous_experience.experience_id

    def detect_ambiguity(
        self,
        experience: Any,
        model: Any,
        reward_signals: Dict[str, Any],
    ) -> Optional[AmbiguousExperience]:
        """Analyse an experience and its reward signals for ambiguity.

        If the experience is ambiguous it is automatically shelved and the
        :class:`AmbiguousExperience` is returned.  Otherwise ``None`` is
        returned.

        Parameters:
            experience: A :class:`~openmind.core.experience.Experience`.
            model: The current model wrapper (used for novelty checks).
            reward_signals: Dictionary of reward dimension values and metadata.
        """
        confidence = self._measure_output_confidence(reward_signals)
        is_novel = self._is_novel_domain(experience.domain_tags, model)
        contradiction = self._check_contradiction(experience, model, reward_signals)

        # Determine ambiguity type and score
        ambiguity_type: Optional[str] = None
        ambiguity_score: float = 0.0

        if contradiction["detected"]:
            ambiguity_type = "contradiction"
            ambiguity_score = contradiction["severity"]
        elif confidence < self.CONFIDENCE_THRESHOLD:
            # Check whether signals are mixed or just weak
            has_conflict = reward_signals.get("has_conflicts", False)
            if has_conflict:
                ambiguity_type = "mixed_signals"
                ambiguity_score = 1.0 - confidence
            else:
                ambiguity_type = "low_confidence"
                ambiguity_score = 1.0 - confidence
        elif is_novel:
            ambiguity_type = "novel_domain"
            ambiguity_score = 0.7  # default high-ish for novel domains

        if ambiguity_type is None:
            return None  # not ambiguous

        now = datetime.now(timezone.utc).isoformat()
        amb = AmbiguousExperience(
            experience_id=experience.experience_id,
            timestamp=now,
            input_context=experience.input_context,
            output=experience.output,
            domain_tags=list(experience.domain_tags),
            ambiguity_type=ambiguity_type,
            ambiguity_score=ambiguity_score,
            partial_signals=reward_signals,
        )
        self.shelve(amb)
        return amb

    def load_all(self) -> List[AmbiguousExperience]:
        """Load every shelved ambiguous experience."""
        if not self._path.exists():
            return []
        items: List[AmbiguousExperience] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    items.append(AmbiguousExperience.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping corrupt entry at %s:%d -- %s",
                        self._path, lineno, exc,
                    )
        return items

    def load_unresolved(self) -> List[AmbiguousExperience]:
        """Return only unresolved ambiguous experiences."""
        return [a for a in self.load_all() if a.resolution is None]

    def update(self, updated: AmbiguousExperience) -> None:
        """Rewrite the buffer replacing the entry with matching id.

        This is an O(n) rewrite; suitable for the modest sizes expected in
        an ambiguity buffer.
        """
        all_items = self.load_all()
        found = False
        for idx, item in enumerate(all_items):
            if item.experience_id == updated.experience_id:
                all_items[idx] = updated
                found = True
                break
        if not found:
            logger.warning(
                "update() called for unknown experience %s; appending instead",
                updated.experience_id,
            )
            all_items.append(updated)
        self._rewrite(all_items)

    def __len__(self) -> int:
        return len(self.load_all())

    # ---- internal helpers -----------------------------------------------

    def _measure_output_confidence(
        self, reward_signals: Dict[str, Any],
    ) -> float:
        """Derive a scalar confidence from reward signals.

        Uses the ``confidence`` key if present, otherwise averages the
        absolute values of measured dimensions as a rough proxy.
        """
        if "confidence" in reward_signals:
            return float(reward_signals["confidence"])

        dims = [
            "task_success", "coherence", "user_satisfaction",
            "groundedness", "novelty_value",
        ]
        values = [
            abs(float(reward_signals[d]))
            for d in dims
            if reward_signals.get(d) is not None
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _is_novel_domain(
        self,
        domain_tags: Sequence[str],
        model: Any,
    ) -> bool:
        """Return True if these domain tags are under-represented.

        Checks the buffer itself: if fewer than
        ``NOVELTY_DOMAIN_MIN_EXPERIENCES`` existing entries share any of
        the tags, the domain is considered novel.
        """
        existing = self.load_all()
        tag_set = set(domain_tags)
        count = sum(
            1 for e in existing
            if tag_set.intersection(e.domain_tags)
        )
        return count < self.NOVELTY_DOMAIN_MIN_EXPERIENCES

    def _check_contradiction(
        self,
        experience: Any,
        model: Any,
        reward_signals: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check whether the experience contradicts established knowledge.

        Parameters:
            experience: The experience being evaluated.
            model: The current model (used for generating comparison outputs).
            reward_signals: Current reward data.

        Returns:
            Dict with ``detected`` (bool) and ``severity`` (float) keys.
        """
        # Build a prompt that asks the model to evaluate contradiction
        prompt = (
            "You are a knowledge-consistency checker.  Given the following "
            "input and output pair, determine whether the output contradicts "
            "any previously established knowledge you are confident about.\n\n"
            f"Input: {experience.input_context}\n"
            f"Output: {experience.output}\n\n"
            "Respond with a JSON object: "
            '{"contradicts": true/false, "severity": 0.0-1.0, '
            '"explanation": "..."}.'
        )

        # TODO: Send *prompt* to *model* and parse the JSON response.
        # For now, fall back to a simple heuristic: if the aggregate reward
        # is strongly negative but confidence is high, flag as contradiction.
        aggregate = reward_signals.get("aggregate", 0.0)
        confidence = reward_signals.get("confidence", 0.0)

        if aggregate is not None and aggregate < -0.4 and confidence > 0.6:
            return {"detected": True, "severity": min(1.0, abs(aggregate))}
        return {"detected": False, "severity": 0.0}

    def _rewrite(self, items: List[AmbiguousExperience]) -> None:
        """Overwrite the backing JSONL file with *items*."""
        with open(self._path, "w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item.to_dict()) + "\n")


# -----------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------

class AmbiguityResolver:
    """Periodically reviews shelved ambiguities and attempts resolution.

    Resolution follows three possible paths:

    1. **clarified_by_experience** -- newer experiences in the same domain
       provide enough consistent signal to resolve the ambiguity.
    2. **clarified_by_weight_changes** -- the model's weights have shifted
       (via other training) such that the ambiguity is no longer present.
    3. **expired** -- the item has been reviewed at least
       ``MIN_REVIEWS_FOR_EXPIRY`` times and is older than
       ``MAX_AGE_DAYS`` days, so it is retired without learning from it.

    Parameters:
        ambiguity_buffer: The :class:`AmbiguityBuffer` to draw from.
        experience_buffer: The main :class:`ExperienceBuffer` for context.
    """

    MAX_AGE_DAYS: int = 30
    MIN_REVIEWS_FOR_EXPIRY: int = 5
    CLARIFICATION_CONFIDENCE_THRESHOLD: float = 0.65

    def __init__(
        self,
        ambiguity_buffer: AmbiguityBuffer,
        experience_buffer: Any,
    ) -> None:
        self._ambiguity_buffer = ambiguity_buffer
        self._experience_buffer = experience_buffer

    def review_cycle(
        self,
        model: Any,
        current_cycle: int,
    ) -> Dict[str, List[str]]:
        """Run one review cycle over all unresolved ambiguities.

        Parameters:
            model: The current model wrapper.
            current_cycle: The training-cycle ordinal (for logging).

        Returns:
            Dict mapping resolution outcome to lists of experience ids.
        """
        outcomes: Dict[str, List[str]] = {
            "clarified_by_experience": [],
            "clarified_by_weight_changes": [],
            "expired": [],
            "still_unresolved": [],
        }

        unresolved = self._ambiguity_buffer.load_unresolved()
        logger.info(
            "AmbiguityResolver cycle %d: reviewing %d unresolved items",
            current_cycle, len(unresolved),
        )

        for amb in unresolved:
            resolution = self._attempt_resolution(amb, model)
            now = datetime.now(timezone.utc).isoformat()
            amb.review_count += 1
            amb.last_reviewed = now

            if resolution is not None:
                amb.resolution = resolution
                outcomes[resolution].append(amb.experience_id)
                logger.info(
                    "Resolved %s as %s after %d reviews",
                    amb.experience_id, resolution, amb.review_count,
                )
            else:
                outcomes["still_unresolved"].append(amb.experience_id)

            self._ambiguity_buffer.update(amb)

        return outcomes

    def _attempt_resolution(
        self,
        amb: AmbiguousExperience,
        model: Any,
    ) -> Optional[str]:
        """Try each resolution path in order, return the first success.

        Returns:
            A resolution string or ``None`` if still unresolved.
        """
        # Path 1: clarified by newer experiences
        if self._clarified_by_experience(amb):
            return "clarified_by_experience"

        # Path 2: clarified by weight changes
        if self._clarified_by_weight_changes(amb, model):
            return "clarified_by_weight_changes"

        # Path 3: expired (old enough + reviewed enough times)
        if self._is_expired(amb):
            return "expired"

        return None

    # ---- resolution path helpers ----------------------------------------

    def _clarified_by_experience(self, amb: AmbiguousExperience) -> bool:
        """Check whether newer experiences resolve the ambiguity.

        Looks for recent experiences in the same domain that have high,
        consistent reward signals -- enough to override the original
        ambiguity.
        """
        recent = self._experience_buffer.get_by_domain(
            amb.domain_tags, days_back=14,
        )

        if len(recent) < 3:
            return False

        # Check whether recent domain experiences are consistently high-signal
        rewards = [e.reward_signal for e in recent]
        avg_reward = sum(rewards) / len(rewards)
        consistency = 1.0 - (
            sum(abs(r - avg_reward) for r in rewards) / len(rewards)
        )

        return (
            abs(avg_reward) > 0.5
            and consistency > self.CLARIFICATION_CONFIDENCE_THRESHOLD
        )

    def _clarified_by_weight_changes(
        self,
        amb: AmbiguousExperience,
        model: Any,
    ) -> bool:
        """Re-run the ambiguous input through the model and check whether
        the output is now confident and non-contradictory.

        Parameters:
            amb: The ambiguous experience to re-evaluate.
            model: The current model wrapper.
        """
        # Build prompt for re-evaluation
        prompt = (
            "Given the following input, provide a confident answer.\n\n"
            f"Input: {amb.input_context}\n\n"
            "Answer:"
        )

        # TODO: Send *prompt* to *model*, capture the new output and its
        # reward signals.  Compare the new confidence / reward to the
        # original partial_signals stored in *amb*.  If the new confidence
        # exceeds CLARIFICATION_CONFIDENCE_THRESHOLD, return True.
        #
        # Placeholder: always return False until model integration is wired.
        return False

    def _is_expired(self, amb: AmbiguousExperience) -> bool:
        """Return True if the item is old enough and reviewed enough."""
        if amb.review_count < self.MIN_REVIEWS_FOR_EXPIRY:
            return False
        created = datetime.fromisoformat(amb.timestamp)
        age = datetime.now(timezone.utc) - created
        return age > timedelta(days=self.MAX_AGE_DAYS)
