"""Experience recording and replay buffer for continuous learning.

Stores structured learning experiences as JSONL and provides sampling
strategies for training cycles (recent high-signal, replay, domain-filtered).
"""
from __future__ import annotations

import json
import logging
import os
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single learning experience captured from model interaction.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of when the experience occurred.
        input_context: The prompt / context that was presented to the model.
        output: The model's generated response.
        reward_signal: Scalar reward in [-1.0, 1.0] assigned by the reward model
            or human feedback.  Higher is better.
        domain_tags: Free-form tags indicating the knowledge domain(s) involved
            (e.g. ["math", "algebra"]).
        confidence: Model's self-reported confidence in [0.0, 1.0].
        experience_id: Unique identifier, auto-generated if not supplied.
        training_weight: Multiplier applied during training loss computation.
            Defaults to 1.0; callers can up-weight or down-weight specific
            experiences before a training cycle.
    """

    timestamp: str
    input_context: str
    output: str
    reward_signal: float
    domain_tags: List[str]
    confidence: float
    experience_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    training_weight: float = 1.0

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if not (-1.0 <= self.reward_signal <= 1.0):
            raise ValueError(
                f"reward_signal must be in [-1.0, 1.0], got {self.reward_signal}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )
        if self.training_weight < 0.0:
            raise ValueError(
                f"training_weight must be >= 0.0, got {self.training_weight}"
            )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Experience:
        """Reconstruct an Experience from a dictionary."""
        return cls(**data)


class ExperienceBuffer:
    """Append-only experience store backed by a JSONL file.

    Each line in the backing file is a JSON object representing one
    :class:`Experience`.  The buffer supports several sampling strategies
    needed by the training cycle:

    * **record** -- append a new experience.
    * **get_training_batch** -- high-reward experiences suitable for a
      training step.
    * **get_replay_sample** -- uniformly sampled past experiences for
      experience-replay regularisation.
    * **get_recent / get_historical** -- time-ordered slices.
    * **get_by_domain** -- domain-filtered retrieval.

    Parameters:
        storage_path: Path to the JSONL file.  Created on first write.
    """

    def __init__(self, storage_path: str | os.PathLike) -> None:
        self._path = Path(storage_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, experience: Experience) -> str:
        """Persist *experience* and return its ``experience_id``."""
        with open(self._path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(experience.to_dict()) + "\n")
        logger.debug("Recorded experience %s", experience.experience_id)
        return experience.experience_id

    def get_training_batch(
        self,
        min_reward: float = 0.3,
        max_samples: int = 256,
    ) -> List[Experience]:
        """Return up to *max_samples* experiences whose reward >= *min_reward*.

        Experiences are sorted by reward (descending) so the strongest
        signals come first.  Within equal reward the original insertion
        order is preserved.
        """
        all_exp = self._load_all()
        filtered = [e for e in all_exp if e.reward_signal >= min_reward]
        filtered.sort(key=lambda e: e.reward_signal, reverse=True)
        batch = filtered[:max_samples]
        logger.info(
            "Training batch: %d / %d experiences above reward %.2f",
            len(batch),
            len(all_exp),
            min_reward,
        )
        return batch

    def get_replay_sample(self, n: int = 64) -> List[Experience]:
        """Return *n* uniformly sampled experiences (experience replay).

        If the buffer contains fewer than *n* experiences, all are returned.
        """
        all_exp = self._load_all()
        if len(all_exp) <= n:
            return all_exp
        return random.sample(all_exp, n)

    def get_recent(self, n: int = 100) -> List[Experience]:
        """Return the *n* most recent experiences (by timestamp)."""
        all_exp = self._load_all()
        all_exp.sort(key=lambda e: e.timestamp, reverse=True)
        return all_exp[:n]

    def get_historical(self, n: int = 100) -> List[Experience]:
        """Return the *n* oldest experiences (by timestamp)."""
        all_exp = self._load_all()
        all_exp.sort(key=lambda e: e.timestamp)
        return all_exp[:n]

    def get_by_domain(
        self,
        domain_tags: Sequence[str],
        days_back: int = 30,
    ) -> List[Experience]:
        """Return experiences matching *any* of the given domain tags
        that were recorded within the last *days_back* days.

        Parameters:
            domain_tags: One or more tags to match against.
            days_back: Only consider experiences newer than this many days.
        """
        tag_set = set(domain_tags)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
        cutoff_iso = cutoff.isoformat()

        results: List[Experience] = []
        for exp in self._load_all():
            if exp.timestamp < cutoff_iso:
                continue
            if tag_set.intersection(exp.domain_tags):
                results.append(exp)
        return results

    def __len__(self) -> int:
        """Return the total number of stored experiences."""
        return len(self._load_all())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_all(self) -> List[Experience]:
        """Read every experience from the backing JSONL file."""
        if not self._path.exists():
            return []
        experiences: List[Experience] = []
        with open(self._path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    experiences.append(Experience.from_dict(data))
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    logger.warning(
                        "Skipping corrupt experience at %s:%d -- %s",
                        self._path,
                        lineno,
                        exc,
                    )
        return experiences
