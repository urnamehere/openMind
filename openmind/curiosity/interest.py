"""Interest model -- tracks learning progress and curiosity across domains.

Inspired by Schmidhuber's theory of curiosity: an agent is most curious
about things where it is *making progress* learning, not things that are
simply unfamiliar (which may be noise) or already mastered (boring).

The interest model maintains per-domain statistics that capture:
- How fast the system is improving (learning velocity)
- How much unexplored territory remains (knowledge gaps)
- How rewarding interactions in this domain tend to be
- How recently the domain was active (recency)

These combine into a **curiosity score** that drives autonomous exploration.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DomainInterest:
    """Tracks the system's evolving interest in a single domain."""

    domain: str

    # Learning progress signals
    total_experiences: int = 0
    positive_experiences: int = 0
    negative_experiences: int = 0
    knowledge_entries: int = 0
    recent_reward_trend: float = 0.0  # positive = improving

    # Curiosity components
    novelty_score: float = 1.0       # 1.0 = brand new, decays with exposure
    mastery_score: float = 0.0       # 0.0 = novice, grows toward 1.0
    learning_velocity: float = 0.0   # rate of mastery change (the key signal)
    gap_score: float = 0.0           # how many unanswered questions exist

    # Engagement history
    last_interaction: float = 0.0
    last_exploration: float = 0.0
    exploration_count: int = 0
    shelved_count: int = 0           # ambiguous experiences = knowledge gaps

    # Cross-domain connections discovered
    connected_domains: List[str] = field(default_factory=list)

    @property
    def curiosity_score(self) -> float:
        """Compute overall curiosity about this domain.

        High curiosity when:
        - Learning velocity is positive (we're making progress)
        - There are knowledge gaps (shelved experiences, unanswered questions)
        - Novelty is moderate (not totally unfamiliar, not fully explored)
        - Recent rewards are improving

        Low curiosity when:
        - Mastery is very high (boring, nothing left to learn)
        - Mastery is very low AND velocity is zero (too hard, no progress)
        - Domain has gone stale (no recent interactions)
        """
        # Learning progress is the strongest signal (Schmidhuber)
        progress_factor = self._sigmoid(self.learning_velocity * 5.0)

        # Knowledge gaps create pull
        gap_factor = min(self.gap_score, 1.0)

        # Novelty follows an inverted-U: moderate novelty is most interesting
        # Too novel (1.0) = random/scary, too familiar (0.0) = boring
        novelty_factor = 4.0 * self.novelty_score * (1.0 - self.novelty_score)

        # Recency: interest fades without interaction, but slowly
        hours_since = (time.time() - self.last_interaction) / 3600.0
        recency_factor = math.exp(-hours_since / 168.0)  # half-life ~1 week

        # Mastery penalty: very high mastery = diminishing returns
        mastery_penalty = 1.0 - (self.mastery_score ** 3)

        # Weighted combination
        score = (
            progress_factor * 0.35
            + gap_factor * 0.25
            + novelty_factor * 0.20
            + recency_factor * 0.10
            + mastery_penalty * 0.10
        )

        return max(0.0, min(1.0, score))

    @property
    def interest_level(self) -> str:
        """Human-readable interest classification."""
        score = self.curiosity_score
        if score > 0.7:
            return "fascinated"
        elif score > 0.5:
            return "curious"
        elif score > 0.3:
            return "interested"
        elif score > 0.15:
            return "aware"
        else:
            return "dormant"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainInterest:
        data.pop("curiosity_score", None)
        data.pop("interest_level", None)
        connected = data.pop("connected_domains", [])
        di = cls(**data)
        di.connected_domains = connected
        return di

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Sigmoid squashing to [0, 1]."""
        return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))


class InterestModel:
    """Tracks curiosity and learning progress across all domains.

    Maintains a persistent model of how interested the system is in
    each domain it has encountered. Updated after every interaction
    and used by the CuriosityEngine to decide what to explore.

    Parameters:
        storage_path: Path to persist interest data as JSON.
        novelty_decay: How fast novelty decays with each interaction (0-1).
        mastery_growth_rate: How fast mastery grows per positive experience.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        novelty_decay: float = 0.02,
        mastery_growth_rate: float = 0.01,
    ) -> None:
        self._path = Path(storage_path) if storage_path else None
        self._novelty_decay = novelty_decay
        self._mastery_growth_rate = mastery_growth_rate
        self._domains: Dict[str, DomainInterest] = {}
        self._reward_windows: Dict[str, List[float]] = defaultdict(list)

        if self._path and self._path.exists():
            self._load()

    def update_from_experience(
        self,
        domain_tags: List[str],
        reward: float,
        was_shelved: bool = False,
    ) -> None:
        """Update interest model after an interaction.

        Parameters:
            domain_tags: Domains this interaction belongs to.
            reward: The aggregate reward signal for this interaction.
            was_shelved: Whether the experience was shelved (ambiguous).
        """
        now = time.time()

        for domain in domain_tags:
            di = self._get_or_create(domain)
            di.total_experiences += 1
            di.last_interaction = now

            if reward > 0.3:
                di.positive_experiences += 1
            elif reward < -0.3:
                di.negative_experiences += 1

            if was_shelved:
                di.shelved_count += 1
                di.gap_score = min(1.0, di.gap_score + 0.1)

            # Decay novelty with exposure
            di.novelty_score = max(0.0, di.novelty_score - self._novelty_decay)

            # Update mastery based on reward
            old_mastery = di.mastery_score
            if reward > 0.3:
                di.mastery_score = min(
                    1.0,
                    di.mastery_score + self._mastery_growth_rate * reward,
                )
            elif reward < -0.3:
                # Mistakes slightly reduce mastery
                di.mastery_score = max(
                    0.0,
                    di.mastery_score + self._mastery_growth_rate * reward * 0.5,
                )

            # Learning velocity = rate of mastery change
            di.learning_velocity = (
                0.7 * di.learning_velocity
                + 0.3 * (di.mastery_score - old_mastery)
            )

            # Track recent reward trend
            self._reward_windows[domain].append(reward)
            window = self._reward_windows[domain][-20:]
            self._reward_windows[domain] = window
            if len(window) >= 5:
                first_half = sum(window[: len(window) // 2]) / (len(window) // 2)
                second_half = sum(window[len(window) // 2 :]) / (
                    len(window) - len(window) // 2
                )
                di.recent_reward_trend = second_half - first_half

        self._save()

    def update_from_knowledge(self, domain_tags: List[str]) -> None:
        """Update when new knowledge is registered in a domain."""
        for domain in domain_tags:
            di = self._get_or_create(domain)
            di.knowledge_entries += 1
            # Knowledge reduces gap score
            di.gap_score = max(0.0, di.gap_score - 0.15)
        self._save()

    def record_exploration(self, domain: str) -> None:
        """Record that an autonomous exploration was done in this domain."""
        di = self._get_or_create(domain)
        di.exploration_count += 1
        di.last_exploration = time.time()
        self._save()

    def discover_connection(self, domain_a: str, domain_b: str) -> None:
        """Record a cross-domain connection."""
        di_a = self._get_or_create(domain_a)
        di_b = self._get_or_create(domain_b)
        if domain_b not in di_a.connected_domains:
            di_a.connected_domains.append(domain_b)
        if domain_a not in di_b.connected_domains:
            di_b.connected_domains.append(domain_a)
        self._save()

    def get_most_curious(self, top_k: int = 5) -> List[Tuple[str, DomainInterest]]:
        """Return domains ranked by curiosity score."""
        ranked = sorted(
            self._domains.items(),
            key=lambda kv: kv[1].curiosity_score,
            reverse=True,
        )
        return ranked[:top_k]

    def get_knowledge_gaps(self) -> List[Tuple[str, DomainInterest]]:
        """Return domains with significant knowledge gaps."""
        gaps = [
            (name, di)
            for name, di in self._domains.items()
            if di.gap_score > 0.3 or di.shelved_count > 3
        ]
        gaps.sort(key=lambda kv: kv[1].gap_score, reverse=True)
        return gaps

    def get_improving_domains(self) -> List[Tuple[str, DomainInterest]]:
        """Return domains where the system is actively improving."""
        improving = [
            (name, di)
            for name, di in self._domains.items()
            if di.learning_velocity > 0.005
        ]
        improving.sort(key=lambda kv: kv[1].learning_velocity, reverse=True)
        return improving

    def get_stale_domains(self, hours: float = 168.0) -> List[Tuple[str, DomainInterest]]:
        """Return domains that haven't been active recently."""
        cutoff = time.time() - hours * 3600.0
        stale = [
            (name, di)
            for name, di in self._domains.items()
            if di.last_interaction < cutoff and di.total_experiences > 5
        ]
        stale.sort(key=lambda kv: kv[1].last_interaction)
        return stale

    def get_domain(self, domain: str) -> Optional[DomainInterest]:
        """Get interest data for a specific domain."""
        return self._domains.get(domain)

    def get_all(self) -> Dict[str, DomainInterest]:
        """Return all domain interest data."""
        return dict(self._domains)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the interest model state."""
        if not self._domains:
            return {"total_domains": 0, "most_curious": [], "knowledge_gaps": []}

        most_curious = self.get_most_curious(3)
        gaps = self.get_knowledge_gaps()[:3]

        return {
            "total_domains": len(self._domains),
            "most_curious": [
                {"domain": name, "score": f"{di.curiosity_score:.2f}", "level": di.interest_level}
                for name, di in most_curious
            ],
            "knowledge_gaps": [
                {"domain": name, "gap_score": f"{di.gap_score:.2f}"}
                for name, di in gaps
            ],
            "total_explorations": sum(
                di.exploration_count for di in self._domains.values()
            ),
        }

    # === Internal ===

    def _get_or_create(self, domain: str) -> DomainInterest:
        if domain not in self._domains:
            self._domains[domain] = DomainInterest(
                domain=domain, last_interaction=time.time()
            )
        return self._domains[domain]

    def _save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {name: di.to_dict() for name, di in self._domains.items()}
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for name, entry in data.items():
                self._domains[name] = DomainInterest.from_dict(entry)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning("Failed to load interest model: %s", e)
