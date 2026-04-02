"""Reward signal representation for the openMind continuous learning system.

A RewardSignal captures multi-dimensional feedback about an LLM interaction,
allowing the system to learn from task success, coherence, user satisfaction,
groundedness, and novelty along independent axes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RewardSignal:
    """Multi-dimensional reward signal for a single interaction.

    Each dimension is Optional[float] in [-1.0, 1.0].  None means the
    dimension was not measured for this interaction.

    Attributes:
        task_success:     Did the output accomplish the requested task?
        coherence:        Is the output internally consistent and well-structured?
        user_satisfaction: Did the user appear satisfied (explicit or inferred)?
        groundedness:     Is the output grounded in retrieved or known facts?
        novelty_value:    Does the output surface useful new information or
                          connections?  Negative means unhelpful repetition.
        signal_sources:   Maps dimension name -> source label (e.g. "self_eval",
                          "explicit_feedback", "execution_result").
        timestamp:        Unix timestamp of signal creation.
    """

    task_success: Optional[float] = None
    coherence: Optional[float] = None
    user_satisfaction: Optional[float] = None
    groundedness: Optional[float] = None
    novelty_value: Optional[float] = None

    signal_sources: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Weights used for the weighted aggregate
    _DIMENSION_WEIGHTS: Dict[str, float] = field(
        default=None,  # type: ignore[assignment]
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self._DIMENSION_WEIGHTS = {
            "task_success": 0.35,
            "coherence": 0.15,
            "user_satisfaction": 0.25,
            "groundedness": 0.15,
            "novelty_value": 0.10,
        }
        # Clamp all dimensions to [-1, 1]
        for dim in self._dimension_names():
            val = getattr(self, dim)
            if val is not None:
                clamped = max(-1.0, min(1.0, float(val)))
                object.__setattr__(self, dim, clamped)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dimension_names() -> List[str]:
        return [
            "task_success",
            "coherence",
            "user_satisfaction",
            "groundedness",
            "novelty_value",
        ]

    def _get_measured_values(self) -> Dict[str, float]:
        """Return {name: value} for dimensions that are not None."""
        return {
            dim: getattr(self, dim)
            for dim in self._dimension_names()
            if getattr(self, dim) is not None
        }

    def _has_conflicts(self) -> bool:
        """Detect contradictory signals.

        Conflicts are defined as having some dimensions strongly positive
        (>= 0.5) and others strongly negative (<= -0.5).
        """
        measured = self._get_measured_values()
        if len(measured) < 2:
            return False
        strong_pos = any(v >= 0.5 for v in measured.values())
        strong_neg = any(v <= -0.5 for v in measured.values())
        return strong_pos and strong_neg

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def confidence(self) -> float:
        """How confident are we in this signal?

        confidence = coverage * 0.6 + agreement * 0.4

        *coverage* is the fraction of dimensions that were actually measured.
        *agreement* is 1.0 when all measured dimensions have the same sign,
        0.0 when they are perfectly split, interpolated otherwise.
        """
        total_dims = len(self._dimension_names())
        measured = self._get_measured_values()
        coverage = len(measured) / total_dims if total_dims else 0.0

        if len(measured) < 2:
            agreement = 1.0  # single signal doesn't disagree with itself
        else:
            pos_count = sum(1 for v in measured.values() if v >= 0)
            neg_count = len(measured) - pos_count
            majority = max(pos_count, neg_count)
            agreement = majority / len(measured)

        return coverage * 0.6 + agreement * 0.4

    @property
    def aggregate(self) -> float:
        """Weighted combination of measured dimensions.

        Only measured dimensions participate; their weights are re-normalised
        so the result stays in [-1, 1].
        """
        measured = self._get_measured_values()
        if not measured:
            return 0.0

        total_weight = sum(self._DIMENSION_WEIGHTS[name] for name in measured)
        if total_weight == 0.0:
            return 0.0

        return sum(
            self._DIMENSION_WEIGHTS[name] * val / total_weight
            for name, val in measured.items()
        )

    @property
    def is_ambiguous(self) -> bool:
        """A signal is ambiguous when confidence is low or conflicts exist."""
        return self.confidence < 0.5 or self._has_conflicts()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dict (JSON-safe)."""
        return {
            dim: getattr(self, dim) for dim in self._dimension_names()
        } | {
            "signal_sources": dict(self.signal_sources),
            "timestamp": self.timestamp,
            "aggregate": self.aggregate,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RewardSignal":
        dim_kwargs = {
            dim: data.get(dim) for dim in cls._dimension_names()
        }
        return cls(
            **dim_kwargs,
            signal_sources=data.get("signal_sources", {}),
            timestamp=data.get("timestamp", time.time()),
        )
