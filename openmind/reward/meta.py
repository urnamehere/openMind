"""Meta-reward system -- rewards on the reward system itself.

Periodically evaluates whether the reward pipeline is actually making the
model better.  Tracks five key metrics (revision_rate, contradiction_rate,
predictive_validity, benchmark_correlation, self_consistency), derives a
composite health score, and generates actionable recommendations.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Stdlib statistics helpers (avoid hard numpy dependency)
# -----------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson correlation coefficient, or 0.0 on degenerate input."""
    n = len(xs)
    if n != len(ys) or n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    sx, sy = _std(xs), _std(ys)
    if sx < 1e-9 or sy < 1e-9:
        return 0.0
    return cov / (sx * sy)


# -----------------------------------------------------------------------
# Meta-reward system
# -----------------------------------------------------------------------

class MetaRewardSystem:
    """Monitors the health of the entire reward pipeline.

    The fundamental question: is our reward signal actually making
    the model better?  Without this check you can have a perfectly
    functioning reward pipeline that optimises for the wrong thing.

    Usage::

        meta = MetaRewardSystem()
        # After each training cycle:
        meta.track_training_outcome(cycle=1, avg_reward=0.6,
                                     benchmark_before=72.3,
                                     benchmark_after=73.1)
        # Periodically:
        metrics, recs = meta.evaluate_reward_quality(temporal_tracker)
    """

    def __init__(self) -> None:
        # Full evaluation history (one entry per evaluate_reward_quality call)
        self.evaluation_history: List[Dict[str, Any]] = []

        # Self-eval consistency samples (each a score in [0, 1])
        self.reward_prediction_accuracy: List[float] = []

        # (avg_reward, benchmark_delta) pairs logged after each training cycle
        self.training_outcome_pairs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_reward_quality(
        self,
        temporal_tracker: Any,
        training_log: Optional[List[Dict[str, Any]]] = None,
        model_evaluator: Optional[Any] = None,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Run a full quality evaluation of the reward system.

        Should be called periodically (e.g. after every N training cycles).

        Args:
            temporal_tracker: A :class:`TemporalRewardTracker` (or any object
                exposing an ``active_rewards`` dict or a ``_load_all()`` method).
            training_log: Optional list of training-cycle log dicts.
            model_evaluator: Optional model evaluator for predictive validity.

        Returns:
            A 2-tuple of ``(metrics_dict, recommendations_list)``.
        """
        metrics: Dict[str, Any] = {}

        # Gather events from the tracker
        events = self._gather_events(temporal_tracker)
        total = len(events)

        # --- METRIC 1: revision_rate ---
        revised = sum(
            1 for ev in events
            if self._get_attr(ev, "revisions", default=[])
        )
        metrics["revision_rate"] = revised / max(total, 1)

        # --- METRIC 2: contradiction_rate ---
        contradicted = sum(
            1 for ev in events
            if self._get_attr(ev, "contradiction_detected", default=False)
        )
        metrics["contradiction_rate"] = contradicted / max(total, 1)

        # --- METRIC 3: predictive_validity ---
        metrics["predictive_validity"] = self._compute_predictive_validity(
            model_evaluator
        )

        # --- METRIC 4: benchmark_correlation ---
        metrics["benchmark_correlation"] = self._compute_benchmark_correlation()

        # --- METRIC 5: self_consistency ---
        metrics["self_consistency"] = self._compute_self_consistency()

        # --- Composite health score ---
        metrics["reward_system_health"] = self._compute_health_score(metrics)

        self.evaluation_history.append({
            "timestamp": time.time(),
            "event_count": total,
            "metrics": metrics,
        })

        recommendations = self._generate_recommendations(metrics)

        logger.info(
            "Meta-reward evaluation: health=%.2f revision=%.2f contradiction=%.2f "
            "pred_val=%s bench_corr=%s self_cons=%s",
            metrics["reward_system_health"],
            metrics["revision_rate"],
            metrics["contradiction_rate"],
            metrics.get("predictive_validity"),
            metrics.get("benchmark_correlation"),
            metrics.get("self_consistency"),
        )

        return metrics, recommendations

    def track_training_outcome(
        self,
        training_cycle: int,
        avg_reward: float,
        benchmark_before: float,
        benchmark_after: float,
    ) -> None:
        """Record the relationship between reward signals and benchmark outcomes.

        Call this after each training cycle so that predictive_validity and
        benchmark_correlation can be computed.

        Args:
            training_cycle: Ordinal cycle number.
            avg_reward: Mean reward of experiences used in this cycle.
            benchmark_before: Benchmark score before the cycle.
            benchmark_after: Benchmark score after the cycle.
        """
        self.training_outcome_pairs.append({
            "cycle": training_cycle,
            "timestamp": time.time(),
            "avg_reward": avg_reward,
            "benchmark_before": benchmark_before,
            "benchmark_after": benchmark_after,
            "delta": benchmark_after - benchmark_before,
        })
        logger.debug(
            "Tracked training outcome: cycle=%d reward=%.3f delta=%.3f",
            training_cycle, avg_reward, benchmark_after - benchmark_before,
        )

    def record_prediction_accuracy(self, accuracy: float) -> None:
        """Record a self-evaluation prediction accuracy sample.

        Used by the self-consistency metric.

        Args:
            accuracy: A score in [0, 1] measuring how close a re-scored
                reward was to its original score.
        """
        self.reward_prediction_accuracy.append(max(0.0, min(1.0, accuracy)))

    # ------------------------------------------------------------------
    # Metric computations
    # ------------------------------------------------------------------

    def _compute_predictive_validity(
        self,
        evaluator: Optional[Any],
    ) -> Optional[float]:
        """Do high-reward experiences correspond to better model behaviour?

        Computes Pearson correlation between per-cycle average reward and
        the benchmark delta (improvement).  Returns None when insufficient
        data or no evaluator is provided.
        """
        if evaluator is None or len(self.training_outcome_pairs) < 5:
            return None

        rewards = [p["avg_reward"] for p in self.training_outcome_pairs]
        deltas = [p["delta"] for p in self.training_outcome_pairs]

        r = _pearson(rewards, deltas)
        # Clamp to [0, 1]: negative correlation means the signal is harmful
        return max(0.0, r)

    def _compute_benchmark_correlation(self) -> Optional[float]:
        """Do cycles with higher average reward produce better benchmarks?

        Similar to predictive validity but does not require an evaluator
        -- relies purely on tracked training outcomes.
        """
        if len(self.training_outcome_pairs) < 5:
            return None

        rewards = [p["avg_reward"] for p in self.training_outcome_pairs]
        deltas = [p["delta"] for p in self.training_outcome_pairs]

        return _pearson(rewards, deltas)

    def _compute_self_consistency(self) -> Optional[float]:
        """If we re-scored the same experience, would we get similar scores?

        Measured as ``1 - std(recent_accuracy_samples)``.  Low variance
        means high consistency.  Returns None when fewer than 10 samples
        are available.
        """
        if len(self.reward_prediction_accuracy) < 10:
            return None
        recent = self.reward_prediction_accuracy[-50:]
        return max(0.0, 1.0 - _std(recent))

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """Aggregate health score in [0.0, 1.0].

        Starts at 1.0 and subtracts penalties for each warning sign.
        """
        score = 1.0

        # High revision rate suggests initial signals are unreliable
        rev_rate = metrics.get("revision_rate", 0.0)
        if rev_rate > 0.5:
            score -= 0.3
        elif rev_rate > 0.3:
            score -= 0.2
        elif rev_rate > 0.15:
            score -= 0.1

        # High contradiction rate is the most serious problem
        contr_rate = metrics.get("contradiction_rate", 0.0)
        if contr_rate > 0.2:
            score -= 0.5
        elif contr_rate > 0.1:
            score -= 0.4
        elif contr_rate > 0.05:
            score -= 0.2

        # Low predictive validity
        pv = metrics.get("predictive_validity")
        if pv is not None:
            if pv < 0.1:
                score -= 0.3
            elif pv < 0.3:
                score -= 0.15

        # Low benchmark correlation
        bc = metrics.get("benchmark_correlation")
        if bc is not None:
            if bc < 0.0:
                score -= 0.4  # Negative correlation is very bad
            elif bc < 0.2:
                score -= 0.3

        # Low self-consistency
        sc = metrics.get("self_consistency")
        if sc is not None and sc < 0.7:
            score -= 0.15

        return max(0.0, min(1.0, score))

    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on the current metrics."""
        recs: List[Dict[str, Any]] = []

        # Contradiction rate
        if metrics.get("contradiction_rate", 0.0) > 0.1:
            recs.append({
                "severity": "high",
                "issue": "High contradiction rate",
                "recommendation": (
                    "Increase the cooling-off period before training on new "
                    "signals.  Current signals are being trained on before "
                    "enough delayed feedback arrives."
                ),
                "action": "increase_cooling_period",
                "suggested_value": "24 hours",
            })

        # Revision rate
        if metrics.get("revision_rate", 0.0) > 0.3:
            recs.append({
                "severity": "medium",
                "issue": "High revision rate",
                "recommendation": (
                    "Initial reward scoring is unreliable.  Consider "
                    "increasing the active inquiry budget to get clearer "
                    "signals upfront, or raising the stability threshold "
                    "for training readiness."
                ),
                "action": "increase_inquiry_budget",
                "suggested_value": "15 per day",
            })

        # Benchmark correlation
        bc = metrics.get("benchmark_correlation")
        if bc is not None and bc < 0.2:
            severity = "critical" if bc < 0.0 else "high"
            recs.append({
                "severity": severity,
                "issue": "Low benchmark correlation",
                "recommendation": (
                    "Reward signals are not correlated with actual quality "
                    "improvements.  The reward dimensions or their weights "
                    "may need fundamental restructuring."
                ),
                "action": "recalibrate_reward_weights",
                "suggested_value": None,
            })

        # Predictive validity
        pv = metrics.get("predictive_validity")
        if pv is not None and pv < 0.3:
            recs.append({
                "severity": "high",
                "issue": "Low predictive validity",
                "recommendation": (
                    "High-reward experiences do not predict better future "
                    "performance.  Consider recalibrating self-evaluation "
                    "via calibrate_self_eval() or increasing the weight of "
                    "explicit feedback signals."
                ),
                "action": "recalibrate_self_eval",
                "suggested_value": None,
            })

        # Self-consistency
        sc = metrics.get("self_consistency")
        if sc is not None and sc < 0.7:
            recs.append({
                "severity": "medium",
                "issue": "Low self-consistency",
                "recommendation": (
                    "Reward scores vary significantly when the same "
                    "experience is re-evaluated.  This may indicate "
                    "non-determinism in the self-eval prompt or model "
                    "temperature settings."
                ),
                "action": "stabilise_self_eval",
                "suggested_value": "temperature=0.0",
            })

        # Good health
        health = metrics.get("reward_system_health", 0.0)
        if health >= 0.8 and not recs:
            recs.append({
                "severity": "info",
                "issue": "Reward system healthy",
                "recommendation": "No action needed.  All metrics within acceptable ranges.",
                "action": "none",
                "suggested_value": None,
            })

        return recs

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_events(temporal_tracker: Any) -> List[Any]:
        """Extract event objects from a temporal tracker.

        Supports both the TemporalRewardTracker (which stores events in a
        JSONL file and exposes ``_load_all()``) and simpler dict-based
        trackers that expose an ``active_rewards`` mapping.
        """
        # Prefer _load_all() (our TemporalRewardTracker)
        if hasattr(temporal_tracker, "_load_all"):
            return temporal_tracker._load_all()
        # Fall back to active_rewards dict
        active = getattr(temporal_tracker, "active_rewards", None)
        if active is not None and isinstance(active, dict):
            return list(active.values())
        return []

    @staticmethod
    def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
        """Get an attribute from a dataclass/object or dict transparently."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
