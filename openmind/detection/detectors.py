"""Signal quality detection - the central gatekeeper for learning decisions."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class SignalVerdict(Enum):
    """The four-way decision for each experience."""

    TRAIN = "train"
    SHELVE = "shelve"
    ASK = "ask"
    DISCARD = "discard"


class DetectionFlag(Enum):
    """Specific issues detected in a signal."""

    CLEAN = "clean"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_DIMENSIONS = "conflicting_dimensions"
    REWARD_GAMING = "reward_gaming"
    DISTRIBUTION_SHIFT = "distribution_shift"
    NOVEL_TERRITORY = "novel_territory"
    STALE_REFERENCE = "stale_reference"
    SELF_REINFORCING = "self_reinforcing"
    OUTLIER = "outlier"
    SPARSE_SIGNAL = "sparse_signal"


@dataclass
class DetectionResult:
    """Full diagnostic on a single experience."""

    experience_id: str
    timestamp: float
    verdict: SignalVerdict
    flags: List[DetectionFlag]
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)

    suggested_inquiry: Optional[Dict[str, str]] = None
    training_weight: float = 1.0
    shelve_reason: Optional[str] = None
    revisit_after_days: Optional[float] = None


class SignalQualityDetector:
    """
    The central gatekeeper. Evaluates every experience + reward signal
    pair and determines what to do with it.

    Runs 8 detectors, each looking for a specific failure mode,
    then synthesizes their outputs into a verdict.
    """

    def __init__(
        self,
        experience_buffer: Any = None,
        knowledge_registry: Any = None,
        reward_history_window: int = 500,
    ) -> None:
        self.exp_buffer = experience_buffer
        self.knowledge_registry = knowledge_registry

        self.reward_history: List[float] = []
        self.reward_history_window = reward_history_window
        self.domain_baselines: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"mean": 0.0, "std": 0.5, "n": 0}
        )
        self.detector_performance: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            }
        )

    def evaluate(self, experience: Any, reward_signal: Any) -> DetectionResult:
        """Run all detectors and synthesize a verdict."""
        flags: List[DetectionFlag] = []
        details: Dict[str, Any] = {}

        # DETECTOR 1: Dimensional conflict
        conflict = self._detect_dimensional_conflict(reward_signal)
        if conflict["detected"]:
            flags.append(DetectionFlag.CONFLICTING_DIMENSIONS)
            details["dimensional_conflict"] = conflict

        # DETECTOR 2: Statistical outlier
        outlier = self._detect_outlier(experience, reward_signal)
        if outlier["detected"]:
            flags.append(DetectionFlag.OUTLIER)
            details["outlier"] = outlier

        # DETECTOR 3: Reward gaming
        gaming = self._detect_reward_gaming(experience, reward_signal)
        if gaming["detected"]:
            flags.append(DetectionFlag.REWARD_GAMING)
            details["reward_gaming"] = gaming

        # DETECTOR 4: Novelty
        novelty = self._detect_novelty(experience)
        if novelty["detected"]:
            flags.append(DetectionFlag.NOVEL_TERRITORY)
            details["novelty"] = novelty

        # DETECTOR 5: Distribution shift
        shift = self._detect_distribution_shift(experience)
        if shift["detected"]:
            flags.append(DetectionFlag.DISTRIBUTION_SHIFT)
            details["distribution_shift"] = shift

        # DETECTOR 6: Self-reinforcing loop
        loop = self._detect_self_reinforcement(experience, reward_signal)
        if loop["detected"]:
            flags.append(DetectionFlag.SELF_REINFORCING)
            details["self_reinforcement"] = loop

        # DETECTOR 7: Signal sparsity
        sparse = self._detect_signal_sparsity(reward_signal)
        if sparse["detected"]:
            flags.append(DetectionFlag.SPARSE_SIGNAL)
            details["sparsity"] = sparse

        # DETECTOR 8: Stale reference
        stale = self._detect_stale_reference(experience, reward_signal)
        if stale["detected"]:
            flags.append(DetectionFlag.STALE_REFERENCE)
            details["stale_reference"] = stale

        if not flags:
            flags.append(DetectionFlag.CLEAN)

        verdict, confidence, training_weight = self._synthesize(
            flags, details, reward_signal
        )

        result = DetectionResult(
            experience_id=getattr(experience, "experience_id", "unknown"),
            timestamp=time.time(),
            verdict=verdict,
            flags=flags,
            confidence=confidence,
            details=details,
            training_weight=training_weight,
        )

        if verdict == SignalVerdict.ASK:
            result.suggested_inquiry = self._generate_inquiry_from_flags(
                experience, flags, details
            )

        if verdict == SignalVerdict.SHELVE:
            result.shelve_reason = self._explain_shelve(flags, details)
            result.revisit_after_days = self._estimate_revisit_time(flags, details)

        self._update_baselines(experience, reward_signal)
        return result

    # === INDIVIDUAL DETECTORS ===

    def _detect_dimensional_conflict(self, reward_signal: Any) -> Dict[str, Any]:
        """Check for contradictory reward dimensions."""
        measured = reward_signal._get_measured_values()
        if len(measured) < 2:
            return {"detected": False}

        conflicts: List[Dict] = []

        sat = measured.get("user_satisfaction")
        ground = measured.get("groundedness")
        task = measured.get("task_success")
        coh = measured.get("coherence")

        # User liked it but it was wrong
        if sat is not None and ground is not None and sat > 0.5 and ground < -0.3:
            conflicts.append({
                "type": "satisfied_but_wrong",
                "severity": "high",
                "dimensions": {"user_satisfaction": sat, "groundedness": ground},
            })

        # Correct but user unhappy
        if task is not None and sat is not None and task > 0.5 and sat < -0.3:
            conflicts.append({
                "type": "correct_but_disliked",
                "severity": "medium",
                "dimensions": {"task_success": task, "user_satisfaction": sat},
            })

        # Eloquent but useless
        if coh is not None and task is not None and coh > 0.7 and task < -0.3:
            conflicts.append({
                "type": "eloquent_but_useless",
                "severity": "medium",
                "dimensions": {"coherence": coh, "task_success": task},
            })

        # General disagreement
        values = list(measured.values())
        has_pos = any(v > 0.3 for v in values)
        has_neg = any(v < -0.3 for v in values)
        if has_pos and has_neg and not conflicts:
            conflicts.append({
                "type": "general_disagreement",
                "severity": "low",
                "dimensions": measured,
            })

        return {
            "detected": len(conflicts) > 0,
            "conflicts": conflicts,
            "max_severity": max((c["severity"] for c in conflicts), default=None),
        }

    def _detect_outlier(self, experience: Any, reward_signal: Any) -> Dict[str, Any]:
        """Is this reward statistically unusual for this domain?"""
        domains = getattr(experience, "domain_tags", ["general"])
        aggregate = reward_signal.aggregate

        outlier_scores: List[Dict] = []
        for domain in domains:
            baseline = self.domain_baselines[domain]
            if baseline["n"] < 10 or baseline["std"] < 0.01:
                continue

            z_score = abs((aggregate - baseline["mean"]) / baseline["std"])
            outlier_scores.append({
                "domain": domain,
                "z_score": float(z_score),
                "baseline_mean": baseline["mean"],
                "baseline_std": baseline["std"],
            })

        is_outlier = any(o["z_score"] > 2.5 for o in outlier_scores)
        return {
            "detected": is_outlier,
            "domain_scores": outlier_scores,
            "max_z": max((o["z_score"] for o in outlier_scores), default=0.0),
        }

    def _detect_reward_gaming(
        self, experience: Any, reward_signal: Any
    ) -> Dict[str, Any]:
        """Is the model learning to game its reward signal?"""
        recent = self._get_recent_experiences(50)
        if len(recent) < 20:
            return {"detected": False}

        diversity = self._compute_output_diversity(recent)
        reward_trend = self._compute_trend(
            [e.get("reward_signal", 0) if isinstance(e, dict) else getattr(e, "reward_signal", 0) for e in recent]
        )
        diversity_trend = self._compute_trend(diversity)

        gaming_suspected = reward_trend >= 0 and diversity_trend < -0.1
        template_score = self._detect_template_outputs(recent)

        return {
            "detected": gaming_suspected or template_score > 0.7,
            "diversity_trend": float(diversity_trend),
            "reward_trend": float(reward_trend),
            "template_score": float(template_score),
        }

    def _detect_novelty(self, experience: Any) -> Dict[str, Any]:
        """Is this experience in unfamiliar territory?"""
        domains = getattr(experience, "domain_tags", [])
        known = set(self.domain_baselines.keys())
        novel_domains = [d for d in domains if d not in known]
        domain_novelty = len(novel_domains) / max(len(domains), 1)

        novelty_score = domain_novelty
        return {
            "detected": novelty_score > 0.6,
            "novelty_score": float(novelty_score),
            "novel_domains": novel_domains,
        }

    def _detect_distribution_shift(self, experience: Any) -> Dict[str, Any]:
        """Has the pattern of incoming experiences changed?"""
        recent = self._get_recent_experiences(100)
        historical = self._get_historical_experiences(100)
        if len(recent) < 50 or len(historical) < 50:
            return {"detected": False}

        recent_domains: Dict[str, int] = defaultdict(int)
        hist_domains: Dict[str, int] = defaultdict(int)

        for exp in recent:
            for tag in (exp.get("domain_tags", []) if isinstance(exp, dict) else getattr(exp, "domain_tags", [])):
                recent_domains[tag] += 1
        for exp in historical:
            for tag in (exp.get("domain_tags", []) if isinstance(exp, dict) else getattr(exp, "domain_tags", [])):
                hist_domains[tag] += 1

        recent_total = sum(recent_domains.values()) or 1
        hist_total = sum(hist_domains.values()) or 1
        all_doms = set(list(recent_domains.keys()) + list(hist_domains.keys()))

        kl_div = 0.0
        for domain in all_doms:
            p = max(recent_domains.get(domain, 0) / recent_total, 1e-6)
            q = max(hist_domains.get(domain, 0) / hist_total, 1e-6)
            kl_div += p * np.log(p / q)

        recent_rewards = [
            (e.get("reward_signal", 0) if isinstance(e, dict) else getattr(e, "reward_signal", 0))
            for e in recent
        ]
        hist_rewards = [
            (e.get("reward_signal", 0) if isinstance(e, dict) else getattr(e, "reward_signal", 0))
            for e in historical
        ]
        reward_shift = abs(float(np.mean(recent_rewards)) - float(np.mean(hist_rewards)))

        return {
            "detected": kl_div > 0.5 or reward_shift > 0.3,
            "domain_divergence": float(kl_div),
            "reward_shift": float(reward_shift),
        }

    def _detect_self_reinforcement(
        self, experience: Any, reward_signal: Any
    ) -> Dict[str, Any]:
        """Is the model converging on its own outputs?"""
        output = getattr(experience, "output", "")
        trained_outputs = self._get_trained_outputs(
            getattr(experience, "domain_tags", []), n=20
        )
        if not trained_outputs:
            return {"detected": False}

        similarities = [self._text_similarity(output, prev) for prev in trained_outputs]
        max_sim = max(similarities) if similarities else 0
        avg_sim = float(np.mean(similarities)) if similarities else 0
        sim_trend = self._compute_trend(similarities)

        is_reinforcing = (
            avg_sim > 0.8
            and reward_signal.aggregate > 0.3
            and sim_trend > 0
        )
        return {
            "detected": is_reinforcing,
            "avg_similarity": float(avg_sim),
            "max_similarity": float(max_sim),
            "convergence_trend": float(sim_trend),
        }

    def _detect_signal_sparsity(self, reward_signal: Any) -> Dict[str, Any]:
        """Do we have enough reward dimensions measured?"""
        measured = reward_signal._get_measured_values()
        coverage = len(measured) / 5.0

        critical_missing = []
        if "task_success" not in measured:
            critical_missing.append("task_success")
        if "groundedness" not in measured:
            critical_missing.append("groundedness")

        return {
            "detected": coverage < 0.4 or len(critical_missing) > 1,
            "coverage": coverage,
            "critical_missing": critical_missing,
        }

    def _detect_stale_reference(
        self, experience: Any, reward_signal: Any
    ) -> Dict[str, Any]:
        """Is the reward evaluated against outdated knowledge?"""
        if self.knowledge_registry is None:
            return {"detected": False}

        domains = getattr(experience, "domain_tags", [])
        stale: List[Dict] = []

        for domain in domains:
            knowledge_ids = getattr(self.knowledge_registry, "domain_index", {}).get(
                domain, []
            )
            for kid in knowledge_ids:
                k = self.knowledge_registry.knowledge.get(kid)
                if k is None:
                    continue
                status = (
                    k.get("status") if isinstance(k, dict) else getattr(k, "status", "active")
                )
                if status in ("under_review", "deprecated"):
                    stale.append({"knowledge_id": kid, "status": status, "domain": domain})

        return {"detected": len(stale) > 0, "stale_references": stale}

    # === SYNTHESIS ===

    def _synthesize(
        self,
        flags: List[DetectionFlag],
        details: Dict[str, Any],
        reward_signal: Any,
    ) -> Tuple[SignalVerdict, float, float]:
        """Combine detector outputs into a verdict."""
        if DetectionFlag.REWARD_GAMING in flags:
            return SignalVerdict.DISCARD, 0.9, 0.0

        if DetectionFlag.SELF_REINFORCING in flags:
            return SignalVerdict.SHELVE, 0.8, 0.0

        high_severity = {
            DetectionFlag.CONFLICTING_DIMENSIONS,
            DetectionFlag.REWARD_GAMING,
            DetectionFlag.SELF_REINFORCING,
        }
        medium_severity = {
            DetectionFlag.OUTLIER,
            DetectionFlag.DISTRIBUTION_SHIFT,
            DetectionFlag.STALE_REFERENCE,
        }
        low_severity = {DetectionFlag.NOVEL_TERRITORY, DetectionFlag.SPARSE_SIGNAL}

        high_count = sum(1 for f in flags if f in high_severity)
        medium_count = sum(1 for f in flags if f in medium_severity)
        low_count = sum(1 for f in flags if f in low_severity)

        severity_score = high_count * 0.4 + medium_count * 0.2 + low_count * 0.1
        signal_confidence = reward_signal.confidence
        overall_confidence = max(0.0, signal_confidence - severity_score)

        if overall_confidence > 0.6:
            weight = 0.5 + overall_confidence * 0.5
            if DetectionFlag.NOVEL_TERRITORY in flags and high_count == 0:
                weight *= 1.3
            return SignalVerdict.TRAIN, overall_confidence, min(weight, 2.0)
        elif overall_confidence > 0.3:
            return SignalVerdict.ASK, overall_confidence, 0.0
        else:
            return SignalVerdict.SHELVE, overall_confidence, 0.0

    def _generate_inquiry_from_flags(
        self, experience: Any, flags: List[DetectionFlag], details: Dict
    ) -> Dict[str, str]:
        """Generate a targeted question from detection flags."""
        if DetectionFlag.CONFLICTING_DIMENSIONS in flags:
            conflicts = details.get("dimensional_conflict", {}).get("conflicts", [])
            if conflicts:
                ctype = conflicts[0]["type"]
                if ctype == "satisfied_but_wrong":
                    return {
                        "type": "correctness",
                        "question": "I want to double check - was my earlier response actually accurate?",
                    }
                elif ctype == "correct_but_disliked":
                    return {
                        "type": "preference",
                        "question": "I think the info was right but maybe the delivery wasn't great. Would a different approach have been more helpful?",
                    }

        if DetectionFlag.NOVEL_TERRITORY in flags:
            return {
                "type": "importance",
                "question": "This seems like a new area for us. Is this something that comes up often?",
            }

        if DetectionFlag.SPARSE_SIGNAL in flags:
            missing = details.get("sparsity", {}).get("critical_missing", [])
            if "groundedness" in missing:
                return {
                    "type": "correctness",
                    "question": "Quick accuracy check - did everything check out factually?",
                }

        return {
            "type": "usefulness",
            "question": "Was that actually helpful, or should I have approached it differently?",
        }

    def _explain_shelve(self, flags: List[DetectionFlag], details: Dict) -> str:
        explanations = {
            DetectionFlag.SELF_REINFORCING: "Potential self-reinforcing feedback loop",
            DetectionFlag.CONFLICTING_DIMENSIONS: "Reward dimensions contradict each other",
            DetectionFlag.OUTLIER: "Statistically unusual reward pattern",
            DetectionFlag.DISTRIBUTION_SHIFT: "Input distribution has shifted",
            DetectionFlag.STALE_REFERENCE: "Evaluated against outdated knowledge",
            DetectionFlag.SPARSE_SIGNAL: "Insufficient reward dimensions measured",
        }
        reasons = [explanations[f] for f in flags if f in explanations]
        return "; ".join(reasons) if reasons else "Unknown"

    def _estimate_revisit_time(self, flags: List[DetectionFlag], details: Dict) -> float:
        if DetectionFlag.STALE_REFERENCE in flags:
            return 3.0
        if DetectionFlag.DISTRIBUTION_SHIFT in flags:
            return 7.0
        if DetectionFlag.NOVEL_TERRITORY in flags:
            return 5.0
        return 14.0

    # === HELPERS ===

    def _update_baselines(self, experience: Any, reward_signal: Any) -> None:
        domains = getattr(experience, "domain_tags", ["general"])
        aggregate = reward_signal.aggregate

        self.reward_history.append(aggregate)
        if len(self.reward_history) > self.reward_history_window:
            self.reward_history.pop(0)

        for domain in domains:
            b = self.domain_baselines[domain]
            b["n"] += 1
            n = b["n"]
            old_mean = b["mean"]
            b["mean"] = old_mean + (aggregate - old_mean) / n
            if n > 1:
                b["std"] = float(np.sqrt(
                    ((n - 2) / (n - 1)) * b["std"] ** 2
                    + (aggregate - old_mean) ** 2 / n
                ))

    def _compute_output_diversity(self, experiences: list) -> List[float]:
        outputs = [
            (e.get("output", "") if isinstance(e, dict) else getattr(e, "output", ""))
            for e in experiences
        ]
        diversities = []
        for i in range(5, len(outputs)):
            window = outputs[i - 5 : i]
            vocab: set = set()
            for o in window:
                vocab.update(o.lower().split())
            diversities.append(float(len(vocab)))
        return diversities if diversities else [0.0]

    def _compute_trend(self, values: list) -> float:
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values, dtype=float)
        try:
            return float(np.polyfit(x, y, 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _detect_template_outputs(self, experiences: list) -> float:
        outputs = [
            (e.get("output", "") if isinstance(e, dict) else getattr(e, "output", ""))
            for e in experiences
        ]
        if len(outputs) < 5:
            return 0.0
        # Check first-sentence similarity as template indicator
        first_sentences = []
        for o in outputs:
            sentences = o.split(".")
            if sentences:
                first_sentences.append(sentences[0].strip().lower())
        if not first_sentences:
            return 0.0
        unique_ratio = len(set(first_sentences)) / len(first_sentences)
        return 1.0 - unique_ratio

    def _get_recent_experiences(self, n: int = 50) -> list:
        if self.exp_buffer is None:
            return []
        if hasattr(self.exp_buffer, "get_recent"):
            return self.exp_buffer.get_recent(n)
        return []

    def _get_historical_experiences(self, n: int = 100) -> list:
        if self.exp_buffer is None:
            return []
        if hasattr(self.exp_buffer, "get_historical"):
            return self.exp_buffer.get_historical(n)
        return []

    def _get_trained_outputs(self, domain_tags: list, n: int = 20) -> List[str]:
        if self.exp_buffer is None:
            return []
        recent = self._get_recent_experiences(n * 2)
        outputs = []
        for e in recent:
            tags = (
                e.get("domain_tags", []) if isinstance(e, dict) else getattr(e, "domain_tags", [])
            )
            if any(t in domain_tags for t in tags):
                output = (
                    e.get("output", "") if isinstance(e, dict) else getattr(e, "output", "")
                )
                if output:
                    outputs.append(output)
            if len(outputs) >= n:
                break
        return outputs
