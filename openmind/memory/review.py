"""Belief reviewer - the 'Pythagoras checker' for periodic knowledge re-examination."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np


class BeliefReviewer:
    """
    Periodically re-examines promoted knowledge to check whether
    it still holds up against current evidence and model state.

    Five tests:
    1. Behavioral consistency - does the model still act on this belief?
    2. Evidence review - does recent evidence support or contradict?
    3. Contradiction scan - does newer knowledge logically conflict?
    4. Counterfactual probe - would the model perform better without it?
    5. Relevance drift - is this domain still active?
    """

    def __init__(
        self,
        model: Any = None,
        experience_buffer: Any = None,
        reward_collector: Any = None,
        knowledge_registry: Any = None,
    ) -> None:
        self.model = model
        self.exp_buffer = experience_buffer
        self.reward_collector = reward_collector
        self.registry = knowledge_registry

    def review(self, knowledge_id: str) -> Dict[str, Any]:
        """Run comprehensive review of a piece of promoted knowledge."""
        knowledge = self.registry.knowledge[knowledge_id]

        belief = (
            knowledge.get("belief_summary", "")
            if isinstance(knowledge, dict)
            else getattr(knowledge, "belief_summary", "")
        )

        results: Dict[str, Any] = {
            "knowledge_id": knowledge_id,
            "review_timestamp": time.time(),
            "tests": {},
        }

        results["tests"]["behavioral_consistency"] = (
            self._test_behavioral_consistency(knowledge, belief)
        )
        results["tests"]["evidence_review"] = (
            self._test_against_recent_evidence(knowledge, belief)
        )
        results["tests"]["contradiction_scan"] = (
            self._scan_for_contradictions(knowledge, belief)
        )
        results["tests"]["counterfactual"] = (
            self._counterfactual_test(knowledge)
        )
        results["tests"]["relevance_drift"] = (
            self._test_relevance_drift(knowledge)
        )

        results["verdict"] = self._synthesize_verdict(results["tests"])
        self._apply_verdict(knowledge_id, results)

        return results

    def _test_behavioral_consistency(
        self, knowledge: Any, belief: str
    ) -> Dict[str, Any]:
        """Does the model's current behavior still reflect this belief?"""
        if self.model is None:
            return {"result": "no_model", "score": 0.5}

        # TODO: Generate test inputs from belief via model call and check alignment
        return {
            "result": "not_implemented",
            "score": 0.5,
            "note": "Requires model inference to generate and test behavioral probes",
        }

    def _test_against_recent_evidence(
        self, knowledge: Any, belief: str
    ) -> Dict[str, Any]:
        """Do recent experiences in the same domain support or contradict?"""
        if self.exp_buffer is None:
            return {"result": "no_buffer", "score": 0.5}

        domains = (
            knowledge.get("source_domain", [])
            if isinstance(knowledge, dict)
            else getattr(knowledge, "source_domain", [])
        )

        recent: list = []
        if hasattr(self.exp_buffer, "get_by_domain"):
            recent = self.exp_buffer.get_by_domain(domains, days_back=30)
        elif hasattr(self.exp_buffer, "get_recent"):
            all_recent = self.exp_buffer.get_recent(100)
            recent = [
                e for e in all_recent
                if any(
                    tag in (e.get("domain_tags", []) if isinstance(e, dict)
                            else getattr(e, "domain_tags", []))
                    for tag in domains
                )
            ]

        if not recent:
            return {"result": "insufficient_data", "score": 0.5}

        positive_count = 0
        negative_count = 0
        for exp in recent:
            reward = (
                exp.get("reward_signal", 0)
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", 0)
            )
            if reward > 0.3:
                positive_count += 1
            elif reward < -0.3:
                negative_count += 1

        total = positive_count + negative_count
        if total == 0:
            return {"result": "no_signal", "score": 0.5}

        support_ratio = positive_count / total
        return {
            "result": "evaluated",
            "supporting": positive_count,
            "contradicting": negative_count,
            "support_ratio": float(support_ratio),
            "score": float(support_ratio),
        }

    def _scan_for_contradictions(
        self, knowledge: Any, belief: str
    ) -> Dict[str, Any]:
        """Check if newer promoted knowledge logically contradicts this belief."""
        if self.registry is None:
            return {"result": "no_registry", "score": 0.5}

        promoted_at = (
            knowledge.get("promoted_at", 0)
            if isinstance(knowledge, dict)
            else getattr(knowledge, "promoted_at", 0)
        )

        contradictions = []
        for k in self.registry.knowledge.values():
            k_promoted = (
                k.get("promoted_at", 0)
                if isinstance(k, dict)
                else getattr(k, "promoted_at", 0)
            )
            if k_promoted <= promoted_at:
                continue

            newer_belief = (
                k.get("belief_summary", "")
                if isinstance(k, dict)
                else getattr(k, "belief_summary", "")
            )
            newer_domains = set(
                k.get("source_domain", [])
                if isinstance(k, dict)
                else getattr(k, "source_domain", [])
            )
            current_domains = set(
                knowledge.get("source_domain", [])
                if isinstance(knowledge, dict)
                else getattr(knowledge, "source_domain", [])
            )

            overlap = newer_domains & current_domains
            if overlap and newer_belief and belief:
                contradictions.append({
                    "conflicting_knowledge": (
                        k.get("knowledge_id", "unknown")
                        if isinstance(k, dict)
                        else getattr(k, "knowledge_id", "unknown")
                    ),
                    "conflicting_belief": newer_belief,
                    "overlapping_domains": list(overlap),
                    "conflict_type": "potential_overlap",
                })

        return {
            "contradictions_found": len(contradictions),
            "details": contradictions,
            "score": max(0.0, 1.0 - len(contradictions) * 0.3),
        }

    def _counterfactual_test(self, knowledge: Any) -> Dict[str, Any]:
        """Would the model perform better without this knowledge?"""
        return {
            "result": "not_run",
            "score": 0.5,
            "note": "Counterfactual testing requires model inference pipeline",
        }

    def _test_relevance_drift(self, knowledge: Any) -> Dict[str, Any]:
        """Is this domain still active?"""
        if self.exp_buffer is None:
            return {"result": "no_buffer", "score": 0.5}

        domains = (
            knowledge.get("source_domain", [])
            if isinstance(knowledge, dict)
            else getattr(knowledge, "source_domain", [])
        )

        recent: list = []
        if hasattr(self.exp_buffer, "get_recent"):
            recent = self.exp_buffer.get_recent(200)

        if not recent:
            return {"result": "insufficient_data", "score": 0.5}

        mid = len(recent) // 2
        recent_half = recent[:mid]
        older_half = recent[mid:]

        def count_domain(experiences: list, dtags: list) -> int:
            count = 0
            for exp in experiences:
                tags = (
                    exp.get("domain_tags", [])
                    if isinstance(exp, dict)
                    else getattr(exp, "domain_tags", [])
                )
                if any(t in dtags for t in tags):
                    count += 1
            return count

        recent_count = count_domain(recent_half, domains)
        older_count = count_domain(older_half, domains)

        if older_count == 0:
            return {"result": "no_baseline", "score": 0.5}

        drift_ratio = recent_count / (older_count + 1e-8)
        return {
            "result": "evaluated",
            "recent_frequency": recent_count,
            "historical_frequency": older_count,
            "drift_ratio": float(drift_ratio),
            "still_relevant": drift_ratio > 0.1,
            "score": float(min(drift_ratio, 1.0)),
        }

    def _synthesize_verdict(self, tests: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine all test results into a final verdict."""
        scores = [
            t.get("score", 0.5)
            for t in tests.values()
            if t.get("score") is not None
        ]

        if not scores:
            return {"action": "defer", "reason": "insufficient test data"}

        avg_score = float(np.mean(scores))
        min_score = float(min(scores))

        if min_score < 0.2:
            return {
                "action": "deprecate_and_correct",
                "reason": "critical failure in at least one test",
                "confidence": 1.0 - min_score,
                "failed_tests": [
                    name for name, t in tests.items()
                    if t.get("score", 1.0) < 0.2
                ],
            }

        if avg_score < 0.4:
            return {
                "action": "revise",
                "reason": "multiple tests show degradation",
                "confidence": 1.0 - avg_score,
            }

        if avg_score < 0.6:
            return {
                "action": "flag_for_inquiry",
                "reason": "mixed results, needs active clarification",
                "confidence": 0.5,
            }

        return {
            "action": "reinforce",
            "reason": "knowledge still valid",
            "confidence": avg_score,
        }

    def _apply_verdict(self, knowledge_id: str, results: Dict[str, Any]) -> None:
        """Act on the review verdict."""
        verdict = results["verdict"]
        knowledge = self.registry.knowledge[knowledge_id]
        action = verdict["action"]

        if isinstance(knowledge, dict):
            knowledge["review_count"] = knowledge.get("review_count", 0) + 1
            knowledge["last_reviewed"] = time.time()
        else:
            knowledge.review_count = getattr(knowledge, "review_count", 0) + 1
            knowledge.last_reviewed = time.time()

        if action == "reinforce":
            if isinstance(knowledge, dict):
                knowledge["current_confidence"] = min(
                    1.0, knowledge.get("current_confidence", 0.8) + 0.05
                )
                knowledge["status"] = "active"
            else:
                knowledge.current_confidence = min(1.0, knowledge.current_confidence + 0.05)
                knowledge.status = "active"

        elif action == "revise":
            if isinstance(knowledge, dict):
                knowledge["status"] = "under_review"
                knowledge["current_confidence"] = max(
                    0.2, knowledge.get("current_confidence", 0.8) - 0.2
                )
            else:
                knowledge.status = "under_review"
                knowledge.current_confidence = max(0.2, knowledge.current_confidence - 0.2)

        elif action == "deprecate_and_correct":
            if isinstance(knowledge, dict):
                knowledge["status"] = "deprecated"
                knowledge["current_confidence"] = 0.0
            else:
                knowledge.status = "deprecated"
                knowledge.current_confidence = 0.0

        elif action == "flag_for_inquiry":
            if isinstance(knowledge, dict):
                knowledge["status"] = "under_review"
            else:
                knowledge.status = "under_review"

        review_record = {
            "timestamp": time.time(),
            "verdict": verdict,
            "test_scores": {
                name: t.get("score") for name, t in results["tests"].items()
            },
        }
        if isinstance(knowledge, dict):
            knowledge.setdefault("review_history", []).append(review_record)
        else:
            if not hasattr(knowledge, "review_history"):
                knowledge.review_history = []
            knowledge.review_history.append(review_record)
