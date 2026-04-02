"""Active inquiry system - the 'curiosity engine' for seeking clarification."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class InquiryPriority(Enum):
    """How urgently does this need clarification?"""

    CRITICAL = "critical"  # could cause harm if learned wrong
    HIGH = "high"  # high novelty + high potential value
    MODERATE = "moderate"  # useful to know but can wait
    LOW = "low"  # mild curiosity, ask only if natural opening


class InquiryType(Enum):
    """What kind of clarification do we need?"""

    CORRECTNESS = "correctness"  # was the information accurate?
    USEFULNESS = "usefulness"  # did this actually help?
    PREFERENCE = "preference"  # right approach/tone/depth?
    IMPORTANCE = "importance"  # does this topic matter?
    CONTRADICTION = "contradiction"  # conflicting signals
    BOUNDARY = "boundary"  # overstepping or understepping?


@dataclass
class ClarificationInquiry:
    """A pending or resolved clarification request."""

    inquiry_id: str
    experience_id: str
    timestamp: float

    inquiry_type: InquiryType
    priority: InquiryPriority

    question_text: str
    reasoning: str

    target_dimensions: List[str] = field(default_factory=list)
    known_signals: Dict[str, float] = field(default_factory=dict)
    ambiguous_signals: Dict[str, str] = field(default_factory=dict)

    asked: bool = False
    asked_at: Optional[float] = None
    response: Optional[str] = None
    resolved_signal: Optional[Dict[str, Optional[float]]] = None


# Question templates by inquiry type
_QUESTION_TEMPLATES: Dict[InquiryType, List[str]] = {
    InquiryType.CORRECTNESS: [
        "I gave you some specifics on {topic} earlier - did that check out or was anything off?",
        "Want to flag something: I wasn't fully confident in my answer about {topic}. Worth double-checking if accuracy matters here.",
    ],
    InquiryType.USEFULNESS: [
        "Did that actually help with what you were working on, or did I miss the mark?",
        "Just want to make sure that was useful and not just noise.",
    ],
    InquiryType.PREFERENCE: [
        "Was that the right level of detail, or would you have preferred I went deeper or kept it shorter?",
        "Tone check - was that framing helpful or would a different approach work better?",
    ],
    InquiryType.IMPORTANCE: [
        "Is {topic} something that comes up a lot for you? Trying to gauge how much to prioritize getting this domain right.",
        "Should I treat {topic} as a recurring area of focus?",
    ],
    InquiryType.CONTRADICTION: [
        "I've gotten mixed signals on {topic}. Last time it seemed like one thing but now it feels different. Which is closer?",
        "Quick clarification on {topic} - I want to make sure I'm calibrated correctly.",
    ],
    InquiryType.BOUNDARY: [
        "Was that too much detail or about right?",
        "Let me know if I'm overcomplicating things or if this depth is useful.",
    ],
}


class ActiveInquirySystem:
    """
    The curiosity engine. Decides when the system should actively
    seek feedback rather than guessing or shelving.

    Core philosophy: asking is a resource. Spend it where the
    expected information gain is highest.
    """

    def __init__(
        self,
        ask_budget_per_session: int = 3,
        ask_budget_per_day: int = 10,
        cooldown_seconds: float = 120.0,
    ) -> None:
        self.pending_inquiries: List[ClarificationInquiry] = []
        self.inquiry_history: List[ClarificationInquiry] = []

        self.session_asks_remaining = ask_budget_per_session
        self.daily_asks_remaining = ask_budget_per_day
        self.cooldown_seconds = cooldown_seconds
        self.last_ask_time: float = 0.0

        # Tracks which types of questions yield useful responses
        self.inquiry_effectiveness: Dict[InquiryType, Dict[str, Any]] = {
            itype: {"asked": 0, "useful_response": 0, "effectiveness": 0.5}
            for itype in InquiryType
        }

        # Domain frequency tracking
        self._domain_counts: Dict[str, int] = {}
        self._total_experiences: int = 0

    def evaluate_for_inquiry(
        self,
        experience: Any,
        reward_signal: Any,
        model_state: Optional[Dict] = None,
    ) -> Optional[ClarificationInquiry]:
        """Decide whether to ask for clarification on this experience."""
        gap_analysis = self._analyze_information_gap(experience, reward_signal)

        if gap_analysis["expected_value"] < 0.3:
            return None

        if not self._can_ask():
            if gap_analysis["priority"] == InquiryPriority.CRITICAL:
                inquiry = self._formulate_inquiry(
                    experience, reward_signal, gap_analysis
                )
                self.pending_inquiries.append(inquiry)
            return None

        inquiry = self._formulate_inquiry(experience, reward_signal, gap_analysis)
        inquiry.asked = True
        inquiry.asked_at = time.time()
        self.pending_inquiries.append(inquiry)
        self.session_asks_remaining -= 1
        self.daily_asks_remaining -= 1
        self.last_ask_time = time.time()
        self.inquiry_effectiveness[inquiry.inquiry_type]["asked"] += 1
        return inquiry

    def _analyze_information_gap(
        self, experience: Any, reward_signal: Any
    ) -> Dict[str, Any]:
        """Determine the value of getting clarification."""
        importance = self._assess_topic_importance(experience)
        gap_size = self._measure_signal_gap(reward_signal)

        likely_type = self._predict_inquiry_type(reward_signal)
        ask_effectiveness = self.inquiry_effectiveness[likely_type]["effectiveness"]

        training_value = self._estimate_training_value(experience, reward_signal)

        expected_value = (
            importance * 0.30
            + gap_size * 0.25
            + ask_effectiveness * 0.20
            + training_value * 0.25
        )

        if importance > 0.8 and gap_size > 0.6:
            priority = InquiryPriority.CRITICAL
        elif expected_value > 0.7:
            priority = InquiryPriority.HIGH
        elif expected_value > 0.4:
            priority = InquiryPriority.MODERATE
        else:
            priority = InquiryPriority.LOW

        return {
            "expected_value": expected_value,
            "priority": priority,
            "importance": importance,
            "gap_size": gap_size,
            "likely_type": likely_type,
            "target_dimensions": self._identify_unclear_dimensions(reward_signal),
        }

    def _assess_topic_importance(self, experience: Any) -> float:
        """Is this topic important enough to seek clarification on?"""
        score = 0.0
        domain_tags = getattr(experience, "domain_tags", [])
        input_context = getattr(experience, "input_context", "")

        # Frequency check
        for tag in domain_tags:
            freq = self._domain_counts.get(tag, 0) / max(self._total_experiences, 1)
            if freq > 0.1:
                score += 0.3
                break

        # High-stakes domains
        high_stakes = {"safety", "medical", "financial", "legal", "security", "infrastructure"}
        if any(tag in high_stakes for tag in domain_tags):
            score += 0.4

        # Novelty bonus
        if any(self._domain_counts.get(tag, 0) == 0 for tag in domain_tags):
            score += 0.2

        # User emphasis signals
        emphasis_markers = [
            "important", "critical", "need to get right",
            "please make sure", "this matters",
        ]
        lower_input = input_context.lower()
        if any(m in lower_input for m in emphasis_markers):
            score += 0.3

        return min(score, 1.0)

    def _measure_signal_gap(self, reward_signal: Any) -> float:
        """How incomplete or conflicted is the current reward signal?"""
        measured = reward_signal._get_measured_values()
        coverage_gap = 1.0 - (len(measured) / 5.0)

        conflict_gap = 0.5 if reward_signal._has_conflicts() else 0.0
        confidence_gap = 1.0 - reward_signal.confidence

        return coverage_gap * 0.3 + conflict_gap * 0.4 + confidence_gap * 0.3

    def _predict_inquiry_type(self, reward_signal: Any) -> InquiryType:
        """Based on which dimensions are unclear, determine question type."""
        unclear = self._identify_unclear_dimensions(reward_signal)

        if "groundedness" in unclear:
            return InquiryType.CORRECTNESS
        elif "task_success" in unclear:
            return InquiryType.USEFULNESS
        elif "user_satisfaction" in unclear:
            return InquiryType.PREFERENCE
        elif reward_signal._has_conflicts():
            return InquiryType.CONTRADICTION
        else:
            return InquiryType.IMPORTANCE

    def _identify_unclear_dimensions(self, reward_signal: Any) -> List[str]:
        """Which reward dimensions are missing or unmeasured?"""
        all_dims = [
            "task_success", "coherence", "user_satisfaction",
            "groundedness", "novelty_value",
        ]
        measured_names = set(reward_signal._get_measured_values().keys())
        return [d for d in all_dims if d not in measured_names]

    def _estimate_training_value(self, experience: Any, reward_signal: Any) -> float:
        """If resolved, how valuable would the training signal be?"""
        novelty = getattr(reward_signal, "novelty_value", None) or 0.5
        domain_tags = getattr(experience, "domain_tags", [])

        existing_coverage = 0.0
        for tag in domain_tags:
            count = self._domain_counts.get(tag, 0)
            existing_coverage = max(
                existing_coverage, count / max(self._total_experiences, 1)
            )

        return novelty * 0.5 + (1.0 - existing_coverage) * 0.5

    def _can_ask(self) -> bool:
        """Rate limiting and social awareness."""
        if self.session_asks_remaining <= 0:
            return False
        if self.daily_asks_remaining <= 0:
            return False
        if time.time() - self.last_ask_time < self.cooldown_seconds:
            return False
        return True

    def _formulate_inquiry(
        self,
        experience: Any,
        reward_signal: Any,
        gap_analysis: Dict[str, Any],
    ) -> ClarificationInquiry:
        """Generate the actual question."""
        inquiry_type = gap_analysis["likely_type"]
        target_dims = gap_analysis["target_dimensions"]

        topic = self._extract_topic(experience)
        templates = _QUESTION_TEMPLATES.get(inquiry_type, ["How did that land?"])
        question = random.choice(templates).format(topic=topic)

        return ClarificationInquiry(
            inquiry_id=f"inq_{int(time.time() * 1000)}",
            experience_id=getattr(experience, "experience_id", "unknown"),
            timestamp=time.time(),
            inquiry_type=inquiry_type,
            priority=gap_analysis["priority"],
            question_text=question,
            reasoning=(
                f"Gap size: {gap_analysis['gap_size']:.2f}, "
                f"Importance: {gap_analysis['importance']:.2f}, "
                f"Missing: {target_dims}"
            ),
            target_dimensions=target_dims,
            known_signals=dict(reward_signal._get_measured_values()),
            ambiguous_signals={d: "unmeasured" for d in target_dims},
        )

    def _extract_topic(self, experience: Any) -> str:
        """Pull the main topic from the experience."""
        tags = getattr(experience, "domain_tags", [])
        return tags[0] if tags else "that"

    def process_response(
        self, inquiry_id: str, response: str
    ) -> Optional[Dict[str, Optional[float]]]:
        """Handle the answer to a clarification question."""
        inquiry = self._find_inquiry(inquiry_id)
        if not inquiry:
            return None

        inquiry.asked = True
        inquiry.response = response

        resolved = self._parse_clarification(inquiry, response)
        inquiry.resolved_signal = resolved

        # Update effectiveness tracking
        useful = any(v is not None for v in resolved.values())
        eff = self.inquiry_effectiveness[inquiry.inquiry_type]
        eff["asked"] += 1
        if useful:
            eff["useful_response"] += 1
        eff["effectiveness"] = eff["useful_response"] / max(eff["asked"], 1)

        self.inquiry_history.append(inquiry)

        return resolved

    def _parse_clarification(
        self, inquiry: ClarificationInquiry, response: str
    ) -> Dict[str, Optional[float]]:
        """Convert natural language response into reward signal values."""
        resolved: Dict[str, Optional[float]] = {}
        lower = response.lower()

        positive = any(
            w in lower
            for w in [
                "yes", "correct", "helpful", "exactly", "perfect",
                "good", "right", "useful", "that works", "spot on",
            ]
        )
        negative = any(
            w in lower
            for w in [
                "wrong", "not really", "missed", "off", "incorrect",
                "didn't help", "not what I", "useless",
            ]
        )
        nuanced = any(
            w in lower
            for w in [
                "sort of", "partially", "mostly", "kind of",
                "close but", "almost", "not quite",
            ]
        )

        for dim in inquiry.target_dimensions:
            if positive:
                resolved[dim] = 0.8
            elif negative:
                resolved[dim] = -0.6
            elif nuanced:
                resolved[dim] = 0.3
            else:
                resolved[dim] = None

        return resolved

    def _find_inquiry(self, inquiry_id: str) -> Optional[ClarificationInquiry]:
        for inq in self.pending_inquiries:
            if inq.inquiry_id == inquiry_id:
                return inq
        return None

    def get_batched_checkin(self) -> Optional[Dict[str, Any]]:
        """Consolidated check-in for accumulated moderate/low priority inquiries."""
        pending_moderate = [
            inq
            for inq in self.pending_inquiries
            if inq.priority in (InquiryPriority.MODERATE, InquiryPriority.LOW)
            and not inq.asked
        ]

        if len(pending_moderate) < 3:
            return None

        return {
            "inquiries": pending_moderate[:5],
            "suggested_format": "consolidated_checkin",
            "intro": (
                "A few things I wanted to verify from "
                "recent conversations, no rush on any of these:"
            ),
        }

    def update_domain_stats(self, domain_tags: List[str]) -> None:
        """Track domain frequency for importance assessment."""
        self._total_experiences += 1
        for tag in domain_tags:
            self._domain_counts[tag] = self._domain_counts.get(tag, 0) + 1

    def reset_session(self, budget: int = 3) -> None:
        """Reset session-level ask budget."""
        self.session_asks_remaining = budget
