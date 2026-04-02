"""Curiosity engine -- autonomous exploration and cross-domain discovery.

The CuriosityEngine uses the InterestModel to identify what the system
is most curious about, then generates exploration queries, investigates
them via the backend, and feeds the results back into the experience buffer.

Three exploration modes:
1. **Deep dive** -- explore a high-curiosity domain in more depth
2. **Gap filling** -- investigate shelved/ambiguous experiences
3. **Cross-pollination** -- find connections between domains

The engine has an exploration budget to prevent runaway API costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openmind.curiosity.interest import DomainInterest, InterestModel

logger = logging.getLogger(__name__)


@dataclass
class Exploration:
    """Record of a single autonomous exploration."""

    exploration_id: str = ""
    timestamp: float = 0.0
    mode: str = ""              # "deep_dive", "gap_fill", "cross_pollinate"
    domain: str = ""
    query: str = ""
    response: str = ""
    insight: Optional[str] = None
    reward: float = 0.0
    connected_domain: Optional[str] = None  # for cross-pollination

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploration_id": self.exploration_id,
            "timestamp": self.timestamp,
            "mode": self.mode,
            "domain": self.domain,
            "query": self.query,
            "response": self.response,
            "insight": self.insight,
            "reward": self.reward,
            "connected_domain": self.connected_domain,
        }


class CuriosityEngine:
    """Drives autonomous exploration based on the system's curiosity.

    The engine decides WHAT to explore (based on InterestModel scores),
    HOW to explore it (deep dive, gap fill, or cross-pollinate), and
    evaluates the results to update the interest model.

    Parameters:
        interest_model: The InterestModel tracking per-domain curiosity.
        backend: The LLM backend for generating exploration queries/responses.
        experience_buffer: Where to store exploration results.
        knowledge_registry: For checking existing knowledge and gaps.
        budget_per_cycle: Max explorations per cycle (controls API costs).
        min_curiosity_threshold: Minimum curiosity score to trigger exploration.
    """

    # Templates for generating exploration queries
    DEEP_DIVE_TEMPLATES = [
        "I've been working with {domain} topics and I'm curious: what are the most common misconceptions or subtle pitfalls that people encounter?",
        "In the area of {domain}, what are the underlying principles that connect the specific things I've been learning? What's the deeper pattern?",
        "What advanced concepts in {domain} would naturally follow from the basics I already know? What should I learn next?",
        "What are the most interesting recent developments or shifts in thinking within {domain}?",
    ]

    GAP_FILL_TEMPLATES = [
        "I encountered something ambiguous about {domain}: {context}. Can you help me understand this better?",
        "I'm uncertain about an aspect of {domain} related to: {context}. What's the clearest way to think about this?",
        "I got conflicting signals about {domain} regarding: {context}. What's the most accurate understanding?",
    ]

    CROSS_POLLINATE_TEMPLATES = [
        "I've been learning about both {domain_a} and {domain_b}. Are there interesting connections, shared principles, or analogies between them?",
        "How might insights from {domain_a} apply to problems in {domain_b}? Are there transferable patterns?",
        "What would someone who is expert in both {domain_a} and {domain_b} notice that a specialist in only one might miss?",
    ]

    def __init__(
        self,
        interest_model: InterestModel,
        backend: Any = None,
        experience_buffer: Any = None,
        knowledge_registry: Any = None,
        budget_per_cycle: int = 3,
        min_curiosity_threshold: float = 0.3,
    ) -> None:
        self.interest_model = interest_model
        self.backend = backend
        self.experience_buffer = experience_buffer
        self.knowledge_registry = knowledge_registry
        self.budget_per_cycle = budget_per_cycle
        self.min_curiosity_threshold = min_curiosity_threshold

        self._exploration_history: List[Exploration] = []
        self._cycle_count = 0

    def explore(self) -> List[Exploration]:
        """Run an exploration cycle.

        Selects the most promising exploration targets, generates queries,
        investigates them, and records the results.

        Returns:
            List of Exploration records from this cycle.
        """
        if self.backend is None or not self.backend.is_ready:
            logger.info("Curiosity: no backend available, skipping exploration")
            return []

        self._cycle_count += 1
        explorations: List[Exploration] = []
        budget_remaining = self.budget_per_cycle

        # Plan explorations based on current curiosity state
        plan = self._plan_explorations(budget_remaining)

        for mode, target_info in plan:
            if budget_remaining <= 0:
                break

            try:
                exploration = self._execute_exploration(mode, target_info)
                if exploration:
                    explorations.append(exploration)
                    self._exploration_history.append(exploration)
                    budget_remaining -= 1

                    # Feed result back into the system
                    self._integrate_exploration(exploration)

            except Exception as e:
                logger.error("Exploration failed (%s): %s", mode, e)

        logger.info(
            "Curiosity cycle %d: %d explorations completed",
            self._cycle_count,
            len(explorations),
        )
        return explorations

    def get_curious_about(self) -> Dict[str, Any]:
        """Return a summary of what the system is currently curious about.

        Useful for showing the user what the system wants to learn.
        """
        most_curious = self.interest_model.get_most_curious(5)
        gaps = self.interest_model.get_knowledge_gaps()
        improving = self.interest_model.get_improving_domains()

        result: Dict[str, Any] = {
            "top_interests": [],
            "knowledge_gaps": [],
            "making_progress_in": [],
            "would_like_to_explore": [],
        }

        for name, di in most_curious:
            result["top_interests"].append({
                "domain": name,
                "curiosity": f"{di.curiosity_score:.2f}",
                "level": di.interest_level,
                "reason": self._explain_curiosity(di),
            })

        for name, di in gaps[:3]:
            result["knowledge_gaps"].append({
                "domain": name,
                "gap_score": f"{di.gap_score:.2f}",
                "shelved_experiences": di.shelved_count,
            })

        for name, di in improving[:3]:
            result["making_progress_in"].append({
                "domain": name,
                "velocity": f"{di.learning_velocity:.4f}",
                "mastery": f"{di.mastery_score:.2f}",
            })

        # Generate specific exploration ideas
        plan = self._plan_explorations(3)
        for mode, target_info in plan:
            result["would_like_to_explore"].append({
                "mode": mode,
                "target": target_info.get("domain", "unknown"),
                "reason": target_info.get("reason", ""),
            })

        return result

    # === Planning ===

    def _plan_explorations(
        self, budget: int
    ) -> List[tuple]:
        """Decide what to explore this cycle.

        Allocation strategy:
        - 1 slot for cross-pollination (if connected domains exist)
        - 1 slot for gap filling (if gaps exist)
        - Remaining slots for deep dives in high-curiosity domains
        """
        plan: List[tuple] = []

        # Cross-pollination: find domain pairs with connections
        cross_target = self._find_cross_pollination_target()
        if cross_target and budget > 0:
            plan.append(("cross_pollinate", cross_target))

        # Gap filling: address ambiguity
        gap_target = self._find_gap_fill_target()
        if gap_target and len(plan) < budget:
            plan.append(("gap_fill", gap_target))

        # Deep dives: explore high-curiosity domains
        most_curious = self.interest_model.get_most_curious(budget)
        for name, di in most_curious:
            if len(plan) >= budget:
                break
            if di.curiosity_score < self.min_curiosity_threshold:
                break
            # Don't duplicate if already planned
            if any(t.get("domain") == name for _, t in plan):
                continue
            plan.append((
                "deep_dive",
                {
                    "domain": name,
                    "interest": di,
                    "reason": self._explain_curiosity(di),
                },
            ))

        return plan

    def _find_cross_pollination_target(self) -> Optional[Dict[str, Any]]:
        """Find the best pair of domains to cross-pollinate."""
        all_domains = self.interest_model.get_all()

        best_pair = None
        best_score = 0.0

        domains_list = list(all_domains.items())
        for i, (name_a, di_a) in enumerate(domains_list):
            for name_b, di_b in domains_list[i + 1 :]:
                # Both domains should be somewhat known
                if di_a.total_experiences < 3 or di_b.total_experiences < 3:
                    continue
                # Skip if already connected
                if name_b in di_a.connected_domains:
                    continue

                score = (
                    di_a.curiosity_score + di_b.curiosity_score
                ) / 2.0

                if score > best_score:
                    best_score = score
                    best_pair = {
                        "domain": name_a,
                        "domain_a": name_a,
                        "domain_b": name_b,
                        "score": score,
                        "reason": f"Both {name_a} and {name_b} are active -- looking for connections",
                    }

        if best_pair and best_score > self.min_curiosity_threshold:
            return best_pair
        return None

    def _find_gap_fill_target(self) -> Optional[Dict[str, Any]]:
        """Find the best knowledge gap to investigate."""
        gaps = self.interest_model.get_knowledge_gaps()
        if not gaps:
            return None

        name, di = gaps[0]

        # Try to get context from shelved experiences
        context = self._get_gap_context(name)

        return {
            "domain": name,
            "interest": di,
            "context": context,
            "reason": f"{di.shelved_count} ambiguous experiences in {name}",
        }

    # === Execution ===

    def _execute_exploration(
        self, mode: str, target_info: Dict[str, Any]
    ) -> Optional[Exploration]:
        """Execute a single exploration."""
        import uuid

        query = self._generate_query(mode, target_info)
        if not query:
            return None

        domain = target_info.get("domain", "general")

        logger.info("Curiosity exploring [%s] %s: %s", mode, domain, query[:80])

        # Ask the backend
        response = self.backend.generate(query)
        if not response or response.startswith("[openMind:"):
            return None

        # Evaluate the exploration
        reward = self._evaluate_exploration(query, response, mode)

        # Extract insight if the response was valuable
        insight = None
        if reward > 0.3:
            insight = self._extract_insight(domain, query, response)

        exploration = Exploration(
            exploration_id=uuid.uuid4().hex[:12],
            timestamp=time.time(),
            mode=mode,
            domain=domain,
            query=query,
            response=response,
            insight=insight,
            reward=reward,
            connected_domain=target_info.get("domain_b"),
        )

        return exploration

    def _generate_query(self, mode: str, target_info: Dict) -> Optional[str]:
        """Generate an exploration query from templates."""
        import random

        domain = target_info.get("domain", "general")

        if mode == "deep_dive":
            template = random.choice(self.DEEP_DIVE_TEMPLATES)
            return template.format(domain=domain)

        elif mode == "gap_fill":
            context = target_info.get("context", f"general concepts in {domain}")
            template = random.choice(self.GAP_FILL_TEMPLATES)
            return template.format(domain=domain, context=context)

        elif mode == "cross_pollinate":
            domain_a = target_info.get("domain_a", domain)
            domain_b = target_info.get("domain_b", "general")
            template = random.choice(self.CROSS_POLLINATE_TEMPLATES)
            return template.format(domain_a=domain_a, domain_b=domain_b)

        return None

    def _evaluate_exploration(
        self, query: str, response: str, mode: str
    ) -> float:
        """Score how valuable an exploration was."""
        # Use backend self-eval if available
        if hasattr(self.backend, "evaluate_output"):
            scores = self.backend.evaluate_output(query, response)
            if scores:
                # Weight coherence and task_success for explorations
                task = scores.get("task_success", 0.0)
                coh = scores.get("coherence", 0.0)
                return (task + coh) / 2.0

        # Heuristic fallback: longer, substantive responses are better
        words = response.split()
        if len(words) < 20:
            return 0.1
        elif len(words) > 200:
            return 0.6
        else:
            return 0.4

    def _extract_insight(
        self, domain: str, query: str, response: str
    ) -> Optional[str]:
        """Ask the backend to distill the exploration into a concise insight."""
        if not hasattr(self.backend, "generate"):
            return None

        prompt = (
            f"Distill this into ONE concise sentence -- the key insight or "
            f"takeaway:\n\n"
            f"Question: {query}\n"
            f"Answer: {response[:500]}\n\n"
            f"Key insight:"
        )

        try:
            insight = self.backend.generate(prompt)
            if insight and not insight.startswith("[openMind:"):
                return insight.strip()
        except Exception:
            pass
        return None

    # === Integration ===

    def _integrate_exploration(self, exploration: Exploration) -> None:
        """Feed exploration results back into the system."""
        # Update interest model
        self.interest_model.record_exploration(exploration.domain)
        self.interest_model.update_from_experience(
            domain_tags=[exploration.domain],
            reward=exploration.reward,
        )

        # Record cross-domain connection
        if exploration.mode == "cross_pollinate" and exploration.connected_domain:
            if exploration.reward > 0.3:
                self.interest_model.discover_connection(
                    exploration.domain, exploration.connected_domain
                )

        # Store valuable explorations in experience buffer
        if (
            exploration.reward > 0.3
            and self.experience_buffer is not None
            and hasattr(self.experience_buffer, "record")
        ):
            from openmind.core.experience import Experience

            exp = Experience(
                timestamp=exploration.timestamp,
                input_context=f"[curiosity:{exploration.mode}] {exploration.query}",
                output=exploration.response,
                reward_signal=exploration.reward,
                domain_tags=[exploration.domain],
                confidence=0.6,  # slightly lower confidence for self-generated
            )
            self.experience_buffer.record(exp)

        # Register insight as knowledge if high quality
        if (
            exploration.insight
            and exploration.reward > 0.5
            and self.knowledge_registry is not None
        ):
            try:
                self.knowledge_registry.register_promotion(
                    promotion_result={
                        "training_cycle": 0,
                        "promoted_layers": [],
                        "promotion_scores": {"curiosity_reward": exploration.reward},
                        "confidence": min(exploration.reward, 0.8),
                    },
                    articulate_fn=lambda se, ps: exploration.insight,
                    source_experiences=[],
                )
                self.interest_model.update_from_knowledge([exploration.domain])
            except Exception as e:
                logger.warning("Failed to register curiosity insight: %s", e)

    # === Helpers ===

    def _get_gap_context(self, domain: str) -> str:
        """Get context about knowledge gaps from shelved experiences."""
        if self.experience_buffer is None:
            return f"general concepts in {domain}"

        if not hasattr(self.experience_buffer, "get_recent"):
            return f"general concepts in {domain}"

        recent = self.experience_buffer.get_recent(50)
        domain_exps = [
            e
            for e in recent
            if domain
            in (
                e.get("domain_tags", [])
                if isinstance(e, dict)
                else getattr(e, "domain_tags", [])
            )
        ]

        if domain_exps:
            # Use the most recent interaction as context
            latest = domain_exps[0]
            input_ctx = (
                latest.get("input_context", "")
                if isinstance(latest, dict)
                else getattr(latest, "input_context", "")
            )
            return input_ctx[:200] if input_ctx else f"general concepts in {domain}"

        return f"general concepts in {domain}"

    def _explain_curiosity(self, di: DomainInterest) -> str:
        """Generate a human-readable explanation of why we're curious."""
        reasons = []

        if di.learning_velocity > 0.005:
            reasons.append("actively improving")
        if di.gap_score > 0.3:
            reasons.append(f"{di.shelved_count} unanswered questions")
        if 0.3 < di.novelty_score < 0.8:
            reasons.append("moderately novel territory")
        if di.recent_reward_trend > 0.1:
            reasons.append("rewards are improving")
        if di.mastery_score < 0.3 and di.learning_velocity > 0:
            reasons.append("early stage with momentum")

        if not reasons:
            if di.curiosity_score > 0.3:
                reasons.append("general interest")
            else:
                reasons.append("low priority")

        return "; ".join(reasons)

    @property
    def exploration_history(self) -> List[Exploration]:
        """All explorations performed so far."""
        return list(self._exploration_history)

    @property
    def total_explorations(self) -> int:
        return len(self._exploration_history)
