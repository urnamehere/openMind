"""Claude API backend for openMind continuous learning.

Instead of training weights, this backend builds dynamic system prompts
from a knowledge registry and uses RAG over past experiences for few-shot
example selection. "Consolidation" distills high-value experiences into
structured knowledge entries. "Reflection" uses Claude to review beliefs.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from openmind.backends.base import Backend, GenerationContext

logger = logging.getLogger(__name__)


# ======================================================================
# System Prompt Builder
# ======================================================================


class SystemPromptBuilder:
    """Builds dynamic system prompts from the knowledge registry.

    Assembles a system prompt by combining a base identity prompt with
    learned knowledge entries, sorted by confidence and relevance.
    Respects a token budget to stay within context limits.
    """

    DEFAULT_BASE_PROMPT = (
        "You are an AI assistant with a continuous learning memory system. "
        "You have accumulated knowledge from past interactions that helps "
        "you provide better, more personalized responses. Below is what "
        "you have learned so far.\n"
    )

    def __init__(
        self,
        knowledge_registry: Any,
        base_prompt: Optional[str] = None,
    ) -> None:
        self._registry = knowledge_registry
        self._base_prompt = base_prompt or self.DEFAULT_BASE_PROMPT

    def build(
        self,
        domain_tags: Optional[List[str]] = None,
        token_budget: int = 4000,
    ) -> str:
        """Assemble a system prompt from active knowledge entries.

        Parameters:
            domain_tags: If provided, prioritize knowledge from these domains.
            token_budget: Approximate max tokens for the system prompt.

        Returns:
            The assembled system prompt string.
        """
        parts = [self._base_prompt]
        budget_used = self._estimate_tokens(self._base_prompt)

        # Get all active knowledge
        active = self._registry.get_active()
        if not active:
            return self._base_prompt

        # Score and sort: domain relevance first, then confidence
        scored = self._score_entries(active, domain_tags)

        # Add knowledge entries until budget is exhausted
        knowledge_parts = []
        for entry, _score in scored:
            belief = (
                entry.belief_summary
                if hasattr(entry, "belief_summary")
                else entry.get("belief_summary", "")
                if isinstance(entry, dict)
                else ""
            )
            if not belief or belief.startswith("[Placeholder]"):
                continue

            domains = (
                entry.source_domain
                if hasattr(entry, "source_domain")
                else entry.get("source_domain", [])
                if isinstance(entry, dict)
                else []
            )
            confidence = (
                entry.current_confidence
                if hasattr(entry, "current_confidence")
                else entry.get("current_confidence", 0.5)
                if isinstance(entry, dict)
                else 0.5
            )

            entry_text = (
                f"- [{', '.join(domains)}] (confidence: {confidence:.0%}): "
                f"{belief}"
            )
            entry_tokens = self._estimate_tokens(entry_text)

            if budget_used + entry_tokens > token_budget:
                break

            knowledge_parts.append(entry_text)
            budget_used += entry_tokens

        if knowledge_parts:
            parts.append("\n## What you have learned:\n")
            parts.extend(knowledge_parts)
            parts.append(
                "\n\nUse this knowledge naturally in your responses. "
                "Don't reference it explicitly unless asked."
            )

        return "\n".join(parts)

    def _score_entries(
        self, entries: list, domain_tags: Optional[List[str]]
    ) -> List[Tuple[Any, float]]:
        """Score entries by relevance and confidence."""
        domain_set = set(domain_tags) if domain_tags else set()
        scored = []

        for entry in entries:
            domains = (
                entry.source_domain
                if hasattr(entry, "source_domain")
                else entry.get("source_domain", [])
                if isinstance(entry, dict)
                else []
            )
            confidence = (
                entry.current_confidence
                if hasattr(entry, "current_confidence")
                else entry.get("current_confidence", 0.5)
                if isinstance(entry, dict)
                else 0.5
            )

            # Domain relevance bonus
            domain_overlap = (
                len(set(domains) & domain_set) / max(len(domain_set), 1)
                if domain_set
                else 0.5
            )

            score = confidence * 0.6 + domain_overlap * 0.4
            scored.append((entry, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate: ~4 chars per token."""
        return max(1, len(text) // 4)


# ======================================================================
# Experience Retriever (RAG)
# ======================================================================


class ExperienceRetriever:
    """RAG over past experiences for few-shot example selection.

    Retrieves high-quality past interactions that are semantically
    similar to the current input, for use as few-shot examples.
    """

    def __init__(
        self,
        experience_buffer: Any,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self._buffer = experience_buffer
        self._embedder = embedding_provider
        self._embedding_cache: Dict[str, np.ndarray] = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_reward: float = 0.3,
    ) -> List[Any]:
        """Retrieve the most relevant high-quality past experiences.

        Parameters:
            query: The current user input to match against.
            top_k: Maximum number of experiences to return.
            min_reward: Minimum reward threshold for candidate experiences.

        Returns:
            List of Experience objects, most relevant first.
        """
        if self._buffer is None:
            return []

        # Get recent experiences above reward threshold
        candidates = []
        recent = (
            self._buffer.get_recent(200)
            if hasattr(self._buffer, "get_recent")
            else []
        )
        for exp in recent:
            reward = (
                exp.get("reward_signal", 0)
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", 0)
            )
            if reward >= min_reward:
                candidates.append(exp)

        if not candidates:
            return []

        # If no embedding provider, fall back to keyword matching
        if self._embedder is None:
            return self._keyword_retrieve(query, candidates, top_k)

        # Embed query and candidates, rank by similarity
        try:
            query_vec = self._embedder.embed(query)
            scored = []
            for exp in candidates:
                input_ctx = (
                    exp.get("input_context", "")
                    if isinstance(exp, dict)
                    else getattr(exp, "input_context", "")
                )
                exp_id = (
                    exp.get("experience_id", id(exp))
                    if isinstance(exp, dict)
                    else getattr(exp, "experience_id", id(exp))
                )

                cache_key = str(exp_id)
                if cache_key not in self._embedding_cache:
                    self._embedding_cache[cache_key] = self._embedder.embed(
                        input_ctx
                    )

                sim = self._cosine_similarity(
                    query_vec, self._embedding_cache[cache_key]
                )
                scored.append((exp, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [exp for exp, _ in scored[:top_k]]

        except Exception:
            logger.warning("Embedding retrieval failed, falling back to keywords")
            return self._keyword_retrieve(query, candidates, top_k)

    def build_few_shot_messages(
        self, experiences: List[Any]
    ) -> List[Dict[str, str]]:
        """Convert experiences to Claude message format for few-shot examples."""
        messages = []
        for exp in experiences:
            input_ctx = (
                exp.get("input_context", "")
                if isinstance(exp, dict)
                else getattr(exp, "input_context", "")
            )
            output = (
                exp.get("output", "")
                if isinstance(exp, dict)
                else getattr(exp, "output", "")
            )
            if input_ctx and output:
                messages.append({"role": "user", "content": input_ctx})
                messages.append({"role": "assistant", "content": output})
        return messages

    def _keyword_retrieve(
        self, query: str, candidates: list, top_k: int
    ) -> list:
        """Simple keyword overlap retrieval as fallback."""
        query_words = set(query.lower().split())
        scored = []
        for exp in candidates:
            input_ctx = (
                exp.get("input_context", "")
                if isinstance(exp, dict)
                else getattr(exp, "input_context", "")
            )
            exp_words = set(input_ctx.lower().split())
            if not query_words or not exp_words:
                continue
            overlap = len(query_words & exp_words) / len(query_words | exp_words)
            scored.append((exp, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [exp for exp, _ in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ======================================================================
# Claude Backend
# ======================================================================


class ClaudeBackend(Backend):
    """Backend using the Anthropic Claude API.

    Learning happens through knowledge distillation rather than weight
    training. The system builds dynamic system prompts from accumulated
    knowledge and uses RAG over past experiences for few-shot examples.

    Parameters:
        config: A ClaudeConfig instance.
        knowledge_registry: The KnowledgeRegistry for accumulated knowledge.
        experience_buffer: The ExperienceBuffer for past interactions.
        embedding_provider: Optional EmbeddingProvider for semantic search.
    """

    def __init__(
        self,
        config: Any,
        knowledge_registry: Any,
        experience_buffer: Any,
        embedding_provider: Optional[Any] = None,
    ) -> None:
        self._config = config
        self._knowledge_registry = knowledge_registry
        self._experience_buffer = experience_buffer

        self._client: Any = None
        self._prompt_builder = SystemPromptBuilder(
            knowledge_registry=knowledge_registry,
            base_prompt=getattr(config, "base_system_prompt", None),
        )
        self._retriever = ExperienceRetriever(
            experience_buffer=experience_buffer,
            embedding_provider=embedding_provider,
        )

        self._try_init_client()

    def _try_init_client(self) -> None:
        """Initialize the Anthropic client."""
        env_var = getattr(self._config, "api_key_env_var", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(env_var)

        if not api_key:
            logger.warning(
                "No API key found in %s. Set this environment variable "
                "to enable Claude API calls.",
                env_var,
            )
            return

        try:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
            logger.info("Claude API client initialized (model: %s)", self._config.model)
        except ImportError:
            logger.warning(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        except Exception as e:
            logger.warning("Failed to initialize Claude client: %s", e)

    def generate(
        self,
        user_input: str,
        context: Optional[GenerationContext] = None,
    ) -> str:
        if self._client is None:
            return "[openMind: Claude API not configured - recording interaction]"

        ctx = context or GenerationContext()

        # Build system prompt from knowledge
        domain_tags = ctx.domain_tags
        system_prompt = ctx.system_prompt or self._prompt_builder.build(
            domain_tags=domain_tags,
            token_budget=self._config.system_prompt_token_budget,
        )

        # Retrieve relevant past experiences for few-shot
        few_shot_messages = []
        if ctx.few_shot_examples is not None:
            few_shot_messages = ctx.few_shot_examples
        else:
            relevant = self._retriever.retrieve(
                query=user_input,
                top_k=self._config.few_shot_count,
                min_reward=self._config.few_shot_min_reward,
            )
            if relevant:
                few_shot_messages = self._retriever.build_few_shot_messages(
                    relevant
                )

        # Assemble messages
        messages = []
        messages.extend(few_shot_messages)

        # Add conversation history if provided
        if ctx.conversation_history:
            messages.extend(ctx.conversation_history)

        messages.append({"role": "user", "content": user_input})

        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=ctx.max_tokens or self._config.max_tokens,
                temperature=ctx.temperature or self._config.temperature,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text

        except Exception as e:
            logger.error("Claude API call failed: %s", e)
            return f"[openMind: API error - {e}]"

    def evaluate_output(
        self, input_context: str, output: str
    ) -> Optional[Dict[str, float]]:
        if self._client is None:
            return None

        eval_model = getattr(self._config, "eval_model", self._config.model)

        prompt = (
            "Rate the following AI output on these dimensions. "
            "Each score should be a float from -1.0 (terrible) to 1.0 (excellent).\n\n"
            f"USER REQUEST:\n{input_context}\n\n"
            f"AI OUTPUT:\n{output}\n\n"
            "Respond in exactly this JSON format:\n"
            '{"task_success": <score>, "coherence": <score>, '
            '"groundedness": <score>}\n'
        )

        try:
            response = self._client.messages.create(
                model=eval_model,
                max_tokens=100,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return self._parse_eval_json(raw)
        except Exception:
            logger.warning("Claude self-evaluation failed", exc_info=True)
            return None

    def articulate_learning(
        self,
        source_experiences: Sequence[Any],
        promotion_scores: Dict[str, float],
    ) -> str:
        if self._client is None:
            return self._placeholder_articulation(source_experiences)

        examples = []
        for exp in source_experiences[:5]:
            input_ctx = (
                exp.get("input_context", "?")
                if isinstance(exp, dict)
                else getattr(exp, "input_context", "?")
            )
            output = (
                exp.get("output", "?")
                if isinstance(exp, dict)
                else getattr(exp, "output", "?")
            )
            reward = (
                exp.get("reward_signal", "?")
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", "?")
            )
            examples.append(
                f"  Input: {input_ctx}\n"
                f"  Output: {output}\n"
                f"  Reward: {reward}"
            )
        examples_text = "\n---\n".join(examples)

        prompt = (
            "You are a reflective learning system. Based on the following "
            "interactions and their reward signals, articulate in one to three "
            "sentences what you have learned. Focus on the rule, pattern, or "
            "factual knowledge — not the interactions themselves.\n\n"
            f"Interactions:\n{examples_text}\n\n"
            f"Quality scores: {json.dumps(promotion_scores)}\n\n"
            "Your articulation:"
        )

        try:
            response = self._client.messages.create(
                model=getattr(self._config, "eval_model", self._config.model),
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception:
            return self._placeholder_articulation(source_experiences)

    def consolidate(
        self,
        experience_buffer: Any,
        knowledge_registry: Any,
        cycle: int,
    ) -> Dict[str, Any]:
        """Claude-specific consolidation: knowledge distillation.

        1. Gather high-reward recent experiences
        2. Cluster by domain
        3. For each cluster, ask Claude to extract generalizable knowledge
        4. Register extracted knowledge in the KnowledgeRegistry
        """
        stats: Dict[str, Any] = {"backend": "claude", "distilled": 0, "errors": 0}

        if self._client is None:
            stats["status"] = "no_client"
            return stats

        # Get high-reward experiences not yet distilled
        recent = (
            experience_buffer.get_recent(200)
            if hasattr(experience_buffer, "get_recent")
            else []
        )
        high_quality = []
        for exp in recent:
            reward = (
                exp.get("reward_signal", 0)
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", 0)
            )
            if reward >= self._config.few_shot_min_reward:
                high_quality.append(exp)

        if not high_quality:
            stats["status"] = "no_high_quality_experiences"
            return stats

        # Cluster by domain
        domain_clusters: Dict[str, List] = defaultdict(list)
        for exp in high_quality:
            tags = (
                exp.get("domain_tags", ["general"])
                if isinstance(exp, dict)
                else getattr(exp, "domain_tags", ["general"])
            )
            for tag in tags:
                domain_clusters[tag].append(exp)

        # Cap total API calls per consolidation cycle
        max_clusters = getattr(self._config, "consolidation_max_clusters", 5)
        cluster_size = getattr(self._config, "consolidation_cluster_size", 10)

        for domain, exps in list(domain_clusters.items())[:max_clusters]:
            batch = exps[:cluster_size]
            try:
                belief = self._distill_knowledge(batch, domain)
                if belief:
                    # Compute average reward as confidence proxy
                    rewards = [
                        (
                            e.get("reward_signal", 0.5)
                            if isinstance(e, dict)
                            else getattr(e, "reward_signal", 0.5)
                        )
                        for e in batch
                    ]
                    avg_reward = sum(rewards) / len(rewards)

                    knowledge_registry.register_promotion(
                        promotion_result={
                            "training_cycle": cycle,
                            "promoted_layers": [],
                            "promotion_scores": {"avg_reward": avg_reward},
                            "confidence": min(avg_reward, 0.95),
                            "layer_names": [],
                        },
                        articulate_fn=lambda se, ps: belief,
                        source_experiences=batch,
                    )
                    stats["distilled"] += 1

            except Exception as e:
                logger.error("Knowledge distillation failed for %s: %s", domain, e)
                stats["errors"] += 1

        stats["status"] = "completed"
        stats["domains_processed"] = min(len(domain_clusters), max_clusters)
        return stats

    def reflect(
        self,
        knowledge_registry: Any,
        experience_buffer: Any,
        belief_reviewer: Any,
    ) -> Dict[str, Any]:
        """Claude-specific reflection: review beliefs using Claude.

        1. Get review candidates from knowledge_registry
        2. For each candidate, use Claude to evaluate against recent evidence
        3. Update knowledge entry status
        """
        stats: Dict[str, Any] = {"backend": "claude", "reviews": []}

        candidates = knowledge_registry.get_review_candidates()
        max_reviews = getattr(self._config, "max_reviews_per_cycle", 10)

        for pk in candidates[:max_reviews]:
            kid = pk.knowledge_id if hasattr(pk, "knowledge_id") else pk
            try:
                if self._client is not None:
                    # Enhanced review: use Claude to assess the belief
                    enhanced = self._claude_enhanced_review(pk, experience_buffer)
                    if enhanced:
                        stats["reviews"].append(enhanced)
                        continue

                # Fall back to standard belief reviewer
                result = belief_reviewer.review(kid)
                stats["reviews"].append(result)
            except Exception as e:
                logger.error("Review failed for %s: %s", kid, e)

        stats["reviewed_count"] = len(stats["reviews"])
        return stats

    @property
    def is_ready(self) -> bool:
        return self._client is not None

    # === Private helpers ===

    def _distill_knowledge(self, experiences: List, domain: str) -> Optional[str]:
        """Ask Claude to extract generalizable knowledge from experiences."""
        examples = []
        for exp in experiences[:8]:
            input_ctx = (
                exp.get("input_context", "")
                if isinstance(exp, dict)
                else getattr(exp, "input_context", "")
            )
            output = (
                exp.get("output", "")
                if isinstance(exp, dict)
                else getattr(exp, "output", "")
            )
            reward = (
                exp.get("reward_signal", 0)
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", 0)
            )
            examples.append(
                f"Input: {input_ctx[:200]}\n"
                f"Output: {output[:200]}\n"
                f"Reward: {reward:.2f}"
            )
        examples_text = "\n---\n".join(examples)

        prompt = (
            f"Domain: {domain}\n\n"
            "Below are interactions from this domain with their quality scores. "
            "Extract ONE concise, generalizable insight or rule that explains "
            "what makes responses in this domain successful. Focus on actionable "
            "patterns, not specific content.\n\n"
            f"Interactions:\n{examples_text}\n\n"
            "Your insight (1-3 sentences):"
        )

        response = self._client.messages.create(
            model=getattr(self._config, "eval_model", self._config.model),
            max_tokens=200,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        result = response.content[0].text.strip()
        return result if result else None

    def _claude_enhanced_review(
        self, knowledge_entry: Any, experience_buffer: Any
    ) -> Optional[Dict[str, Any]]:
        """Use Claude to review a knowledge entry against recent evidence."""
        belief = (
            knowledge_entry.belief_summary
            if hasattr(knowledge_entry, "belief_summary")
            else knowledge_entry.get("belief_summary", "")
            if isinstance(knowledge_entry, dict)
            else ""
        )
        if not belief or belief.startswith("[Placeholder]"):
            return None

        domains = (
            knowledge_entry.source_domain
            if hasattr(knowledge_entry, "source_domain")
            else knowledge_entry.get("source_domain", [])
            if isinstance(knowledge_entry, dict)
            else []
        )
        kid = (
            knowledge_entry.knowledge_id
            if hasattr(knowledge_entry, "knowledge_id")
            else knowledge_entry.get("knowledge_id", "unknown")
            if isinstance(knowledge_entry, dict)
            else "unknown"
        )

        # Get recent evidence from the same domain
        recent = []
        if hasattr(experience_buffer, "get_by_domain"):
            recent = experience_buffer.get_by_domain(domains, days_back=30)
        elif hasattr(experience_buffer, "get_recent"):
            recent = experience_buffer.get_recent(50)

        evidence_text = ""
        for exp in recent[:5]:
            input_ctx = (
                exp.get("input_context", "")[:150]
                if isinstance(exp, dict)
                else getattr(exp, "input_context", "")[:150]
            )
            reward = (
                exp.get("reward_signal", 0)
                if isinstance(exp, dict)
                else getattr(exp, "reward_signal", 0)
            )
            evidence_text += f"- Input: {input_ctx} (reward: {reward:.2f})\n"

        prompt = (
            f"Review this learned belief:\n\"{belief}\"\n\n"
            f"Domains: {', '.join(domains)}\n\n"
            f"Recent evidence from these domains:\n{evidence_text}\n\n"
            "Evaluate:\n"
            "1. Is this belief still supported by recent evidence? (0.0-1.0)\n"
            "2. Does it contradict any patterns in the evidence? (yes/no)\n"
            "3. Is this domain still active? (yes/no)\n"
            "4. Recommended action: reinforce / revise / deprecate / flag_for_inquiry\n\n"
            "Respond in JSON:\n"
            '{"support_score": <float>, "contradicted": <bool>, '
            '"still_active": <bool>, "action": "<string>", '
            '"reasoning": "<brief explanation>"}'
        )

        try:
            response = self._client.messages.create(
                model=getattr(self._config, "eval_model", self._config.model),
                max_tokens=200,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Parse JSON response
            # Handle potential markdown wrapping
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = json.loads(raw)

            return {
                "knowledge_id": kid,
                "review_timestamp": time.time(),
                "tests": {
                    "claude_review": {
                        "score": result.get("support_score", 0.5),
                        "contradicted": result.get("contradicted", False),
                        "still_active": result.get("still_active", True),
                    }
                },
                "verdict": {
                    "action": result.get("action", "reinforce"),
                    "reason": result.get("reasoning", "Claude review"),
                    "confidence": result.get("support_score", 0.5),
                },
            }
        except Exception:
            logger.warning("Claude enhanced review failed", exc_info=True)
            return None

    @staticmethod
    def _parse_eval_json(raw: str) -> Optional[Dict[str, float]]:
        """Parse JSON evaluation response."""
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            scores = {}
            for key in ("task_success", "coherence", "groundedness"):
                if key in data:
                    scores[key] = max(-1.0, min(1.0, float(data[key])))
            return scores if scores else None
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _placeholder_articulation(source_experiences: Sequence[Any]) -> str:
        return (
            f"[Placeholder] Knowledge distilled from {len(source_experiences)} "
            f"experience(s)."
        )
