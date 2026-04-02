"""ContinualWrapper - the main orchestrator that ties everything together."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openmind.backends.base import Backend
from openmind.core.experience import Experience, ExperienceBuffer
from openmind.detection.detectors import DetectionResult, SignalQualityDetector, SignalVerdict
from openmind.memory.ambiguity import AmbiguityBuffer, AmbiguityResolver
from openmind.memory.knowledge import KnowledgeRegistry
from openmind.memory.review import BeliefReviewer
from openmind.reward.collector import RewardCollector
from openmind.reward.inquiry import ActiveInquirySystem
from openmind.reward.meta import MetaRewardSystem
from openmind.reward.temporal import TemporalRewardTracker
from openmind.utils.config import OpenMindConfig, load_config
from openmind.utils.domain_tagger import DomainTagger

logger = logging.getLogger(__name__)


class ContinualWrapper:
    """
    A continuous learning wrapper for LLMs that enables experiential
    memory, knowledge accumulation, and belief revision.

    Supports two backends:
    - **Claude API**: Learning via knowledge distillation, dynamic system
      prompts, and RAG over past experiences.
    - **Local HuggingFace**: Learning via LoRA adapter training, EWC
      regularization, and selective weight promotion.

    Three lifecycle phases:
    - Awake: process interactions, collect rewards, detect signal quality
    - Sleep (consolidate): distill knowledge or train adapter weights
    - Reflect: review accumulated knowledge, run meta-reward evaluation

    Usage::

        # Claude API backend (default)
        from openmind import ContinualWrapper
        model = ContinualWrapper()
        response = model.chat("hello")

        # Local model backend
        model = ContinualWrapper(base_model="mistralai/Mistral-7B-v0.3")
        response = model.chat("hello")

        model.consolidate()  # trigger sleep cycle
        model.reflect()      # trigger belief review
    """

    def __init__(
        self,
        base_model: Optional[str] = None,
        config: Optional[OpenMindConfig] = None,
        config_path: Optional[str] = None,
        data_dir: str = "./openmind_data",
    ) -> None:
        # Load configuration
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            self.config = OpenMindConfig()

        # If a base_model path is given, switch to local backend
        if base_model is not None:
            self.config.model.base_model_path = base_model
            self.config.backend.backend_type = "local"

        # Override data directory
        self.config.storage.data_dir = data_dir
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        # Initialize storage paths
        exp_db = os.path.join(data_dir, "experiences.jsonl")
        amb_db = os.path.join(data_dir, "ambiguity_buffer.jsonl")
        know_db = os.path.join(data_dir, "knowledge_registry.jsonl")
        temp_db = os.path.join(data_dir, "temporal_rewards.jsonl")

        # === Core subsystems ===
        self.experience_buffer = ExperienceBuffer(db_path=exp_db)
        self.domain_tagger = DomainTagger()
        self.knowledge_registry = KnowledgeRegistry(storage_path=know_db)

        # === Backend ===
        self.backend: Backend = self._create_backend()

        # === Reward subsystems ===
        self.reward_collector = RewardCollector(
            model=self.backend if self.backend.is_ready else None,
        )
        self.temporal_tracker = TemporalRewardTracker(db_path=temp_db)
        self.inquiry_system = ActiveInquirySystem(
            ask_budget_per_session=self.config.inquiry.ask_budget_per_session,
            ask_budget_per_day=self.config.inquiry.ask_budget_per_day,
            cooldown_seconds=self.config.inquiry.cooldown_seconds,
        )
        self.meta_rewards = MetaRewardSystem()

        # === Detection ===
        self.signal_detector = SignalQualityDetector(
            experience_buffer=self.experience_buffer,
            knowledge_registry=self.knowledge_registry,
        )

        # === Memory subsystems ===
        self.ambiguity_buffer = AmbiguityBuffer(db_path=amb_db)
        self.ambiguity_resolver = AmbiguityResolver(
            ambiguity_buffer=self.ambiguity_buffer,
            experience_buffer=self.experience_buffer,
            stability_tracker=None,
        )

        # For local backend, wire up stability tracker
        if self.backend.supports_weight_training:
            self.ambiguity_resolver.stability_tracker = getattr(
                self.backend, "stability_tracker", None
            )

        self.belief_reviewer = BeliefReviewer(
            model=self.backend if self.backend.is_ready else None,
            experience_buffer=self.experience_buffer,
            reward_collector=self.reward_collector,
            knowledge_registry=self.knowledge_registry,
        )

        # Tracking
        self._training_cycle = 0
        self._interaction_count = 0
        self._discard_log: List[DetectionResult] = []

    def _create_backend(self) -> Backend:
        """Factory method to create the appropriate backend."""
        backend_type = self.config.backend.backend_type

        if backend_type == "claude":
            from openmind.backends.claude import ClaudeBackend

            embedding_provider = None
            try:
                from openmind.utils.embeddings import EmbeddingProvider

                embedding_provider = EmbeddingProvider()
            except Exception:
                pass

            return ClaudeBackend(
                config=self.config.backend.claude,
                knowledge_registry=self.knowledge_registry,
                experience_buffer=self.experience_buffer,
                embedding_provider=embedding_provider,
            )

        elif backend_type == "local":
            from openmind.backends.local import LocalHFBackend

            return LocalHFBackend(
                model_config=self.config.model,
                promotion_config=self.config.promotion,
            )

        else:
            raise ValueError(f"Unknown backend type: {backend_type!r}")

    def chat(
        self,
        user_input: str,
        user_response: Optional[str] = None,
        execution_result: Optional[Dict] = None,
    ) -> str:
        """
        Main interaction loop.

        Args:
            user_input: The user's message.
            user_response: Optional follow-up for reward inference.
            execution_result: Optional code execution result for task_success.

        Returns:
            The model's response (may include appended inquiry question).
        """
        self._interaction_count += 1

        # Generate response
        output = self._generate(user_input)

        # Tag domain
        domain_tags = self.domain_tagger.tag(user_input, output)
        self.inquiry_system.update_domain_stats(domain_tags)

        # Collect reward signal
        reward = self.reward_collector.collect(
            input_context=user_input,
            output=output,
            user_response=user_response,
            execution_result=execution_result,
        )

        # Create experience
        experience = Experience(
            timestamp=time.time(),
            input_context=user_input,
            output=output,
            reward_signal=reward.aggregate,
            domain_tags=domain_tags,
            confidence=reward.confidence,
        )

        # Run signal quality detection (the 4-way gatekeeper)
        detection = self.signal_detector.evaluate(experience, reward)

        if detection.verdict == SignalVerdict.TRAIN:
            experience.training_weight = detection.training_weight
            self.experience_buffer.record(experience)
            self.temporal_tracker.register(experience.experience_id, reward)

        elif detection.verdict == SignalVerdict.ASK:
            inquiry = detection.suggested_inquiry
            if inquiry:
                output += f"\n\n{inquiry['question']}"
                self.inquiry_system.session_asks_remaining -= 1
                self.inquiry_system.daily_asks_remaining -= 1
                self.inquiry_system.last_ask_time = time.time()

        elif detection.verdict == SignalVerdict.SHELVE:
            self.ambiguity_buffer.shelve(
                experience=experience,
                ambiguity_type=detection.shelve_reason or "low_confidence",
                ambiguity_score=1.0 - detection.confidence,
                partial_signals=reward._get_measured_values(),
            )

        elif detection.verdict == SignalVerdict.DISCARD:
            self._discard_log.append(detection)

        return output

    def consolidate(self) -> Dict[str, Any]:
        """
        The 'sleep' cycle. Delegates to the backend for learning, then
        resolves ambiguities.

        Returns:
            Dictionary of cycle statistics.
        """
        stats: Dict[str, Any] = {
            "cycle": self._training_cycle,
            "timestamp": time.time(),
            "consolidation": None,
            "ambiguity_resolution": None,
        }

        # Backend-specific consolidation
        try:
            backend_stats = self.backend.consolidate(
                experience_buffer=self.experience_buffer,
                knowledge_registry=self.knowledge_registry,
                cycle=self._training_cycle,
            )
            stats["consolidation"] = backend_stats
            self._training_cycle += 1
        except Exception as e:
            logger.error("Consolidation failed: %s", e)
            stats["consolidation"] = {"error": str(e)}

        # Resolve ambiguities (backend-agnostic)
        try:
            resolved, still_pending = self.ambiguity_resolver.review_cycle(
                model=self.backend if self.backend.is_ready else None,
                current_cycle=self._training_cycle,
            )

            for resolution in resolved:
                res_data = resolution.get("resolution", {})
                if res_data.get("resolved_reward") is not None:
                    amb = resolution["ambiguity"]
                    input_ctx = (
                        amb.get("input_context", "")
                        if isinstance(amb, dict)
                        else getattr(amb, "input_context", "")
                    )
                    output = (
                        amb.get("output", "")
                        if isinstance(amb, dict)
                        else getattr(amb, "output", "")
                    )
                    dtags = (
                        amb.get("domain_tags", [])
                        if isinstance(amb, dict)
                        else getattr(amb, "domain_tags", [])
                    )
                    self.experience_buffer.record(
                        Experience(
                            timestamp=time.time(),
                            input_context=input_ctx,
                            output=output,
                            reward_signal=res_data["resolved_reward"],
                            domain_tags=dtags,
                            confidence=res_data.get("new_confidence", 0.7),
                        )
                    )

            stats["ambiguity_resolution"] = {
                "resolved": len(resolved),
                "still_pending": len(still_pending),
            }
        except Exception as e:
            logger.error("Ambiguity resolution failed: %s", e)
            stats["ambiguity_resolution"] = {"error": str(e)}

        return stats

    def reflect(self) -> Dict[str, Any]:
        """
        The 'reflection' cycle. Delegates to the backend for belief review,
        then runs meta-reward evaluation.

        Returns:
            Dictionary of review statistics.
        """
        stats: Dict[str, Any] = {
            "timestamp": time.time(),
            "reflection": None,
            "meta_rewards": None,
        }

        # Backend-specific reflection
        try:
            reflection_stats = self.backend.reflect(
                knowledge_registry=self.knowledge_registry,
                experience_buffer=self.experience_buffer,
                belief_reviewer=self.belief_reviewer,
            )
            stats["reflection"] = reflection_stats
        except Exception as e:
            logger.error("Reflection failed: %s", e)
            stats["reflection"] = {"error": str(e)}

        # Meta-reward evaluation (backend-agnostic)
        try:
            metrics, recommendations = self.meta_rewards.evaluate_reward_quality(
                temporal_tracker=self.temporal_tracker,
            )
            stats["meta_rewards"] = {
                "metrics": metrics,
                "recommendations": recommendations,
            }
        except Exception as e:
            logger.error("Meta-reward evaluation failed: %s", e)
            stats["meta_rewards"] = {"error": str(e)}

        return stats

    def get_status(self) -> Dict[str, Any]:
        """Get current system state."""
        return {
            "backend": self.config.backend.backend_type,
            "backend_ready": self.backend.is_ready,
            "interaction_count": self._interaction_count,
            "training_cycle": self._training_cycle,
            "experience_buffer_size": len(self.experience_buffer.buffer),
            "ambiguity_buffer_size": len(self.ambiguity_buffer.active_ambiguities),
            "knowledge_count": len(self.knowledge_registry),
            "temporal_tracker_size": len(self.temporal_tracker.active_rewards),
            "pending_inquiries": len(self.inquiry_system.pending_inquiries),
            "discarded_count": len(self._discard_log),
            "domain_frequencies": self.domain_tagger.get_all_frequencies(),
        }

    def _generate(self, user_input: str) -> str:
        """Generate a response via the backend."""
        if not self.backend.is_ready:
            return "[openMind: backend not ready - recording interaction for future learning]"

        from openmind.backends.base import GenerationContext

        domain_tags = self.domain_tagger.tag(user_input, "")

        context = GenerationContext(
            domain_tags=domain_tags,
            max_tokens=self.config.backend.claude.max_tokens
            if self.config.backend.backend_type == "claude"
            else 1024,
            temperature=self.config.backend.claude.temperature
            if self.config.backend.backend_type == "claude"
            else 0.7,
        )

        return self.backend.generate(user_input, context=context)
