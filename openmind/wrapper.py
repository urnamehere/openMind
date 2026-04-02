"""ContinualWrapper - the main orchestrator that ties everything together."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from openmind.core.experience import Experience, ExperienceBuffer
from openmind.core.promotion import WeightPromoter, WeightStabilityTracker
from openmind.core.training import TrainingCycleManager
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
    memory, weight promotion, and belief revision.

    Three lifecycle phases:
    - Awake: process interactions, collect rewards, detect signal quality
    - Sleep (consolidate): train adapter, promote stable weights, resolve ambiguities
    - Reflect: review promoted knowledge, run meta-reward evaluation

    Usage::

        from openmind import ContinualWrapper

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

        if base_model is not None:
            self.config.model.base_model_path = base_model

        # Override data directory
        self.config.storage.data_dir = data_dir

        # Ensure data directory exists
        Path(data_dir).mkdir(parents=True, exist_ok=True)

        # Initialize storage paths
        exp_db = os.path.join(data_dir, "experiences.jsonl")
        amb_db = os.path.join(data_dir, "ambiguity_buffer.jsonl")
        know_db = os.path.join(data_dir, "knowledge_registry.jsonl")
        temp_db = os.path.join(data_dir, "temporal_rewards.jsonl")

        # === Core subsystems ===
        self.experience_buffer = ExperienceBuffer(db_path=exp_db)
        self.domain_tagger = DomainTagger()

        # === Reward subsystems ===
        self.reward_collector = RewardCollector()
        self.temporal_tracker = TemporalRewardTracker(db_path=temp_db)
        self.inquiry_system = ActiveInquirySystem(
            ask_budget_per_session=self.config.inquiry.ask_budget_per_session,
            ask_budget_per_day=self.config.inquiry.ask_budget_per_day,
            cooldown_seconds=self.config.inquiry.cooldown_seconds,
        )
        self.meta_rewards = MetaRewardSystem()

        # === Detection ===
        self.knowledge_registry = KnowledgeRegistry(db_path=know_db)
        self.signal_detector = SignalQualityDetector(
            experience_buffer=self.experience_buffer,
            knowledge_registry=self.knowledge_registry,
        )

        # === Memory subsystems ===
        self.ambiguity_buffer = AmbiguityBuffer(db_path=amb_db)
        self.ambiguity_resolver = AmbiguityResolver(
            ambiguity_buffer=self.ambiguity_buffer,
            experience_buffer=self.experience_buffer,
            stability_tracker=None,  # set after stability_tracker init
        )

        # === Training subsystems (require GPU) ===
        self._model = None
        self._tokenizer = None
        self._adapter_model = None
        self.stability_tracker = WeightStabilityTracker(
            history_window=self.config.promotion.min_cycles_before_promotion + 7
        )
        self.ambiguity_resolver.stability_tracker = self.stability_tracker

        self.weight_promoter: Optional[WeightPromoter] = None
        self.training_manager: Optional[TrainingCycleManager] = None
        self.belief_reviewer = BeliefReviewer(
            model=None,
            experience_buffer=self.experience_buffer,
            reward_collector=self.reward_collector,
            knowledge_registry=self.knowledge_registry,
        )

        # Tracking
        self._training_cycle = 0
        self._interaction_count = 0
        self._discard_log: List[DetectionResult] = []

        # Try to load model if specified
        if self.config.model.base_model_path:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the base model and set up LoRA adapter."""
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = self.config.model.base_model_path
            logger.info("Loading model: %s", model_path)

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )

            lora_config = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=self.config.model.target_modules,
                lora_dropout=self.config.model.lora_dropout,
            )
            self._adapter_model = get_peft_model(self._model, lora_config)

            self.weight_promoter = WeightPromoter(
                base_model=self._model,
                adapter_model=self._adapter_model,
                eval_fn=self._evaluate_model,
                rollback_threshold=self.config.promotion.rollback_threshold,
                merge_alpha=self.config.model.merge_alpha,
            )

            self.training_manager = TrainingCycleManager(
                base_model=self._model,
                adapter_model=self._adapter_model,
                experience_buffer=self.experience_buffer,
            )

            self.belief_reviewer.model = self._model
            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning(
                "GPU dependencies not available (torch, peft, transformers). "
                "Running in data-collection-only mode."
            )
        except Exception as e:
            logger.warning("Failed to load model: %s. Running in data-collection mode.", e)

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
        The 'sleep' cycle. Train adapter, check for promotions,
        resolve ambiguities.

        Returns:
            Dictionary of cycle statistics.
        """
        stats: Dict[str, Any] = {
            "cycle": self._training_cycle,
            "timestamp": time.time(),
            "training": None,
            "promotion": None,
            "ambiguity_resolution": None,
        }

        # 1. Training cycle
        if self.training_manager is not None:
            try:
                training_stats = self.training_manager.run_cycle()
                stats["training"] = training_stats
                self._training_cycle += 1

                # 2. Record adapter state for stability tracking
                if self._adapter_model is not None:
                    self.stability_tracker.record_adapter_state(
                        self._adapter_model.state_dict(),
                        self._training_cycle,
                    )

                # 3. Check for promotable weights
                scores = self.stability_tracker.compute_promotion_scores(
                    threshold=self.config.promotion.stability_threshold
                )
                promotable = {
                    k: v for k, v in scores.items()
                    if v.get("ready_for_promotion", False)
                }

                if promotable and self.weight_promoter is not None:
                    promoted, failed = self.weight_promoter.selective_merge(
                        scores, eval_dataset=None
                    )
                    stats["promotion"] = {
                        "promoted": len(promoted),
                        "failed": len(failed),
                        "details": promoted,
                    }

                    # Register promoted knowledge
                    for p in promoted:
                        self.knowledge_registry.register_promotion(
                            promotion_result={
                                "cycle": self._training_cycle,
                                "promoted_layers": [p["layer"]],
                                "scores": scores,
                                "confidence": 0.8,
                            },
                            model=self._model,
                            source_experiences=[],
                        )

            except Exception as e:
                logger.error("Training cycle failed: %s", e)
                stats["training"] = {"error": str(e)}
        else:
            stats["training"] = {"status": "no_model_loaded"}

        # 4. Resolve ambiguities
        try:
            resolved, still_pending = self.ambiguity_resolver.review_cycle(
                model=self._model, current_cycle=self._training_cycle
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
        The 'reflection' cycle. Review promoted knowledge and
        evaluate reward system health.

        Returns:
            Dictionary of review statistics.
        """
        stats: Dict[str, Any] = {
            "timestamp": time.time(),
            "reviews": [],
            "meta_rewards": None,
        }

        # 1. Review knowledge
        candidates = self.knowledge_registry.get_review_candidates()
        for knowledge_id, priority in candidates[:10]:  # cap at 10 per cycle
            try:
                result = self.belief_reviewer.review(knowledge_id)
                stats["reviews"].append(result)
            except Exception as e:
                logger.error("Review failed for %s: %s", knowledge_id, e)

        # 2. Meta-reward evaluation
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
            "interaction_count": self._interaction_count,
            "training_cycle": self._training_cycle,
            "experience_buffer_size": len(self.experience_buffer.buffer),
            "ambiguity_buffer_size": len(self.ambiguity_buffer.active_ambiguities),
            "knowledge_count": len(self.knowledge_registry.knowledge),
            "temporal_tracker_size": len(self.temporal_tracker.active_rewards),
            "pending_inquiries": len(self.inquiry_system.pending_inquiries),
            "discarded_count": len(self._discard_log),
            "model_loaded": self._model is not None,
            "domain_frequencies": self.domain_tagger.get_all_frequencies(),
        }

    def _generate(self, user_input: str) -> str:
        """Generate a response from the model."""
        if self._model is None or self._tokenizer is None:
            return f"[openMind: no model loaded - recording interaction for future training]"

        try:
            import torch

            inputs = self._tokenizer(user_input, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return response.strip()

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"[openMind: generation error - {e}]"

    def _evaluate_model(self, model: Any, dataset: Any) -> float:
        """Evaluate model on a benchmark dataset."""
        # TODO: Implement proper evaluation pipeline
        # For now, return a baseline score
        return 0.5
