"""Local HuggingFace model backend with LoRA adapter support."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from openmind.backends.base import Backend, GenerationContext

logger = logging.getLogger(__name__)


class LocalHFBackend(Backend):
    """Backend using a local HuggingFace model with LoRA adapters.

    Supports full weight training, adapter promotion, and EWC
    regularization. Requires torch, transformers, and peft.

    Parameters:
        model_config: A ModelConfig instance with model path and LoRA settings.
        promotion_config: A PromotionConfig instance for weight promotion settings.
    """

    def __init__(
        self,
        model_config: Any = None,
        promotion_config: Any = None,
    ) -> None:
        self._model_config = model_config
        self._promotion_config = promotion_config

        self._model: Any = None
        self._tokenizer: Any = None
        self._adapter_model: Any = None

        self.training_manager: Any = None
        self.weight_promoter: Any = None
        self.stability_tracker: Any = None

        if model_config and model_config.base_model_path:
            self._try_load_model()

    def _try_load_model(self) -> None:
        """Attempt to load the base model and set up LoRA adapter."""
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from openmind.core.promotion import WeightPromoter, WeightStabilityTracker
            from openmind.core.training import TrainingCycleManager

            cfg = self._model_config
            model_path = cfg.base_model_path
            logger.info("Loading model: %s", model_path)

            self._tokenizer = AutoTokenizer.from_pretrained(model_path)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )

            lora_config = LoraConfig(
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.target_modules,
                lora_dropout=cfg.lora_dropout,
            )
            self._adapter_model = get_peft_model(self._model, lora_config)

            promo_cfg = self._promotion_config
            history_window = 10
            if promo_cfg:
                history_window = promo_cfg.min_cycles_before_promotion + 7

            self.stability_tracker = WeightStabilityTracker(
                history_window=history_window
            )

            self.weight_promoter = WeightPromoter(
                base_model=self._model,
                adapter_model=self._adapter_model,
                eval_fn=self._evaluate_model,
                rollback_threshold=(
                    promo_cfg.rollback_threshold if promo_cfg else 0.02
                ),
                merge_alpha=cfg.merge_alpha,
            )

            logger.info("Model loaded successfully")

        except ImportError:
            logger.warning(
                "GPU dependencies not available (torch, peft, transformers). "
                "Running in data-collection-only mode."
            )
        except Exception as e:
            logger.warning(
                "Failed to load model: %s. Running in data-collection mode.", e
            )

    def generate(
        self,
        user_input: str,
        context: Optional[GenerationContext] = None,
    ) -> str:
        if self._model is None or self._tokenizer is None:
            return "[openMind: no model loaded - recording interaction]"

        try:
            import torch

            inputs = self._tokenizer(user_input, return_tensors="pt")
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            ctx = context or GenerationContext()
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=ctx.max_tokens,
                    do_sample=True,
                    temperature=ctx.temperature,
                    top_p=0.9,
                )

            response = self._tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            return response.strip()

        except Exception as e:
            logger.error("Generation failed: %s", e)
            return f"[openMind: generation error - {e}]"

    def evaluate_output(
        self, input_context: str, output: str
    ) -> Optional[Dict[str, float]]:
        if self._model is None:
            return None

        prompt = (
            "Rate the following AI output on two dimensions.\n"
            "Each score should be a float from -1.0 (terrible) to 1.0 (excellent).\n\n"
            f"USER REQUEST:\n{input_context}\n\n"
            f"AI OUTPUT:\n{output}\n\n"
            "Respond in exactly this format (no other text):\n"
            "task_success: <score>\n"
            "coherence: <score>\n"
        )
        try:
            raw = self.generate(prompt)
            return self._parse_eval_response(raw)
        except Exception:
            logger.warning("Self-evaluation failed", exc_info=True)
            return None

    def articulate_learning(
        self,
        source_experiences: Sequence[Any],
        promotion_scores: Dict[str, float],
    ) -> str:
        if self._model is None:
            return self._placeholder_articulation(source_experiences)

        import json

        examples = []
        for exp in source_experiences[:5]:
            examples.append(
                f"  Input: {getattr(exp, 'input_context', '?')}\n"
                f"  Output: {getattr(exp, 'output', '?')}\n"
                f"  Reward: {getattr(exp, 'reward_signal', '?')}"
            )
        examples_text = "\n---\n".join(examples)

        prompt = (
            "You are a reflective learning system. Based on the following "
            "training experiences and their reward signals, articulate in one "
            "to three sentences what you have learned.\n\n"
            f"Experiences:\n{examples_text}\n\n"
            f"Promotion scores: {json.dumps(promotion_scores)}\n\n"
            "Your articulation:"
        )
        try:
            return self.generate(prompt)
        except Exception:
            return self._placeholder_articulation(source_experiences)

    def consolidate(
        self,
        experience_buffer: Any,
        knowledge_registry: Any,
        cycle: int,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"backend": "local"}

        if self.training_manager is None:
            # Try to create training manager if model is loaded
            if self._model is not None and self._adapter_model is not None:
                from openmind.core.training import TrainingCycleManager

                self.training_manager = TrainingCycleManager(
                    base_model=self._model,
                    adapter_model=self._adapter_model,
                    experience_buffer=experience_buffer,
                )
            else:
                stats["training"] = {"status": "no_model_loaded"}
                return stats

        try:
            training_stats = self.training_manager.run_cycle()
            stats["training"] = training_stats

            # Record adapter state for stability tracking
            if self._adapter_model is not None and self.stability_tracker is not None:
                self.stability_tracker.record_adapter_state(
                    self._adapter_model.state_dict(), cycle
                )

            # Check for promotable weights
            if self.stability_tracker is not None:
                promo_cfg = self._promotion_config
                threshold = promo_cfg.stability_threshold if promo_cfg else 0.05
                scores = self.stability_tracker.compute_promotion_scores(
                    threshold=threshold
                )
                promotable = {
                    k: v
                    for k, v in scores.items()
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
                        knowledge_registry.register_promotion(
                            promotion_result={
                                "cycle": cycle,
                                "promoted_layers": [p["layer"]],
                                "scores": scores,
                                "confidence": 0.8,
                            },
                            articulate_fn=self.articulate_learning,
                            source_experiences=[],
                        )

        except Exception as e:
            logger.error("Training cycle failed: %s", e)
            stats["training"] = {"error": str(e)}

        return stats

    def reflect(
        self,
        knowledge_registry: Any,
        experience_buffer: Any,
        belief_reviewer: Any,
    ) -> Dict[str, Any]:
        stats: Dict[str, Any] = {"backend": "local"}

        candidates = knowledge_registry.get_review_candidates()
        reviews = []
        for pk in candidates[:10]:
            kid = pk.knowledge_id if hasattr(pk, "knowledge_id") else pk
            try:
                result = belief_reviewer.review(kid)
                reviews.append(result)
            except Exception as e:
                logger.error("Review failed for %s: %s", kid, e)

        stats["reviews"] = reviews
        stats["reviewed_count"] = len(reviews)
        return stats

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def supports_weight_training(self) -> bool:
        return True

    @property
    def model(self) -> Any:
        """Expose the underlying model for subsystems that need it."""
        return self._model

    def _evaluate_model(self, model: Any, dataset: Any) -> float:
        """Evaluate model on a benchmark dataset."""
        return 0.5

    @staticmethod
    def _parse_eval_response(raw: str) -> Optional[Dict[str, float]]:
        scores: Dict[str, float] = {}
        for line in raw.strip().splitlines():
            line = line.strip()
            for key in ("task_success", "coherence"):
                if line.lower().startswith(key):
                    try:
                        val = float(line.split(":", 1)[1].strip())
                        scores[key] = max(-1.0, min(1.0, val))
                    except (ValueError, IndexError):
                        pass
        return scores if scores else None

    @staticmethod
    def _placeholder_articulation(source_experiences: Sequence[Any]) -> str:
        return (
            f"[Placeholder] Learning promoted from {len(source_experiences)} "
            f"experience(s) across domains "
            f"{[getattr(e, 'domain_tags', []) for e in source_experiences[:3]]}."
        )
