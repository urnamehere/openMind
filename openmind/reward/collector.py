"""Reward signal collector for the openMind continuous learning system.

The RewardCollector gathers evidence from multiple sources (self-evaluation,
explicit user feedback, execution results, retrieval groundedness checks)
and fuses them into a single RewardSignal.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from .signal import RewardSignal

logger = logging.getLogger(__name__)


class RewardCollector:
    """Gathers reward signals from heterogeneous sources and fuses them.

    Parameters:
        model:           An LLM callable (or wrapper) used for self-evaluation.
                         Expected signature: model(prompt: str) -> str.
                         Pass *None* to disable self-eval.
        retrieval_index: An embedding-based retrieval index used to score
                         groundedness.  Expected to expose a ``search(query)``
                         method returning scored results.  Pass *None* to
                         disable groundedness checks.
    """

    # Default trust levels for each signal source (0-1 scale).
    _DEFAULT_RELIABILITY: Dict[str, float] = {
        "self_eval": 0.5,
        "explicit_feedback": 0.95,
        "execution_result": 0.90,
        "inferred_satisfaction": 0.4,
        "groundedness_check": 0.7,
        "novelty_check": 0.45,
    }

    def __init__(
        self,
        model: Any = None,
        retrieval_index: Any = None,
    ) -> None:
        self.model = model
        self.retrieval_index = retrieval_index
        self.source_reliability: Dict[str, float] = dict(self._DEFAULT_RELIABILITY)

        # Running calibration data for self-eval metacognition
        self._calibration_pairs: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def collect(
        self,
        input_context: str,
        output: str,
        user_response: Optional[str] = None,
        execution_result: Optional[Dict[str, Any]] = None,
    ) -> RewardSignal:
        """Collect reward signals from all available sources.

        Args:
            input_context:    The original user request / prompt context.
            output:           The model's generated output.
            user_response:    Optional follow-up text from the user (may
                              contain explicit feedback).
            execution_result: Optional dict describing code execution outcome,
                              e.g. ``{"success": True, "stdout": "...", "stderr": ""}``.

        Returns:
            A fused ``RewardSignal`` combining all available evidence.
        """
        signal = RewardSignal()

        # 1. Self-evaluation (requires model)
        self_scores = self._self_evaluate(input_context, output)
        if self_scores is not None:
            reliability = self.source_reliability["self_eval"]
            if self_scores.get("task_success") is not None:
                signal.task_success = self_scores["task_success"] * reliability
                signal.signal_sources["task_success"] = "self_eval"
            if self_scores.get("coherence") is not None:
                signal.coherence = self_scores["coherence"] * reliability
                signal.signal_sources["coherence"] = "self_eval"

        # 2. Explicit user feedback
        if user_response is not None:
            explicit = self._check_explicit_feedback(user_response)
            if explicit is not None:
                signal.user_satisfaction = explicit
                signal.signal_sources["user_satisfaction"] = "explicit_feedback"

        # 3. Inferred satisfaction from user behaviour
        if user_response is not None and signal.user_satisfaction is None:
            inferred = self._infer_satisfaction(input_context, output, user_response)
            if inferred is not None:
                reliability = self.source_reliability["inferred_satisfaction"]
                signal.user_satisfaction = inferred * reliability
                signal.signal_sources["user_satisfaction"] = "inferred_satisfaction"

        # 4. Execution-based task success
        if execution_result is not None:
            exec_score = self._score_execution(execution_result)
            # Execution result is high-reliability; override self-eval if present
            if exec_score is not None:
                reliability = self.source_reliability["execution_result"]
                signal.task_success = exec_score * reliability
                signal.signal_sources["task_success"] = "execution_result"

        # 5. Groundedness check (requires retrieval_index)
        groundedness = self._check_groundedness(input_context, output)
        if groundedness is not None:
            reliability = self.source_reliability["groundedness_check"]
            signal.groundedness = groundedness * reliability
            signal.signal_sources["groundedness"] = "groundedness_check"

        # 6. Novelty assessment
        novelty = self._assess_novelty(input_context, output)
        if novelty is not None:
            reliability = self.source_reliability["novelty_check"]
            signal.novelty_value = novelty * reliability
            signal.signal_sources["novelty_value"] = "novelty_check"

        return signal

    def calibrate_self_eval(
        self,
        self_eval_scores: List[float],
        human_scores: List[float],
    ) -> Dict[str, float]:
        """Calibrate self-evaluation reliability by comparing to human labels.

        This implements a simple metacognition loop: if the model's self-eval
        consistently over- or under-estimates quality relative to human ratings,
        we adjust the reliability weight accordingly.

        Args:
            self_eval_scores: Model's self-assigned quality scores (0-1 each).
            human_scores:     Corresponding human quality scores (0-1 each).

        Returns:
            Dict with calibration metrics: mean_error, bias, new_reliability.
        """
        if len(self_eval_scores) != len(human_scores) or not self_eval_scores:
            raise ValueError("Score lists must be non-empty and equal length")

        # Store pairs for future reference
        for se, hs in zip(self_eval_scores, human_scores):
            self._calibration_pairs.append({"self_eval": se, "human": hs})

        # Compute bias (positive = model overestimates itself)
        errors = [se - hs for se, hs in zip(self_eval_scores, human_scores)]
        mean_error = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)

        # Correlation-based reliability: how well does self-eval track human?
        n = len(self_eval_scores)
        mean_se = sum(self_eval_scores) / n
        mean_hs = sum(human_scores) / n

        cov = sum(
            (se - mean_se) * (hs - mean_hs)
            for se, hs in zip(self_eval_scores, human_scores)
        ) / n
        var_se = sum((se - mean_se) ** 2 for se in self_eval_scores) / n
        var_hs = sum((hs - mean_hs) ** 2 for hs in human_scores) / n

        denom = (var_se * var_hs) ** 0.5
        correlation = cov / denom if denom > 1e-9 else 0.0

        # New reliability is a blend of correlation quality and low-error bonus
        # correlation in [-1, 1]; we map to [0, 1]
        corr_component = max(0.0, (correlation + 1.0) / 2.0)
        error_component = max(0.0, 1.0 - mean_error)
        new_reliability = 0.6 * corr_component + 0.4 * error_component
        new_reliability = max(0.1, min(0.95, new_reliability))

        self.source_reliability["self_eval"] = new_reliability

        logger.info(
            "Self-eval calibration: mean_error=%.3f bias=%.3f corr=%.3f -> reliability=%.3f",
            mean_error,
            bias,
            correlation,
            new_reliability,
        )

        return {
            "mean_error": mean_error,
            "bias": bias,
            "correlation": correlation,
            "new_reliability": new_reliability,
        }

    # ------------------------------------------------------------------
    # Private signal sources
    # ------------------------------------------------------------------

    def _self_evaluate(
        self, input_context: str, output: str
    ) -> Optional[Dict[str, float]]:
        """Ask the model to rate its own output.

        Returns dict with 'task_success' and 'coherence' in [-1, 1], or None
        if no model is available.
        """
        if self.model is None:
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

        # TODO: Replace with proper LLM call once model interface is finalised.
        # The model parameter should accept a prompt string and return the
        # generated text.  For now we attempt a direct call; if the model
        # object isn't callable or doesn't return parseable text, we return
        # None gracefully.
        try:
            raw = self.model(prompt)
            return self._parse_self_eval(raw)
        except Exception:
            logger.warning("Self-evaluation failed", exc_info=True)
            return None

    @staticmethod
    def _parse_self_eval(raw: str) -> Optional[Dict[str, float]]:
        """Parse 'task_success: <float>\\ncoherence: <float>' format."""
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

    def _check_explicit_feedback(self, user_response: str) -> Optional[float]:
        """Detect explicit positive/negative feedback in user text.

        Returns a satisfaction score in [-1, 1] or None if no explicit signal.
        """
        text = user_response.lower().strip()

        # Strong positive patterns
        strong_positive = [
            r"\bthanks?\b", r"\bthank you\b", r"\bperfect\b", r"\bexcellent\b",
            r"\bgreat job\b", r"\bexactly what i (?:needed|wanted)\b",
            r"\bawesome\b", r"\bwell done\b", r"\blgtm\b",
        ]
        # Moderate positive
        moderate_positive = [
            r"\bgood\b", r"\bnice\b", r"\bhelpful\b", r"\bworks\b",
            r"\bcorrect\b", r"\bright\b", r"\byep\b", r"\byes\b",
        ]
        # Strong negative
        strong_negative = [
            r"\bwrong\b", r"\bincorrect\b", r"\bbad\b", r"\bterrible\b",
            r"\bhallucin", r"\bmade up\b", r"\bnot what i (?:asked|wanted|needed)\b",
            r"\buseless\b", r"\bawful\b",
        ]
        # Moderate negative
        moderate_negative = [
            r"\bnot quite\b", r"\bnot really\b", r"\bnot helpful\b",
            r"\bcould be better\b", r"\bmissing\b", r"\bno\b",
        ]

        score = 0.0
        matched = False

        for pattern in strong_positive:
            if re.search(pattern, text):
                score += 0.9
                matched = True
                break

        if not matched:
            for pattern in moderate_positive:
                if re.search(pattern, text):
                    score += 0.5
                    matched = True
                    break

        for pattern in strong_negative:
            if re.search(pattern, text):
                score -= 0.9
                matched = True
                break

        if score == 0.0:
            for pattern in moderate_negative:
                if re.search(pattern, text):
                    score -= 0.4
                    matched = True
                    break

        if not matched:
            return None

        return max(-1.0, min(1.0, score))

    def _infer_satisfaction(
        self, input_context: str, output: str, user_response: str
    ) -> Optional[float]:
        """Infer satisfaction from indirect behavioural cues.

        Heuristics:
        - Very short follow-up that continues the conversation -> mild positive
        - User repeats the same question -> negative (they weren't satisfied)
        - User asks a follow-up question building on the answer -> positive
        - User changes topic entirely -> neutral (no signal)
        """
        user_resp_lower = user_response.lower().strip()
        input_lower = input_context.lower().strip()

        # Repeated question detection: high token overlap with original
        input_tokens = set(input_lower.split())
        resp_tokens = set(user_resp_lower.split())
        if input_tokens and resp_tokens:
            overlap = len(input_tokens & resp_tokens) / len(input_tokens)
            if overlap > 0.7 and len(resp_tokens) < len(input_tokens) * 1.5:
                return -0.6  # They're asking the same thing again

        # Follow-up question building on the answer
        # Heuristic: response contains "?" and references words from the output
        output_tokens = set(output.lower().split())
        if "?" in user_resp_lower and resp_tokens:
            output_overlap = len(resp_tokens & output_tokens) / max(len(resp_tokens), 1)
            if output_overlap > 0.15:
                return 0.4  # Building on the answer -> mild satisfaction

        # Very short acknowledgement (1-3 words, no question mark)
        if len(user_resp_lower.split()) <= 3 and "?" not in user_resp_lower:
            return 0.3  # Brief ack -> mild positive

        return None  # Can't infer

    @staticmethod
    def _score_execution(execution_result: Dict[str, Any]) -> Optional[float]:
        """Score task success from code execution results.

        Expects a dict with at minimum a "success" boolean key.
        Optional keys: "stdout", "stderr", "exit_code", "tests_passed",
        "tests_total".
        """
        if "success" not in execution_result:
            return None

        if execution_result["success"]:
            # Check partial test results
            tests_passed = execution_result.get("tests_passed")
            tests_total = execution_result.get("tests_total")
            if tests_passed is not None and tests_total is not None and tests_total > 0:
                ratio = tests_passed / tests_total
                # Map [0, 1] to [-0.5, 1.0]
                return -0.5 + 1.5 * ratio
            return 0.9  # General success
        else:
            stderr = execution_result.get("stderr", "")
            if stderr and len(stderr) > 200:
                return -0.8  # Noisy failure
            return -0.5  # Clean failure

    def _check_groundedness(
        self, input_context: str, output: str
    ) -> Optional[float]:
        """Score how well the output is grounded in retrieved knowledge.

        Uses the retrieval_index to find relevant documents and checks
        whether key claims in the output can be traced back to sources.
        """
        if self.retrieval_index is None:
            return None

        # TODO: Implement proper claim extraction and verification.
        # The full implementation requires:
        #   1. Extract key claims/facts from `output` (needs an LLM or NLI model).
        #   2. For each claim, query self.retrieval_index.search(claim) to find
        #      supporting documents.
        #   3. Score each claim as supported/unsupported based on retrieval
        #      similarity scores.
        #   4. Aggregate into a single groundedness score.
        #
        # For now, we do a rough proxy: retrieve top docs for the input context
        # and check lexical overlap with the output.
        try:
            results = self.retrieval_index.search(input_context)
            if not results:
                return 0.0  # No sources found; can't verify

            # results expected to be list of dicts with "text" and "score" keys
            retrieved_text = " ".join(
                r.get("text", "") if isinstance(r, dict) else str(r)
                for r in results[:5]
            ).lower()
            output_tokens = set(output.lower().split())
            retrieved_tokens = set(retrieved_text.split())

            if not output_tokens:
                return 0.0

            overlap = len(output_tokens & retrieved_tokens) / len(output_tokens)
            # Map overlap ratio [0, 1] to groundedness [-0.5, 1.0]
            return -0.5 + 1.5 * min(overlap, 1.0)
        except Exception:
            logger.warning("Groundedness check failed", exc_info=True)
            return None

    def _assess_novelty(
        self, input_context: str, output: str
    ) -> Optional[float]:
        """Assess whether the output provides novel, non-obvious information.

        Uses retrieval index overlap as a proxy: low overlap with existing
        knowledge = higher novelty.  Without a retrieval index, returns None.
        """
        if self.retrieval_index is None:
            return None

        # TODO: Proper novelty detection requires:
        #   1. Embedding the output and comparing to existing memory embeddings
        #      to find the nearest neighbours.
        #   2. If the output is very close to existing memories, novelty is low.
        #   3. If it introduces new concepts/relations, novelty is high.
        #   4. Combine with a "usefulness" check so random gibberish doesn't
        #      score as "novel".
        #
        # Proxy implementation: inverse of groundedness overlap.
        try:
            results = self.retrieval_index.search(output)
            if not results:
                return 0.5  # Nothing similar found; moderately novel

            # Take best similarity score
            best_score = max(
                (r.get("score", 0.0) if isinstance(r, dict) else 0.0)
                for r in results[:3]
            )
            # High similarity = low novelty, low similarity = high novelty
            # best_score in [0, 1] -> novelty in [-0.3, 0.8]
            return 0.8 - 1.1 * best_score
        except Exception:
            logger.warning("Novelty assessment failed", exc_info=True)
            return None
