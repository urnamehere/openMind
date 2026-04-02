"""Weight promotion -- selective merge of stable adapter weights into base.

Implements the biological analogy of memory consolidation: adapter weights
that have stabilised across multiple training cycles (low variance) are
"promoted" into the base model, freeing adapter capacity for new learning.

The flow is:

1. After each training cycle, :class:`WeightStabilityTracker` records a
   snapshot of the adapter state dict.
2. When enough snapshots exist, :meth:`compute_promotion_scores` identifies
   layers whose weights have converged (low coefficient of variation).
3. :class:`WeightPromoter` performs a guarded, layer-by-layer merge of those
   stable weights into the base model, rolling back any merge that causes
   evaluation degradation.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Stability tracking
# -----------------------------------------------------------------------

@dataclass
class _Snapshot:
    """Internal record of a single adapter state snapshot."""

    cycle_id: str
    timestamp: float
    state_dict: Dict[str, Any]  # name -> tensor (kept on CPU)


class WeightStabilityTracker:
    """Track per-layer adapter weight stability across training cycles.

    After each training cycle the caller passes the adapter's
    ``state_dict()`` to :meth:`record_adapter_state`.  The tracker keeps
    the most recent *history_window* snapshots and can compute a
    *promotion score* for every tracked layer.

    The promotion score is ``1 - CV`` where *CV* is the coefficient of
    variation (std / mean of absolute values) of each parameter across
    the snapshot window.  A score close to 1.0 means the weights barely
    moved -- they are ready for promotion.

    Parameters:
        history_window: Maximum number of snapshots to retain.
        storage_dir: Optional directory to persist snapshots to disk.  If
            ``None``, snapshots are held in memory only.
    """

    def __init__(
        self,
        history_window: int = 5,
        storage_dir: Optional[str] = None,
    ) -> None:
        if history_window < 2:
            raise ValueError("history_window must be >= 2 to compute variance")
        self.history_window = history_window
        self._snapshots: List[_Snapshot] = []
        self._storage_dir: Optional[Path] = None
        if storage_dir is not None:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_adapter_state(
        self,
        adapter_state_dict: Dict[str, Any],
        cycle_id: str,
    ) -> None:
        """Record a new snapshot of the adapter weights.

        Parameters:
            adapter_state_dict: The adapter model's ``state_dict()``.
                Tensors are cloned and moved to CPU for storage.
            cycle_id: Identifier for the training cycle that produced
                these weights.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for WeightStabilityTracker. "
                "Install it with: pip install openmind[gpu]"
            )

        cpu_state: Dict[str, Any] = {}
        for name, tensor in adapter_state_dict.items():
            if isinstance(tensor, torch.Tensor):
                cpu_state[name] = tensor.detach().cpu().clone()
            else:
                cpu_state[name] = tensor

        snap = _Snapshot(
            cycle_id=cycle_id,
            timestamp=time.time(),
            state_dict=cpu_state,
        )
        self._snapshots.append(snap)

        # Trim to window
        if len(self._snapshots) > self.history_window:
            self._snapshots = self._snapshots[-self.history_window :]

        # Persist if configured
        if self._storage_dir is not None:
            path = self._storage_dir / f"snapshot_{cycle_id}.pt"
            torch.save(cpu_state, path)
            logger.debug("Saved adapter snapshot to %s", path)

        logger.info(
            "Recorded adapter snapshot for cycle %s (%d/%d in window)",
            cycle_id,
            len(self._snapshots),
            self.history_window,
        )

    def compute_promotion_scores(
        self,
        threshold: float = 0.05,
    ) -> Dict[str, float]:
        """Compute per-layer promotion scores from recent snapshots.

        A promotion score of 1.0 means the layer weights are perfectly
        stable across the snapshot window.  Layers with a score above
        ``1 - threshold`` are candidates for promotion.

        Parameters:
            threshold: Maximum coefficient of variation for a layer to be
                considered "stable".  Default 0.05 (5 %).

        Returns:
            Dictionary mapping layer name to its promotion score.  Only
            layers that appear in *all* snapshots are included.

        Raises:
            ValueError: If fewer than 2 snapshots have been recorded.
        """
        if len(self._snapshots) < 2:
            raise ValueError(
                f"Need at least 2 snapshots to compute stability, "
                f"have {len(self._snapshots)}"
            )

        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch is required for compute_promotion_scores")

        # Identify layers present in every snapshot
        all_keys: Optional[set] = None
        for snap in self._snapshots:
            keys = set(snap.state_dict.keys())
            all_keys = keys if all_keys is None else all_keys.intersection(keys)
        if not all_keys:
            logger.warning("No common layers found across snapshots")
            return {}

        scores: Dict[str, float] = {}
        for layer_name in sorted(all_keys):
            tensors = [snap.state_dict[layer_name] for snap in self._snapshots]
            # Skip non-float tensors (e.g. integer index buffers)
            if not tensors[0].is_floating_point():
                continue

            stacked = torch.stack(tensors, dim=0).float()
            mean_abs = stacked.abs().mean()

            if mean_abs < 1e-10:
                # Near-zero weights are trivially stable
                scores[layer_name] = 1.0
                continue

            std = stacked.std(dim=0).mean()
            cv = (std / mean_abs).item()
            score = max(0.0, 1.0 - cv)
            scores[layer_name] = score

        n_stable = sum(1 for s in scores.values() if s >= (1.0 - threshold))
        logger.info(
            "Promotion scores computed: %d layers, %d stable (threshold=%.3f)",
            len(scores),
            n_stable,
            threshold,
        )
        return scores

    @property
    def snapshot_count(self) -> int:
        """Number of snapshots currently in the window."""
        return len(self._snapshots)

    @property
    def cycle_ids(self) -> List[str]:
        """Cycle IDs of retained snapshots, oldest first."""
        return [s.cycle_id for s in self._snapshots]


# -----------------------------------------------------------------------
# Promotion log entry
# -----------------------------------------------------------------------

@dataclass
class PromotionRecord:
    """Record of a single layer promotion event."""

    layer_name: str
    adapter_key: str
    base_key: str
    promotion_score: float
    merge_alpha: float
    eval_before: float
    eval_after: float
    rolled_back: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "adapter_key": self.adapter_key,
            "base_key": self.base_key,
            "promotion_score": self.promotion_score,
            "merge_alpha": self.merge_alpha,
            "eval_before": self.eval_before,
            "eval_after": self.eval_after,
            "rolled_back": self.rolled_back,
            "timestamp": self.timestamp,
        }


# -----------------------------------------------------------------------
# Weight Promoter
# -----------------------------------------------------------------------

class WeightPromoter:
    """Selectively merge stable adapter weights into the base model.

    This is the core "memory consolidation" step.  Weights that have been
    consistently stable across several training cycles (as determined by
    :class:`WeightStabilityTracker`) are blended into the base model at a
    conservative ``merge_alpha``.  After each layer merge an evaluation is
    run; if performance degrades beyond ``rollback_threshold`` the merge is
    undone.

    Parameters:
        base_model: The base ``PreTrainedModel``.
        adapter_model: The LoRA-adapted model wrapping *base_model*.
        eval_fn: Callable that accepts a model and an evaluation dataset
            and returns a scalar metric (higher is better).  Used to gate
            individual layer merges.
        merge_alpha: Blending factor in [0, 1].  ``0`` means no change;
            ``1`` means full replacement with the adapter-derived value.
        rollback_threshold: Maximum allowed relative degradation (e.g.
            0.02 = 2 %) before a merge is rolled back.
        promotion_log_path: Optional path to a JSONL file where promotion
            records are appended.
    """

    def __init__(
        self,
        base_model: Any,
        adapter_model: Any,
        eval_fn: Callable[..., float],
        merge_alpha: float = 0.3,
        rollback_threshold: float = 0.02,
        promotion_log_path: Optional[str] = None,
    ) -> None:
        if not 0.0 < merge_alpha <= 1.0:
            raise ValueError(f"merge_alpha must be in (0, 1], got {merge_alpha}")
        if rollback_threshold < 0.0:
            raise ValueError(
                f"rollback_threshold must be >= 0, got {rollback_threshold}"
            )

        self.base_model = base_model
        self.adapter_model = adapter_model
        self.eval_fn = eval_fn
        self.merge_alpha = merge_alpha
        self.rollback_threshold = rollback_threshold

        self.promotion_log: List[PromotionRecord] = []
        self._log_path: Optional[Path] = None
        if promotion_log_path is not None:
            self._log_path = Path(promotion_log_path)
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # LoRA layer-name mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_adapter_to_base(adapter_key: str) -> Optional[str]:
        """Map a LoRA adapter parameter name to its base-model counterpart.

        Typical LoRA adapter keys follow the pattern::

            base_model.model.<path>.lora_A.weight
            base_model.model.<path>.lora_B.weight

        The original weight lives at ``<path>.weight`` in the base model.

        Returns:
            The base model parameter name, or ``None`` if the adapter key
            does not look like a LoRA parameter.
        """
        # Pattern 1: peft-style "base_model.model.{path}.lora_{A,B}.{suffix}"
        m = re.match(
            r"^base_model\.model\.(.+)\.lora_[AB](?:\.default)?\.weight$",
            adapter_key,
        )
        if m:
            return f"{m.group(1)}.weight"

        # Pattern 2: simplified "{path}.lora_{A,B}.weight"
        m = re.match(r"^(.+)\.lora_[AB](?:\.default)?\.weight$", adapter_key)
        if m:
            return f"{m.group(1)}.weight"

        return None

    # ------------------------------------------------------------------
    # Core merge logic
    # ------------------------------------------------------------------

    def _compute_lora_delta(
        self,
        adapter_state: Dict[str, Any],
        base_path: str,
        adapter_keys: List[str],
        scaling: float,
    ) -> Any:
        """Compute the LoRA delta W = (B @ A) * scaling for a given layer.

        Parameters:
            adapter_state: Full adapter state dict.
            base_path: The base layer path (e.g. "model.layers.0.self_attn.q_proj").
            adapter_keys: Adapter keys that map to this base path.
            scaling: LoRA scaling factor (alpha / r).

        Returns:
            The delta tensor on the same device as the A matrix.
        """
        lora_a: Optional[Any] = None
        lora_b: Optional[Any] = None

        for key in adapter_keys:
            if ".lora_A" in key:
                lora_a = adapter_state[key]
            elif ".lora_B" in key:
                lora_b = adapter_state[key]

        if lora_a is None or lora_b is None:
            raise ValueError(
                f"Could not find both lora_A and lora_B for base path "
                f"'{base_path}' among keys: {adapter_keys}"
            )

        # delta = B @ A  (B is out_features x r, A is r x in_features)
        import torch

        delta = torch.mm(lora_b.float(), lora_a.float()) * scaling
        return delta

    def _get_lora_scaling(self) -> float:
        """Extract LoRA scaling factor from the adapter model config.

        Falls back to ``alpha / r`` from peft config, or 1.0 if unavailable.
        """
        config = getattr(self.adapter_model, "peft_config", None)
        if config is not None:
            # peft_config is usually a dict keyed by adapter name
            if isinstance(config, dict):
                cfg = next(iter(config.values()), None)
            else:
                cfg = config
            if cfg is not None:
                alpha = getattr(cfg, "lora_alpha", None)
                r = getattr(cfg, "r", None)
                if alpha is not None and r is not None and r > 0:
                    return float(alpha) / float(r)
        logger.warning(
            "Could not determine LoRA scaling from adapter config; using 1.0"
        )
        return 1.0

    def selective_merge(
        self,
        promotion_scores: Dict[str, float],
        eval_dataset: Any,
        score_threshold: float = 0.95,
    ) -> List[PromotionRecord]:
        """Merge stable adapter weights into the base model.

        For each LoRA layer whose promotion score meets *score_threshold*:

        1. Compute the LoRA delta: ``delta = (B @ A) * scaling``.
        2. Blend into the base weight:
           ``base_new = base + merge_alpha * delta``.
        3. Evaluate the model on *eval_dataset*.
        4. If performance dropped by more than *rollback_threshold*,
           restore the original base weight.
        5. Otherwise, zero out the promoted adapter weights (A and B)
           to free adapter capacity.

        Parameters:
            promotion_scores: Layer-name -> score mapping from
                :meth:`WeightStabilityTracker.compute_promotion_scores`.
            eval_dataset: Passed to ``self.eval_fn`` for gating.
            score_threshold: Minimum promotion score for a layer to be
                considered.  Defaults to 0.95 (i.e. CV < 5 %).

        Returns:
            List of :class:`PromotionRecord` for this promotion round.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "PyTorch is required for weight promotion. "
                "Install it with: pip install openmind[gpu]"
            )

        records: List[PromotionRecord] = []

        # Group adapter keys by their base-model counterpart
        adapter_state = self.adapter_model.state_dict()
        base_state = dict(self.base_model.named_parameters())

        # Build mapping: base_path -> list of adapter keys
        base_to_adapter: Dict[str, List[str]] = defaultdict(list)
        for akey in adapter_state:
            bkey = self._map_adapter_to_base(akey)
            if bkey is not None:
                base_to_adapter[bkey].append(akey)

        if not base_to_adapter:
            logger.warning("No LoRA layers found in adapter state dict")
            return records

        # Determine which layers to promote
        lora_scaling = self._get_lora_scaling()
        candidates: List[Tuple[str, List[str], float]] = []

        for base_key, adapter_keys in sorted(base_to_adapter.items()):
            # Check if any adapter key has a score above threshold
            best_score = 0.0
            for akey in adapter_keys:
                score = promotion_scores.get(akey, 0.0)
                best_score = max(best_score, score)
            if best_score >= score_threshold:
                candidates.append((base_key, adapter_keys, best_score))

        if not candidates:
            logger.info(
                "No layers meet promotion threshold %.3f", score_threshold
            )
            return records

        logger.info(
            "Promoting %d / %d LoRA layers (threshold=%.3f, alpha=%.2f)",
            len(candidates),
            len(base_to_adapter),
            score_threshold,
            self.merge_alpha,
        )

        # Baseline evaluation
        self.adapter_model.eval()
        baseline_score = self.eval_fn(self.adapter_model, eval_dataset)
        logger.info("Baseline eval score: %.6f", baseline_score)

        current_score = baseline_score

        for base_key, adapter_keys, promo_score in candidates:
            if base_key not in base_state:
                logger.warning(
                    "Base parameter '%s' not found -- skipping", base_key
                )
                continue

            base_param = base_state[base_key]
            original_data = base_param.data.clone()

            # Compute and apply delta
            try:
                delta = self._compute_lora_delta(
                    adapter_state, base_key, adapter_keys, lora_scaling
                )
                delta = delta.to(device=base_param.device, dtype=base_param.dtype)
                base_param.data.add_(self.merge_alpha * delta)
            except (ValueError, RuntimeError) as exc:
                logger.error(
                    "Failed to compute delta for %s: %s", base_key, exc
                )
                continue

            # Evaluate
            eval_after = self.eval_fn(self.adapter_model, eval_dataset)
            degradation = (
                (current_score - eval_after) / abs(current_score)
                if abs(current_score) > 1e-10
                else 0.0
            )

            rolled_back = False
            if degradation > self.rollback_threshold:
                # Rollback
                base_param.data.copy_(original_data)
                logger.warning(
                    "Rolled back merge of %s: score %.6f -> %.6f "
                    "(degradation %.4f > threshold %.4f)",
                    base_key,
                    current_score,
                    eval_after,
                    degradation,
                    self.rollback_threshold,
                )
                eval_after = current_score  # score unchanged after rollback
            else:
                # Success -- zero out the promoted adapter weights
                with torch.no_grad():
                    for akey in adapter_keys:
                        if akey in adapter_state:
                            param = _get_parameter_by_name(
                                self.adapter_model, akey
                            )
                            if param is not None:
                                param.zero_()
                current_score = eval_after
                logger.info(
                    "Promoted %s (score %.3f, eval %.6f -> %.6f)",
                    base_key,
                    promo_score,
                    current_score,
                    eval_after,
                )

            record = PromotionRecord(
                layer_name=base_key,
                adapter_key=adapter_keys[0],
                base_key=base_key,
                promotion_score=promo_score,
                merge_alpha=self.merge_alpha,
                eval_before=current_score if rolled_back else baseline_score,
                eval_after=eval_after,
                rolled_back=rolled_back,
            )
            records.append(record)
            self.promotion_log.append(record)

            # Persist log entry
            if self._log_path is not None:
                with open(self._log_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record.to_dict()) + "\n")

        promoted = sum(1 for r in records if not r.rolled_back)
        rolled = sum(1 for r in records if r.rolled_back)
        logger.info(
            "Promotion complete: %d promoted, %d rolled back, "
            "eval %.6f -> %.6f",
            promoted,
            rolled,
            baseline_score,
            current_score,
        )
        return records

    def get_promotion_summary(self) -> Dict[str, Any]:
        """Return a summary of all promotion activity."""
        total = len(self.promotion_log)
        promoted = sum(1 for r in self.promotion_log if not r.rolled_back)
        rolled_back = total - promoted
        unique_layers = len({r.layer_name for r in self.promotion_log if not r.rolled_back})
        return {
            "total_attempts": total,
            "promoted": promoted,
            "rolled_back": rolled_back,
            "unique_layers_promoted": unique_layers,
            "merge_alpha": self.merge_alpha,
            "rollback_threshold": self.rollback_threshold,
        }


# -----------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------

def _get_parameter_by_name(model: Any, name: str) -> Any:
    """Retrieve a parameter from a model by its dotted name.

    Returns:
        The parameter tensor, or ``None`` if the path is invalid.
    """
    parts = name.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif hasattr(current, "_modules") and part in current._modules:
            current = current._modules[part]
        else:
            logger.debug("Cannot resolve parameter path '%s' at '%s'", name, part)
            return None

    try:
        import torch

        if isinstance(current, torch.nn.Parameter) or isinstance(current, torch.Tensor):
            return current
    except ImportError:
        pass
    return None
