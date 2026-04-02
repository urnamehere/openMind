"""Training cycle orchestration -- the nightly "sleep" phase.

Gathers high-signal recent experiences plus a replay sample, then fine-tunes
a LoRA adapter with an EWC regularisation term to prevent catastrophic
forgetting.  Designed to be invoked on a schedule (e.g. nightly cron) rather
than on every request.
"""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from openmind.core.experience import Experience, ExperienceBuffer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Knobs for a single training cycle.

    Attributes:
        recent_min_reward: Minimum reward to include a recent experience.
        recent_max_samples: Cap on high-signal recent experiences.
        replay_samples: Number of uniformly-sampled replay experiences.
        epochs: Training epochs per cycle.
        learning_rate: Peak learning rate for the LoRA adapter.
        batch_size: Micro-batch size (before gradient accumulation).
        gradient_accumulation_steps: Steps before an optimiser update.
        max_seq_length: Maximum token length for training examples.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout on LoRA layers.
        lora_target_modules: Which linear layers get LoRA adapters.
        ewc_lambda: Elastic Weight Consolidation penalty multiplier.
            Set to 0.0 to disable EWC entirely.
        ewc_sample_size: Number of experiences used to estimate Fisher info.
        device: Torch device string (e.g. "cuda", "cpu", "auto").
    """

    recent_min_reward: float = 0.3
    recent_max_samples: int = 256
    replay_samples: int = 64
    epochs: int = 3
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    ewc_lambda: float = 5000.0
    ewc_sample_size: int = 128
    device: str = "auto"


# -----------------------------------------------------------------------
# Cycle statistics
# -----------------------------------------------------------------------

@dataclass
class CycleStats:
    """Summary returned after a training cycle completes."""

    cycle_id: str
    total_experiences: int
    recent_count: int
    replay_count: int
    training_loss_start: float
    training_loss_end: float
    ewc_penalty_mean: float
    wall_seconds: float
    epochs_completed: int


# -----------------------------------------------------------------------
# Manager
# -----------------------------------------------------------------------

class TrainingCycleManager:
    """Orchestrates a single training cycle ("nightly sleep").

    Parameters:
        base_model: A ``transformers.PreTrainedModel`` (or compatible) used
            as the frozen reference for EWC.
        adapter_model: The LoRA-wrapped model that will be trained.
        experience_buffer: Source of training experiences.
        config: Hyperparameters for the cycle.
    """

    def __init__(
        self,
        base_model: Any,
        adapter_model: Any,
        experience_buffer: ExperienceBuffer,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.base_model = base_model
        self.adapter_model = adapter_model
        self.experience_buffer = experience_buffer
        self.config = config or TrainingConfig()

        # Lazy-loaded heavy deps
        self._torch: Any = None
        self._transformers: Any = None

    # ------------------------------------------------------------------
    # Lazy imports
    # ------------------------------------------------------------------

    @property
    def torch(self) -> Any:
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def transformers(self) -> Any:
        if self._transformers is None:
            import transformers
            self._transformers = transformers
        return self._transformers

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_device(self) -> Any:
        """Return a ``torch.device`` respecting the config."""
        torch = self.torch
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(self.config.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_cycle(self, cycle_id: Optional[str] = None) -> CycleStats:
        """Execute a full training cycle.

        1. Gather training data (recent high-signal + replay).
        2. Train LoRA adapter with EWC regularisation.
        3. Return cycle statistics.

        Parameters:
            cycle_id: Optional human-readable identifier.  Defaults to an
                ISO timestamp.

        Returns:
            :class:`CycleStats` summarising the cycle.
        """
        import time as _time
        from datetime import datetime, timezone

        if cycle_id is None:
            cycle_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        t0 = _time.monotonic()
        logger.info("=== Training cycle %s starting ===", cycle_id)

        # -- 1. Gather data ------------------------------------------------
        recent = self.experience_buffer.get_training_batch(
            min_reward=self.config.recent_min_reward,
            max_samples=self.config.recent_max_samples,
        )
        replay = self.experience_buffer.get_replay_sample(
            n=self.config.replay_samples,
        )
        # De-duplicate by experience_id, keeping higher reward copy
        seen: Dict[str, Experience] = {}
        for exp in recent + replay:
            prev = seen.get(exp.experience_id)
            if prev is None or exp.reward_signal > prev.reward_signal:
                seen[exp.experience_id] = exp
        training_data = list(seen.values())

        if not training_data:
            logger.warning("No training data available -- skipping cycle.")
            return CycleStats(
                cycle_id=cycle_id,
                total_experiences=0,
                recent_count=0,
                replay_count=0,
                training_loss_start=0.0,
                training_loss_end=0.0,
                ewc_penalty_mean=0.0,
                wall_seconds=_time.monotonic() - t0,
                epochs_completed=0,
            )

        logger.info(
            "Gathered %d unique experiences (%d recent, %d replay)",
            len(training_data),
            len(recent),
            len(replay),
        )

        # -- 2. Train -------------------------------------------------------
        loss_start, loss_end, ewc_mean, epochs_done = self._train_adapter(
            training_data
        )

        wall = _time.monotonic() - t0
        stats = CycleStats(
            cycle_id=cycle_id,
            total_experiences=len(training_data),
            recent_count=len(recent),
            replay_count=len(replay),
            training_loss_start=loss_start,
            training_loss_end=loss_end,
            ewc_penalty_mean=ewc_mean,
            wall_seconds=wall,
            epochs_completed=epochs_done,
        )
        logger.info(
            "=== Cycle %s done in %.1fs | loss %.4f -> %.4f ===",
            cycle_id,
            wall,
            loss_start,
            loss_end,
        )
        return stats

    # ------------------------------------------------------------------
    # Training internals
    # ------------------------------------------------------------------

    def _train_adapter(
        self, training_data: List[Experience]
    ) -> tuple[float, float, float, int]:
        """Fine-tune the LoRA adapter with EWC regularisation.

        Returns:
            (loss_start, loss_end, mean_ewc_penalty, epochs_completed)
        """
        torch = self.torch
        device = self._resolve_device()

        # Prepare tokenised dataset
        formatted = self._format_training_data(training_data)
        dataset = self._build_dataset(formatted)

        # Snapshot adapter weights *before* training for EWC reference
        fisher_diag, ewc_means = self._estimate_fisher(dataset, device)

        self.adapter_model.to(device)
        self.adapter_model.train()

        # Build optimiser -- only LoRA parameters
        trainable_params = [
            p for p in self.adapter_model.parameters() if p.requires_grad
        ]
        optimiser = torch.optim.AdamW(
            trainable_params, lr=self.config.learning_rate
        )

        # Simple linear warmup + cosine schedule
        total_steps = (
            max(len(dataset), 1)
            // self.config.batch_size
            // self.config.gradient_accumulation_steps
            * self.config.epochs
        )
        total_steps = max(total_steps, 1)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=total_steps
        )

        # Training loop
        loss_start: Optional[float] = None
        loss_end: float = 0.0
        ewc_penalties: List[float] = []
        global_step = 0
        epochs_completed = 0

        for epoch in range(self.config.epochs):
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, len(indices), self.config.batch_size):
                batch_indices = indices[
                    batch_start : batch_start + self.config.batch_size
                ]
                if not batch_indices:
                    continue

                batch = self._collate(dataset, batch_indices, device)

                outputs = self.adapter_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                ce_loss = outputs.loss

                # Apply per-example training weights
                weights = batch.get("training_weights")
                if weights is not None:
                    ce_loss = ce_loss * weights.mean()

                # EWC penalty
                ewc_pen = self._ewc_penalty(fisher_diag, ewc_means)
                ewc_penalties.append(ewc_pen.item())
                loss = ce_loss + self.config.ewc_lambda * ewc_pen

                scaled_loss = loss / self.config.gradient_accumulation_steps
                scaled_loss.backward()

                global_step += 1
                if global_step % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimiser.step()
                    scheduler.step()
                    optimiser.zero_grad()

                cur_loss = loss.item()
                epoch_loss += cur_loss
                n_batches += 1

                if loss_start is None:
                    loss_start = cur_loss
                loss_end = cur_loss

            if n_batches > 0:
                avg = epoch_loss / n_batches
                logger.info("Epoch %d/%d  avg_loss=%.4f", epoch + 1, self.config.epochs, avg)
            epochs_completed += 1

        # Flush remaining gradients
        if global_step % self.config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimiser.step()
            optimiser.zero_grad()

        mean_ewc = sum(ewc_penalties) / max(len(ewc_penalties), 1)
        return (loss_start or 0.0, loss_end, mean_ewc, epochs_completed)

    # ------------------------------------------------------------------
    # EWC helpers
    # ------------------------------------------------------------------

    def _estimate_fisher(
        self,
        dataset: List[Dict[str, Any]],
        device: Any,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Estimate diagonal Fisher information and save parameter means.

        Uses a sample of the dataset to compute the empirical Fisher
        diagonal, which penalises changes to parameters that were important
        for previous data.
        """
        torch = self.torch

        fisher: Dict[str, Any] = {}
        means: Dict[str, Any] = {}

        # Snapshot current parameters
        for name, param in self.adapter_model.named_parameters():
            if param.requires_grad:
                means[name] = param.data.clone()
                fisher[name] = torch.zeros_like(param.data)

        if self.config.ewc_lambda == 0.0:
            return fisher, means

        self.adapter_model.eval()
        sample_indices = list(range(min(len(dataset), self.config.ewc_sample_size)))

        import random as _random
        if len(dataset) > self.config.ewc_sample_size:
            sample_indices = _random.sample(range(len(dataset)), self.config.ewc_sample_size)

        n_samples = 0
        for idx in sample_indices:
            item = dataset[idx]
            input_ids = torch.tensor([item["input_ids"]], device=device)
            attention_mask = torch.tensor([item["attention_mask"]], device=device)
            labels = torch.tensor([item["labels"]], device=device)

            self.adapter_model.zero_grad()
            outputs = self.adapter_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()

            for name, param in self.adapter_model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data.pow(2)
            n_samples += 1

        # Average
        if n_samples > 0:
            for name in fisher:
                fisher[name] /= n_samples

        self.adapter_model.zero_grad()
        return fisher, means

    def _ewc_penalty(
        self,
        fisher: Dict[str, Any],
        means: Dict[str, Any],
    ) -> Any:
        """Compute the EWC penalty: 0.5 * sum_i F_i * (theta_i - theta*_i)^2."""
        torch = self.torch
        penalty = torch.tensor(0.0, device=self._resolve_device())

        if self.config.ewc_lambda == 0.0:
            return penalty

        for name, param in self.adapter_model.named_parameters():
            if name in fisher and param.requires_grad:
                diff = param - means[name]
                penalty = penalty + (fisher[name] * diff.pow(2)).sum()
        return 0.5 * penalty

    # ------------------------------------------------------------------
    # Data formatting
    # ------------------------------------------------------------------

    def _format_training_data(
        self, experiences: List[Experience]
    ) -> List[Dict[str, Any]]:
        """Convert :class:`Experience` instances to prompt/completion dicts.

        Each entry is ``{"text": ..., "training_weight": ...}`` suitable for
        causal-LM training.
        """
        formatted: List[Dict[str, Any]] = []
        for exp in experiences:
            text = (
                f"### Instruction\n{exp.input_context}\n\n"
                f"### Response\n{exp.output}"
            )
            formatted.append({
                "text": text,
                "training_weight": exp.training_weight,
            })
        return formatted

    def _build_dataset(
        self, formatted: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Tokenise formatted examples and return a list of dicts.

        Each dict has keys ``input_ids``, ``attention_mask``, ``labels``, and
        ``training_weight``.
        """
        tokenizer = self._get_tokenizer()
        dataset: List[Dict[str, Any]] = []
        for item in formatted:
            encoded = tokenizer(
                item["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0).tolist()
            attention_mask = encoded["attention_mask"].squeeze(0).tolist()
            dataset.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.copy(),
                "training_weight": item["training_weight"],
            })
        return dataset

    def _collate(
        self,
        dataset: List[Dict[str, Any]],
        indices: List[int],
        device: Any,
    ) -> Dict[str, Any]:
        """Stack a micro-batch onto the target device."""
        torch = self.torch
        batch_items = [dataset[i] for i in indices]
        return {
            "input_ids": torch.tensor(
                [b["input_ids"] for b in batch_items], device=device
            ),
            "attention_mask": torch.tensor(
                [b["attention_mask"] for b in batch_items], device=device
            ),
            "labels": torch.tensor(
                [b["labels"] for b in batch_items], device=device
            ),
            "training_weights": torch.tensor(
                [b["training_weight"] for b in batch_items],
                dtype=torch.float32,
                device=device,
            ),
        }

    def _get_tokenizer(self) -> Any:
        """Return the tokenizer associated with the base model.

        Tries ``adapter_model.config._name_or_path`` first, then falls back
        to ``base_model.config._name_or_path``.
        """
        from transformers import AutoTokenizer

        name = getattr(
            getattr(self.adapter_model, "config", None), "_name_or_path", None
        ) or getattr(
            getattr(self.base_model, "config", None), "_name_or_path", None
        )
        if name is None:
            raise RuntimeError(
                "Cannot determine tokenizer: neither adapter_model nor "
                "base_model exposes config._name_or_path"
            )
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


# bring random into scope for the training loop shuffle
import random
