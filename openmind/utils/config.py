"""Configuration management for the openMind continuous learning system.

Provides an ``OpenMindConfig`` dataclass with every tunable parameter organised
by subsystem, plus helpers for YAML serialisation and sensible defaults.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Section dataclasses
# ======================================================================

@dataclass
class ModelConfig:
    """Parameters governing the base model and LoRA adapter."""

    base_model_path: Optional[str] = None
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    merge_alpha: float = 0.3


@dataclass
class TrainingConfig:
    """Parameters for the training cycle."""

    batch_size: int = 4
    learning_rate: float = 2e-5
    ewc_lambda: float = 0.5
    replay_ratio: float = 0.1
    min_reward_threshold: float = 0.3
    max_training_samples: int = 500


@dataclass
class DetectionConfig:
    """Thresholds for the signal-quality gatekeeper."""

    outlier_z_threshold: float = 2.5
    gaming_diversity_threshold: float = -0.1
    novelty_threshold: float = 0.6
    distribution_shift_kl_threshold: float = 0.5
    self_reinforcement_similarity: float = 0.8
    sparsity_min_coverage: float = 0.4


@dataclass
class InquiryConfig:
    """Budget and rate-limits for active inquiry."""

    ask_budget_per_session: int = 3
    ask_budget_per_day: int = 10
    cooldown_seconds: int = 120


@dataclass
class PromotionConfig:
    """Criteria for promoting adapter weights into the base model."""

    stability_threshold: float = 0.05
    rollback_threshold: float = 0.02
    min_cycles_before_promotion: int = 3


@dataclass
class ReviewConfig:
    """Spaced-repetition and deprecation settings for belief review."""

    base_review_interval_days: int = 7
    max_review_interval_days: int = 180
    deprecation_age_days: int = 30
    deprecation_min_reviews: int = 5


@dataclass
class TemporalConfig:
    """Temporal reward tracking settings."""

    cooling_off_hours: float = 1.0
    min_stability: float = 0.5


@dataclass
class CuriosityConfig:
    """Configuration for curiosity-driven autonomous exploration."""

    enabled: bool = True
    budget_per_cycle: int = 3
    min_curiosity_threshold: float = 0.3
    novelty_decay: float = 0.02
    mastery_growth_rate: float = 0.01


@dataclass
class ClaudeConfig:
    """Configuration for the Claude API backend."""

    api_key_env_var: str = "ANTHROPIC_API_KEY"
    model: str = "claude-sonnet-4-20250514"
    eval_model: str = "claude-haiku-4-20250414"
    max_tokens: int = 1024
    temperature: float = 0.7
    system_prompt_token_budget: int = 4000
    base_system_prompt: Optional[str] = None
    few_shot_count: int = 3
    few_shot_min_reward: float = 0.3
    consolidation_cluster_size: int = 10
    consolidation_max_clusters: int = 5
    max_reviews_per_cycle: int = 10


@dataclass
class BackendConfig:
    """Which backend to use: 'claude' or 'local'."""

    backend_type: str = "claude"
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)


@dataclass
class StorageConfig:
    """Paths for persistent data stores.

    All database paths are derived from ``data_dir`` when not explicitly set.
    """

    data_dir: str = "./openmind_data"
    experience_db: Optional[str] = None
    ambiguity_db: Optional[str] = None
    knowledge_db: Optional[str] = None
    temporal_db: Optional[str] = None

    def __post_init__(self) -> None:
        """Derive default database paths from *data_dir* if not supplied."""
        base = Path(self.data_dir)
        if self.experience_db is None:
            self.experience_db = str(base / "experience.jsonl")
        if self.ambiguity_db is None:
            self.ambiguity_db = str(base / "ambiguity.jsonl")
        if self.knowledge_db is None:
            self.knowledge_db = str(base / "knowledge.jsonl")
        if self.temporal_db is None:
            self.temporal_db = str(base / "temporal.jsonl")


# ======================================================================
# Top-level config
# ======================================================================

@dataclass
class OpenMindConfig:
    """Complete configuration for the openMind system.

    Each section groups related parameters so that callers can override
    only the subset they care about::

        cfg = default_config()
        cfg.training.learning_rate = 1e-4
        cfg.detection.novelty_threshold = 0.7
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    inquiry: InquiryConfig = field(default_factory=InquiryConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the full config as a nested plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OpenMindConfig:
        """Reconstruct an ``OpenMindConfig`` from a nested dictionary.

        Unknown keys at any level are silently ignored so that config files
        written by newer versions still load in older code.
        """
        def _safe_init(klass, section_data):
            if section_data is None:
                return klass()
            known = {f.name for f in klass.__dataclass_fields__.values()}
            filtered = {k: v for k, v in section_data.items() if k in known}
            return klass(**filtered)

        backend_data = data.get("backend")
        if backend_data is not None:
            backend_obj = BackendConfig(
                backend_type=backend_data.get("backend_type", "claude"),
                claude=_safe_init(ClaudeConfig, backend_data.get("claude")),
            )
        else:
            backend_obj = BackendConfig()

        return cls(
            model=_safe_init(ModelConfig, data.get("model")),
            training=_safe_init(TrainingConfig, data.get("training")),
            detection=_safe_init(DetectionConfig, data.get("detection")),
            inquiry=_safe_init(InquiryConfig, data.get("inquiry")),
            promotion=_safe_init(PromotionConfig, data.get("promotion")),
            review=_safe_init(ReviewConfig, data.get("review")),
            temporal=_safe_init(TemporalConfig, data.get("temporal")),
            storage=_safe_init(StorageConfig, data.get("storage")),
            backend=backend_obj,
            curiosity=_safe_init(CuriosityConfig, data.get("curiosity")),
        )

    # ------------------------------------------------------------------
    # YAML persistence
    # ------------------------------------------------------------------

    def save_config(self, path: str | os.PathLike) -> None:
        """Write the configuration to *path* as YAML.

        Falls back to JSON if PyYAML is not installed.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()

        try:
            import yaml  # type: ignore[import-untyped]

            with open(path, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            logger.info("Config saved to %s (YAML)", path)
        except ImportError:
            import json

            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            logger.info("Config saved to %s (JSON fallback, pyyaml not installed)", path)


# ======================================================================
# Module-level helpers
# ======================================================================

def load_config(path: str | os.PathLike) -> OpenMindConfig:
    """Load an ``OpenMindConfig`` from a YAML (or JSON) file.

    Parameters:
        path: Filesystem path to the configuration file.

    Returns:
        A fully-populated ``OpenMindConfig`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-untyped]
        data = yaml.safe_load(raw)
    except ImportError:
        import json
        data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at top level, got {type(data).__name__}")

    config = OpenMindConfig.from_dict(data)
    logger.info("Loaded config from %s", path)
    return config


def default_config() -> OpenMindConfig:
    """Return an ``OpenMindConfig`` with all default values."""
    return OpenMindConfig()
