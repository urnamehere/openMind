"""Utility modules for the openMind continuous learning system.

Provides configuration management, embedding infrastructure, and domain
classification used across all subsystems.
"""

from openmind.utils.config import OpenMindConfig, load_config, default_config
from openmind.utils.embeddings import EmbeddingProvider
from openmind.utils.domain_tagger import DomainTagger

__all__ = [
    "OpenMindConfig",
    "load_config",
    "default_config",
    "EmbeddingProvider",
    "DomainTagger",
]
