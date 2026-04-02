"""
openMind - A continuous learning wrapper for LLMs.

Enables experiential memory, weight promotion, and belief revision,
inspired by biological memory consolidation.
"""

from openmind.wrapper import ContinualWrapper
from openmind.utils.config import OpenMindConfig

__version__ = "0.1.0"
__all__ = ["ContinualWrapper", "OpenMindConfig", "__version__"]
