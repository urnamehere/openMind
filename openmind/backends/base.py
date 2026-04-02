"""Abstract base class for all LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class GenerationContext:
    """Context passed to the backend during generation."""

    system_prompt: Optional[str] = None
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    domain_tags: Optional[List[str]] = None


class Backend(ABC):
    """Abstract base for all LLM backends.

    Every backend must implement generation, self-evaluation, knowledge
    articulation, consolidation, and reflection. The consolidation and
    reflection methods are backend-specific: local backends train weights,
    while API backends distill knowledge into structured memory.
    """

    @abstractmethod
    def generate(
        self,
        user_input: str,
        context: Optional[GenerationContext] = None,
    ) -> str:
        """Generate a response given user input and optional context."""
        ...

    @abstractmethod
    def evaluate_output(
        self, input_context: str, output: str
    ) -> Optional[Dict[str, float]]:
        """Self-evaluate an output.

        Returns dict with reward dimension keys (e.g. 'task_success',
        'coherence') mapped to scores in [-1, 1], or None if unavailable.
        """
        ...

    @abstractmethod
    def articulate_learning(
        self,
        source_experiences: Sequence[Any],
        promotion_scores: Dict[str, float],
    ) -> str:
        """Generate a natural-language belief summary from experiences."""
        ...

    @abstractmethod
    def consolidate(
        self,
        experience_buffer: Any,
        knowledge_registry: Any,
        cycle: int,
    ) -> Dict[str, Any]:
        """Run backend-specific learning/consolidation. Returns stats dict."""
        ...

    @abstractmethod
    def reflect(
        self,
        knowledge_registry: Any,
        experience_buffer: Any,
        belief_reviewer: Any,
    ) -> Dict[str, Any]:
        """Run backend-specific reflection. Returns stats dict."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the backend is loaded and ready to generate."""
        ...

    @property
    def supports_weight_training(self) -> bool:
        """Whether this backend supports LoRA/weight-level operations."""
        return False

    def __call__(self, prompt: str) -> str:
        """Allow the backend to be used as a callable for self-eval prompts."""
        return self.generate(prompt)
