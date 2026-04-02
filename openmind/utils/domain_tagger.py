"""Domain tagger - classifies interactions into domain tags."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "technical": [
        "code", "debug", "error", "function", "api", "server", "database",
        "deploy", "git", "docker", "kubernetes", "linux", "python", "javascript",
        "typescript", "rust", "compile", "runtime", "stack trace", "refactor",
        "algorithm", "data structure", "cpu", "memory", "network", "http",
        "sql", "nosql", "cloud", "aws", "azure", "terraform",
    ],
    "medical": [
        "diagnosis", "symptom", "treatment", "medication", "patient",
        "clinical", "hospital", "doctor", "disease", "prescription", "therapy",
    ],
    "legal": [
        "contract", "lawsuit", "attorney", "court", "regulation",
        "compliance", "liability", "statute", "jurisdiction", "legal",
    ],
    "financial": [
        "investment", "portfolio", "stock", "bond", "interest rate",
        "mortgage", "tax", "budget", "revenue", "profit", "accounting",
    ],
    "creative": [
        "story", "poem", "creative writing", "fiction", "character",
        "narrative", "plot", "design", "art", "music", "brainstorm",
    ],
    "casual": [
        "hello", "hi", "hey", "thanks", "how are you", "what's up",
        "good morning", "good night", "chat",
    ],
    "educational": [
        "explain", "teach", "learn", "understand", "concept",
        "tutorial", "course", "study", "homework", "research",
    ],
    "safety": [
        "dangerous", "warning", "risk", "hazard", "emergency",
        "safety", "secure", "protect", "vulnerability", "threat",
    ],
    "data_science": [
        "machine learning", "neural network", "training", "model",
        "dataset", "feature", "classification", "regression",
        "deep learning", "nlp", "transformer", "embedding",
        "fine-tune", "lora", "weights", "tensor", "gpu",
    ],
    "devops": [
        "ci/cd", "pipeline", "monitoring", "logging", "alerting",
        "infrastructure", "scaling", "load balancer", "container",
    ],
}


class DomainTagger:
    """Classifies interactions into domain tags using keyword matching."""

    def __init__(self, model: Any = None) -> None:
        self.model = model
        self._domain_counts: Dict[str, int] = defaultdict(int)
        self._total_tagged: int = 0

    def tag(self, input_context: str, output: Optional[str] = None) -> List[str]:
        """Classify the interaction into domain tags."""
        text = input_context.lower()
        if output:
            text += " " + output.lower()

        tags = self._keyword_classify(text)
        if not tags:
            tags = ["general"]

        self._total_tagged += 1
        for t in tags:
            self._domain_counts[t] += 1

        return tags

    def _keyword_classify(self, text: str) -> List[str]:
        """Classify text using keyword matching."""
        matched: List[str] = []
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in text)
            specific_match = any(kw in text for kw in keywords if len(kw) > 8)
            if match_count >= 2 or specific_match:
                matched.append(domain)
        return matched

    def get_domain_frequency(self, domain: str) -> float:
        if self._total_tagged == 0:
            return 0.0
        return self._domain_counts.get(domain, 0) / self._total_tagged

    def get_all_frequencies(self) -> Dict[str, float]:
        if self._total_tagged == 0:
            return {}
        return {d: c / self._total_tagged for d, c in self._domain_counts.items()}

    def is_novel_domain(self, tags: List[str]) -> bool:
        return any(tag not in self._domain_counts for tag in tags)
