"""Pluggable embedding infrastructure for the openMind continuous learning system.

Provides a unified interface for text embedding and similarity computation.
Uses sentence-transformers when available, falling back to a simple TF-IDF
approach so the system can run without GPU or heavy dependencies.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Pluggable text embedding provider with lazy model loading.

    Attempts to use ``sentence-transformers`` for high-quality dense
    embeddings.  When that package is unavailable, falls back to a
    simple TF-IDF bag-of-words representation that requires no external
    dependencies beyond NumPy.

    Parameters:
        model_name: Name or path of a sentence-transformers model.  Only
            used when the package is installed.  Defaults to
            ``"all-MiniLM-L6-v2"`` -- a compact, fast, and widely-used
            model suitable for general semantic similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None
        self._backend: Optional[str] = None  # "sentence_transformers" | "tfidf"

        # TF-IDF fallback state
        self._tfidf_vocab: Dict[str, int] = {}
        self._tfidf_idf: Optional[np.ndarray] = None
        self._tfidf_doc_count: int = 0
        self._tfidf_doc_freq: Counter = Counter()
        self._tfidf_dim: int = 5000  # vocabulary cap

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the embedding model on first use."""
        if self._backend is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

            self._model = SentenceTransformer(self._model_name)
            self._backend = "sentence_transformers"
            logger.info(
                "EmbeddingProvider: using sentence-transformers model '%s'",
                self._model_name,
            )
        except Exception as exc:
            logger.info(
                "EmbeddingProvider: sentence-transformers not available (%s). "
                "Falling back to TF-IDF embeddings.",
                exc,
            )
            self._backend = "tfidf"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        """Compute an embedding vector for *text*.

        Returns:
            A 1-D NumPy float32 array.
        """
        self._ensure_loaded()

        if self._backend == "sentence_transformers":
            vec = self._model.encode(text, convert_to_numpy=True)
            return np.asarray(vec, dtype=np.float32).flatten()

        return self._tfidf_embed(text)

    def batch_embed(self, texts: Sequence[str]) -> np.ndarray:
        """Compute embedding vectors for a batch of texts.

        Returns:
            A 2-D NumPy float32 array of shape ``(len(texts), dim)``.
        """
        self._ensure_loaded()

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self._backend == "sentence_transformers":
            vecs = self._model.encode(list(texts), convert_to_numpy=True)
            return np.asarray(vecs, dtype=np.float32)

        return np.stack([self._tfidf_embed(t) for t in texts], axis=0)

    def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts.

        Returns:
            A float in [-1.0, 1.0].
        """
        vec_a = self.embed(text_a)
        vec_b = self.embed(text_b)
        return float(self._cosine_similarity(vec_a, vec_b))

    def _compute_centroid(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute the centroid (mean) of a set of embedding vectors.

        Parameters:
            embeddings: 2-D array of shape ``(n, dim)``.

        Returns:
            A 1-D array of shape ``(dim,)``.
        """
        if embeddings.ndim == 1:
            return embeddings.copy()
        if len(embeddings) == 0:
            return np.zeros(0, dtype=np.float32)
        return np.mean(embeddings, axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors, safe against zero-norm."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # TF-IDF fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text)
        # Remove very short tokens and common stop words
        stop_words = {
            "a", "an", "the", "is", "it", "in", "on", "at", "to", "of",
            "and", "or", "for", "with", "as", "by", "this", "that", "be",
            "are", "was", "were", "been", "has", "have", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can",
            "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
            "my", "your", "his", "its", "our", "their",
        }
        return [t for t in tokens if len(t) > 1 and t not in stop_words]

    def _tfidf_embed(self, text: str) -> np.ndarray:
        """Produce a TF-IDF-style embedding for a single text.

        Uses a hashing trick to maintain a fixed vocabulary size without
        needing a pre-built dictionary.  IDF weights are approximated
        from an internal running document count.
        """
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self._tfidf_dim, dtype=np.float32)

        # Update document frequency stats
        self._tfidf_doc_count += 1
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            self._tfidf_doc_freq[tok] += 1

        # Build TF vector using hash-based indexing
        tf = Counter(tokens)
        vec = np.zeros(self._tfidf_dim, dtype=np.float32)

        for token, count in tf.items():
            # Hash to a fixed index
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self._tfidf_dim
            # TF: log-normalised
            tf_val = 1.0 + math.log(count)
            # IDF: smoothed inverse document frequency
            df = self._tfidf_doc_freq.get(token, 0)
            idf = math.log(1.0 + self._tfidf_doc_count / (1.0 + df))
            vec[idx] += tf_val * idf

        # L2 normalise
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm

        return vec
