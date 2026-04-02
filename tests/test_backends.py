"""Tests for the backend abstraction layer and Claude-specific components."""

import pytest

from openmind.backends.base import Backend, GenerationContext
from openmind.backends.claude import (
    ClaudeBackend,
    ExperienceRetriever,
    SystemPromptBuilder,
)
from openmind.memory.knowledge import KnowledgeRegistry, PromotedKnowledge
from openmind.utils.config import BackendConfig, ClaudeConfig, OpenMindConfig


# ---------------------------------------------------------------------------
# GenerationContext
# ---------------------------------------------------------------------------


class TestGenerationContext:
    def test_defaults(self):
        ctx = GenerationContext()
        assert ctx.system_prompt is None
        assert ctx.max_tokens == 1024
        assert ctx.temperature == 0.7
        assert ctx.domain_tags is None

    def test_custom(self):
        ctx = GenerationContext(
            system_prompt="You are helpful.",
            max_tokens=512,
            domain_tags=["coding"],
        )
        assert ctx.system_prompt == "You are helpful."
        assert ctx.domain_tags == ["coding"]


# ---------------------------------------------------------------------------
# SystemPromptBuilder
# ---------------------------------------------------------------------------


class TestSystemPromptBuilder:
    def setup_method(self, tmp_path_factory=None):
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "knowledge.jsonl")
        self.registry = KnowledgeRegistry(storage_path=db_path)

    def test_empty_registry_returns_base_prompt(self):
        builder = SystemPromptBuilder(knowledge_registry=self.registry)
        prompt = builder.build()
        assert "continuous learning" in prompt.lower()

    def test_custom_base_prompt(self):
        builder = SystemPromptBuilder(
            knowledge_registry=self.registry,
            base_prompt="Custom base.",
        )
        prompt = builder.build()
        assert prompt.startswith("Custom base.")

    def test_includes_knowledge_entries(self):
        from datetime import datetime, timezone

        pk = PromotedKnowledge(
            knowledge_id="k1",
            promoted_at=datetime.now(timezone.utc).isoformat(),
            training_cycle=1,
            source_domain=["coding"],
            source_experience_ids=[],
            layer_names=[],
            promotion_scores={},
            belief_summary="Always add type hints to Python functions.",
            initial_confidence=0.9,
            current_confidence=0.9,
        )
        self.registry._entries["k1"] = pk
        self.registry._loaded = True

        builder = SystemPromptBuilder(knowledge_registry=self.registry)
        prompt = builder.build()
        assert "type hints" in prompt

    def test_respects_token_budget(self):
        from datetime import datetime, timezone

        # Add many entries
        for i in range(50):
            pk = PromotedKnowledge(
                knowledge_id=f"k{i}",
                promoted_at=datetime.now(timezone.utc).isoformat(),
                training_cycle=1,
                source_domain=["general"],
                source_experience_ids=[],
                layer_names=[],
                promotion_scores={},
                belief_summary=f"Knowledge entry number {i} with substantial text content " * 10,
                initial_confidence=0.9,
                current_confidence=0.9,
            )
            self.registry._entries[f"k{i}"] = pk
        self.registry._loaded = True

        builder = SystemPromptBuilder(knowledge_registry=self.registry)
        prompt = builder.build(token_budget=500)
        # Should be roughly bounded by token budget * 4 chars/token
        assert len(prompt) < 500 * 8  # generous but bounded

    def test_token_estimation(self):
        assert SystemPromptBuilder._estimate_tokens("hello world") >= 1


# ---------------------------------------------------------------------------
# ExperienceRetriever
# ---------------------------------------------------------------------------


class TestExperienceRetriever:
    def test_empty_buffer_returns_empty(self):
        retriever = ExperienceRetriever(experience_buffer=None)
        assert retriever.retrieve("hello") == []

    def test_keyword_retrieval(self):

        class FakeBuffer:
            def get_recent(self, n):
                return [
                    {
                        "experience_id": "e1",
                        "input_context": "how to sort a list in python",
                        "output": "use sorted()",
                        "reward_signal": 0.8,
                    },
                    {
                        "experience_id": "e2",
                        "input_context": "what is the weather today",
                        "output": "sunny",
                        "reward_signal": 0.7,
                    },
                    {
                        "experience_id": "e3",
                        "input_context": "python list comprehension",
                        "output": "[x for x in items]",
                        "reward_signal": 0.9,
                    },
                ]

        retriever = ExperienceRetriever(
            experience_buffer=FakeBuffer(), embedding_provider=None
        )
        results = retriever.retrieve("sort a python list", top_k=2)
        assert len(results) <= 2
        # The python/list/sort query should match e1 and e3 better than e2
        result_ids = [r["experience_id"] for r in results]
        assert "e2" not in result_ids or "e1" in result_ids

    def test_filters_by_min_reward(self):

        class FakeBuffer:
            def get_recent(self, n):
                return [
                    {
                        "experience_id": "e1",
                        "input_context": "test",
                        "output": "ok",
                        "reward_signal": 0.1,  # below threshold
                    },
                ]

        retriever = ExperienceRetriever(
            experience_buffer=FakeBuffer(), embedding_provider=None
        )
        results = retriever.retrieve("test", min_reward=0.3)
        assert len(results) == 0

    def test_build_few_shot_messages(self):
        retriever = ExperienceRetriever(experience_buffer=None)
        experiences = [
            {"input_context": "hello", "output": "hi there"},
            {"input_context": "bye", "output": "goodbye"},
        ]
        messages = retriever.build_few_shot_messages(experiences)
        assert len(messages) == 4  # 2 user + 2 assistant
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"


# ---------------------------------------------------------------------------
# ClaudeConfig and BackendConfig
# ---------------------------------------------------------------------------


class TestClaudeConfig:
    def test_defaults(self):
        cfg = ClaudeConfig()
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.few_shot_count == 3
        assert cfg.system_prompt_token_budget == 4000

    def test_backend_config_default_is_claude(self):
        cfg = BackendConfig()
        assert cfg.backend_type == "claude"

    def test_openmind_config_has_backend(self):
        cfg = OpenMindConfig()
        assert hasattr(cfg, "backend")
        assert cfg.backend.backend_type == "claude"

    def test_round_trip_serialization(self):
        cfg = OpenMindConfig()
        cfg.backend.backend_type = "local"
        cfg.backend.claude.model = "claude-opus-4-20250514"

        data = cfg.to_dict()
        restored = OpenMindConfig.from_dict(data)
        assert restored.backend.backend_type == "local"
        assert restored.backend.claude.model == "claude-opus-4-20250514"


# ---------------------------------------------------------------------------
# ClaudeBackend (without API key - graceful degradation)
# ---------------------------------------------------------------------------


class TestClaudeBackendNoKey:
    """Tests that ClaudeBackend degrades gracefully without an API key."""

    def setup_method(self):
        import tempfile, os
        self._tmpdir = tempfile.mkdtemp()
        db_path = os.path.join(self._tmpdir, "knowledge.jsonl")
        self.registry = KnowledgeRegistry(storage_path=db_path)

    def test_not_ready_without_key(self):
        import os
        # Ensure no key is set
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            backend = ClaudeBackend(
                config=ClaudeConfig(),
                knowledge_registry=self.registry,
                experience_buffer=None,
            )
            assert not backend.is_ready
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_generate_returns_message_without_key(self):
        import os
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            backend = ClaudeBackend(
                config=ClaudeConfig(),
                knowledge_registry=self.registry,
                experience_buffer=None,
            )
            result = backend.generate("hello")
            assert "not configured" in result.lower()
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key

    def test_evaluate_output_returns_none_without_key(self):
        import os
        old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            backend = ClaudeBackend(
                config=ClaudeConfig(),
                knowledge_registry=self.registry,
                experience_buffer=None,
            )
            assert backend.evaluate_output("test", "output") is None
        finally:
            if old_key:
                os.environ["ANTHROPIC_API_KEY"] = old_key
