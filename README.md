# openMind

**A continuous learning wrapper for LLMs that enables experiential memory, weight promotion, and belief revision -- inspired by biological memory consolidation.**

## The Biological Parallel

Human memory does not write everything to long-term storage the moment it happens. Instead, the brain operates in cycles: experiences are captured during waking hours, consolidated during sleep, and periodically reviewed to strengthen or revise existing beliefs. openMind brings this same architecture to large language models.

- **Short-term memory** (hippocampus) maps to an experience buffer that captures interaction outcomes.
- **Sleep consolidation** maps to offline training cycles that promote high-signal experiences into LoRA weight updates.
- **Belief revision** maps to a reflection phase that detects contradictions, confidence drops, and knowledge gaps -- then corrects them.

## Architecture: Three Cycles

### 1. Awake Cycle (online, every interaction)

The model interacts with users. Each interaction produces an `Experience` with a multi-dimensional `RewardSignal`. Eight detectors watch for learning opportunities: contradictions, confidence drops, novelty, user corrections, domain shifts, repetition, hallucination, and reasoning failures.

### 2. Sleep Cycle (offline, scheduled)

A `TrainingCycleManager` gathers high-reward recent experiences plus a replay sample from the buffer, then fine-tunes a LoRA adapter. Elastic Weight Consolidation (EWC) regularisation prevents catastrophic forgetting of previously learned knowledge.

### 3. Reflect Cycle (periodic review)

Spaced-repetition scheduling surfaces old memories for review. The system checks whether stored beliefs still hold, promotes strongly confirmed patterns, and revises or discards contradicted ones.

## Quick Start

```python
from openmind.core.experience import Experience, ExperienceBuffer
from openmind.reward.signal import RewardSignal

# Record an experience
buffer = ExperienceBuffer("openmind_data/experiences.jsonl")

exp = Experience(
    timestamp="2026-04-02T12:00:00Z",
    input_context="What is the capital of France?",
    output="The capital of France is Paris.",
    reward_signal=0.9,
    domain_tags=["geography"],
    confidence=0.95,
)
buffer.record(exp)

# Evaluate a reward signal
signal = RewardSignal(
    task_success=0.9,
    coherence=0.85,
    user_satisfaction=0.8,
    groundedness=0.95,
)
print(f"Aggregate reward: {signal.aggregate:.3f}")
print(f"Confidence: {signal.confidence:.3f}")
```

## Installation

```bash
# Core (CPU only, no model training)
pip install .

# With GPU training support
pip install ".[gpu]"

# With embedding-based memory retrieval
pip install ".[embeddings]"

# Everything
pip install ".[all]"
```

For development:

```bash
pip install -e ".[all]"
pip install pytest ruff
```

## Project Structure

```
openmind/
    core/
        experience.py   - Experience recording and replay buffer
        training.py     - Training cycle orchestration (sleep phase)
    detection/          - Learning opportunity detectors (8 types)
    memory/             - Long-term memory index and promotion logic
    reward/
        signal.py       - Multi-dimensional reward signal representation
    utils/              - Configuration loading, logging, helpers
configs/
    default_config.yaml - All tunable parameters with defaults
tests/                  - Unit and integration tests
```

## Configuration

Copy `configs/default_config.yaml` to `configs/user_config.yaml` and edit as needed. Key sections:

| Section | Controls |
|---|---|
| `model` | LoRA rank, alpha, target modules |
| `training` | Batch size, learning rate, EWC strength, replay ratio |
| `detection` | Sensitivity thresholds for each of the 8 detectors |
| `inquiry` | How many clarifying questions the system may ask |
| `promotion` | When soft memory graduates to weight updates |
| `review` | Spaced-repetition intervals |
| `temporal` | Cooling-off periods, experience TTL |
| `storage` | File paths for buffers, checkpoints, indices |

See the comments in `default_config.yaml` for details on each parameter.

## License

MIT -- see [LICENSE](LICENSE).
