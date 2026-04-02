# openMind

**A continuous learning wrapper for LLMs that enables experiential memory, knowledge accumulation, and belief revision -- inspired by biological memory consolidation.**

openMind gives an LLM a persistent memory. It remembers what worked, learns from mistakes, and gets better over time. Works with the **Claude API** (recommended, easiest setup) or **local HuggingFace models** (advanced, requires GPU).

---

## Getting Started (Step by Step)

This guide assumes you have never done this before. Follow each step in order.

### Step 1: Make sure Python is installed

You need Python 3.9 or newer. Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:

```bash
python --version
```

If you see something like `Python 3.11.4`, you're good. If not, download Python from [python.org](https://www.python.org/downloads/) and install it. During installation, **check the box that says "Add Python to PATH"**.

### Step 2: Download openMind

Open your terminal and run:

```bash
git clone https://github.com/urnamehere/openMind.git
cd openMind
```

If you don't have `git`, you can also download the ZIP from GitHub and unzip it, then open a terminal in that folder.

### Step 3: Install openMind

For the **Claude API backend** (recommended -- no GPU needed):

```bash
pip install ".[claude]"
```

For **everything** (Claude API + local model training + embeddings):

```bash
pip install ".[all]"
```

### Step 4: Get your Claude API key

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up or log in
3. Go to **API Keys** and click **Create Key**
4. Copy the key (it starts with `sk-ant-...`)

Now set it as an environment variable:

**Mac/Linux:**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

**Windows (Command Prompt):**
```bash
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

**Windows (PowerShell):**
```bash
$env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

To make this permanent, add the line to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.).

### Step 5: Run your first conversation

Create a file called `my_first_chat.py`:

```python
from openmind import ContinualWrapper

# Create the wrapper (uses Claude API by default)
mind = ContinualWrapper()

# Have a conversation
response = mind.chat("What are three tips for learning Python?")
print(response)

# Give feedback on the response
response = mind.chat(
    "How about for JavaScript?",
    user_response="That Python advice was really helpful, thanks!"
)
print(response)

# Check what the system has learned so far
status = mind.get_status()
print(f"\nExperiences recorded: {status['experience_buffer_size']}")
print(f"Backend: {status['backend']} (ready: {status['backend_ready']})")
```

Run it:

```bash
python my_first_chat.py
```

### Step 6: Run the learning cycles

After some conversations, trigger consolidation (learning) and reflection (review):

```python
from openmind import ContinualWrapper

mind = ContinualWrapper()

# ... have several conversations with mind.chat() ...

# Sleep cycle: distill knowledge from experiences
consolidation_stats = mind.consolidate()
print(f"Knowledge distilled: {consolidation_stats}")

# Reflect cycle: review what was learned
reflection_stats = mind.reflect()
print(f"Beliefs reviewed: {reflection_stats}")
```

In a real application, you'd run `consolidate()` periodically (e.g., every few hours or daily) and `reflect()` less frequently (e.g., weekly).

---

## How It Works

openMind operates in three phases, inspired by how human memory works:

### 1. Awake -- Every Conversation

Each time you call `chat()`, the system:

- Generates a response using Claude (with a **dynamic system prompt** built from everything it has learned so far)
- Retrieves similar past interactions as **few-shot examples** (RAG)
- Collects a **reward signal** -- did the user like it? Was it correct? Was it novel?
- Runs **8 quality detectors** to decide what to do with this experience:
  - **TRAIN** -- high-quality signal, save it for learning
  - **SHELVE** -- ambiguous signal, save it for later when we know more
  - **ASK** -- unclear, append a clarification question
  - **DISCARD** -- suspected reward gaming or garbage signal

### 2. Sleep -- Consolidation

When you call `consolidate()`, the system:

- Gathers high-quality experiences from the buffer
- Groups them by domain (coding, science, writing, etc.)
- Asks Claude to **extract generalizable knowledge** from each group
- Stores these as structured beliefs in the Knowledge Registry
- These beliefs then get injected into future system prompts

### 3. Reflect -- Belief Review

When you call `reflect()`, the system:

- Uses **spaced repetition** to surface old beliefs for review
- Runs 5 tests on each belief: Is it still consistent? Supported by evidence? Contradicted by newer knowledge? Still relevant?
- **Reinforces** valid beliefs, **revises** weakened ones, **deprecates** stale ones
- Monitors overall reward system health

Over time, the system prompt evolves to reflect accumulated knowledge -- the model effectively "remembers" what it has learned.

---

## Two Backends

| | Claude API (default) | Local Model (advanced) |
|---|---|---|
| **Setup** | Just an API key | GPU + model download |
| **How it learns** | Knowledge distillation into system prompts | LoRA fine-tuning + weight promotion |
| **Install** | `pip install ".[claude]"` | `pip install ".[gpu]"` |
| **Best for** | Most users, prototyping, no GPU needed | Researchers, full weight-level control |

### Using a local model instead

```python
from openmind import ContinualWrapper

# Pass a HuggingFace model path to use local backend
mind = ContinualWrapper(base_model="mistralai/Mistral-7B-v0.3")
response = mind.chat("hello")
```

This requires a GPU and the `torch`, `transformers`, and `peft` packages.

---

## Configuration

You can customize behavior by passing a config file:

```python
mind = ContinualWrapper(config_path="configs/default_config.yaml")
```

Or modify settings directly:

```python
from openmind.utils.config import OpenMindConfig

config = OpenMindConfig()
config.backend.backend_type = "claude"              # or "local"
config.backend.claude.model = "claude-sonnet-4-20250514"  # which Claude model
config.backend.claude.few_shot_count = 5            # more examples from memory
config.backend.claude.system_prompt_token_budget = 6000  # bigger knowledge context
config.inquiry.ask_budget_per_session = 5           # allow more clarification questions

mind = ContinualWrapper(config=config)
```

### Key configuration sections

| Section | What it controls |
|---|---|
| `backend` | Claude API vs local model, API settings |
| `model` | LoRA rank, alpha, target modules (local only) |
| `training` | Batch size, learning rate, EWC strength (local only) |
| `detection` | Sensitivity thresholds for the 8 signal quality detectors |
| `inquiry` | How many clarifying questions the system may ask per session/day |
| `promotion` | When to promote adapter weights into base model (local only) |
| `review` | Spaced-repetition intervals for belief review |
| `storage` | File paths for experience buffer, knowledge registry, etc. |

### Where data is stored

By default, all data lives in `./openmind_data/`:

```
openmind_data/
    experiences.jsonl       -- every recorded interaction
    knowledge_registry.jsonl -- accumulated beliefs/knowledge
    ambiguity_buffer.jsonl  -- shelved uncertain experiences
    temporal_rewards.jsonl  -- reward signal history
```

You can change this with `ContinualWrapper(data_dir="/your/path")`.

---

## Installation Options

```bash
# Claude API only (recommended for most users)
pip install ".[claude]"

# Local GPU training only
pip install ".[gpu]"

# Semantic search over past experiences
pip install ".[embeddings]"

# Everything
pip install ".[all]"

# For development
pip install -e ".[all]"
pip install pytest ruff
```

## Project Structure

```
openmind/
    backends/
        base.py         - Backend interface (ABC)
        claude.py       - Claude API: prompt builder, RAG retriever
        local.py        - Local HuggingFace model with LoRA
    core/
        experience.py   - Experience recording and replay buffer
        training.py     - LoRA training with EWC (local backend)
        promotion.py    - Weight stability tracking and promotion
    detection/
        detectors.py    - 8 signal quality detectors + 4-way gatekeeper
    memory/
        ambiguity.py    - Ambiguity buffer for uncertain experiences
        knowledge.py    - Knowledge registry with spaced repetition
        review.py       - Belief reviewer (the "Pythagoras checker")
    reward/
        signal.py       - Multi-dimensional reward signal (5 dimensions)
        collector.py    - Fuses rewards from 6 sources
        temporal.py     - Tracks reward changes over time
        inquiry.py      - Active clarification question system
        meta.py         - Monitors reward system health
    utils/
        config.py       - All configuration dataclasses
        embeddings.py   - Text embeddings (sentence-transformers or TF-IDF)
        domain_tagger.py - Classifies interactions by domain
    wrapper.py          - ContinualWrapper: the main orchestrator
configs/
    default_config.yaml - All tunable parameters with defaults
tests/                  - Unit tests (73 tests)
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Troubleshooting

**"No API key found"** -- Make sure `ANTHROPIC_API_KEY` is set in your environment. See Step 4 above.

**"anthropic package not installed"** -- Run `pip install anthropic` or `pip install ".[claude]"`.

**"backend not ready"** -- The system will still record interactions for future learning, even without a working backend. Fix the backend issue and your recorded experiences will be there.

**"torch/peft/transformers not available"** -- You're trying to use the local backend without GPU dependencies. Run `pip install ".[gpu]"` or switch to the Claude backend.

## License

MIT -- see [LICENSE](LICENSE).
