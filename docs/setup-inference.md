# Setting Up Inference

This guide covers running Grimoire for inference — via a named agent, the Python API, the interactive terminal CLI, or the Chat UI.

A trained checkpoint is required. If you do not have one yet, see [setup-training.md](setup-training.md).

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10 | |
| RAM | 4 GB | More if using a large corpus |
| GPU | — (CPU works) | CUDA auto-detected; 10–30× faster generation |

---

## 1 — Install

### CPU only

```bash
git clone https://github.com/BasLinders/grimoire_ai.git
cd grimoire
pip install -e "."
```

### CUDA (Windows / Linux, RTX card)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e "."
```

### Chat UI (optional)

```bash
pip install -e ".[ui]"
```

---

## 2 — Load a named agent

The simplest way to run inference. `agents.json` at the project root maps agent names to their checkpoint, vocabulary, corpus directories, and generation defaults. `AgentRegistry.build_engine()` loads everything in one call.

```python
from grimoire_ai.agents.registry import AgentRegistry

registry = AgentRegistry("agents.json")   # loads agents.json
engine   = registry.build_engine("saga")  # checkpoint + corpus auto-loaded
```

`agents.json` entry format:

```json
{
    "saga": {
        "display_name": "Saga",
        "description":  "D&D rules, encounter mathematics, and data-science assistant.",
        "checkpoint":   "checkpoints/saga/latest.pt",
        "vocab":        "data/tokenizer/bpe.json",
        "corpus_dirs":  ["data/corpus/saga/"],
        "gen_config": {
            "max_new_tokens": 256,
            "temperature":    0.8,
            "top_k":          50,
            "top_p":          0.9
        }
    }
}
```

All paths in `agents.json` are resolved relative to the file's location. `corpus_dirs` is optional — omit it for a model-only agent.

---

## 3 — Single-turn inference (Python API)

```python
from grimoire_ai.llm.inference.engine import InferenceEngine

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)

response = engine.respond("What is the speed of a grappled creature?")
print(response)
```

The engine auto-detects CUDA and falls back to CPU.

---

## 4 — Multi-turn chat (Python API)

```python
from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.state.conversation import ConversationState

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
state = ConversationState()

r1 = engine.chat("What happens when a creature is grappled?", state)
r2 = engine.chat("How do I escape the grapple?", state)  # model sees prior turn
print(r1)
print(r2)
```

`ConversationState` packs the full turn history into each prompt, dropping the oldest turns first when the token budget runs out.

---

## 5 — Corpus-grounded responses

Attaching a `GrimoireCorpus` grounds answers in a defined knowledge base. Retrieved passages are injected into the prompt before generation — no retraining required.

```python
from grimoire_ai.corpus.corpus import GrimoireCorpus
from grimoire_ai.llm.inference.engine import InferenceEngine

corpus = GrimoireCorpus()
corpus.add_text(
    "A grappled creature has its speed reduced to zero. "
    "The condition ends if the grappler is incapacitated.",
    source="dnd_srd",
)

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
    corpus=corpus,
)

response = engine.respond("What happens when a creature is grappled?", top_k_corpus=3)
print(response)
```

### Loading a corpus from a directory

```python
from pathlib import Path
from grimoire_ai.corpus.corpus import GrimoireCorpus

corpus = GrimoireCorpus()
for path in Path("data/corpus/saga/").glob("*.txt"):
    corpus.add_text(path.read_text(encoding="utf-8"), source=path.stem)
```

### How retrieval works

The corpus engine indexes stemmed 4-gram multi-tokens from every added text. At query time it scores candidates by Jaccard similarity and returns the top-k results. Each result is an unstemmed 200-character excerpt from the original text — the LLM always sees natural prose, not stemmed tokens.

---

## 6 — Tuning generation behaviour

```python
from grimoire_ai.llm.inference.sampler import GenerationConfig

config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.8,        # higher = more varied, lower = more focused
    top_k=50,               # keep only the 50 most likely next tokens
    top_p=0.9,              # nucleus: cut off the tail of the distribution
    repetition_penalty=1.1, # penalise already-generated tokens (1.0 = off)
)

response = engine.respond("Explain advantage and disadvantage.", gen_config=config)
```

**Parameter guidance:**

| Parameter | Conservative | Balanced | Creative |
|---|---|---|---|
| `temperature` | 0.5 | 0.8 | 1.2 |
| `top_k` | 20 | 50 | 100 |
| `top_p` | 0.8 | 0.9 | 0.95 |
| `repetition_penalty` | 1.2 | 1.1 | 1.0 |

You can also pass `gen_config` per-call to override the engine's default:

```python
engine.chat("Roll for initiative", state, gen_config=GenerationConfig(max_new_tokens=64))
```

---

## 7 — Chat UI

```bash
python -m grimoire_ai.ui
```

Open `http://localhost:7860` and go to the **Chat** tab:

1. **Select an agent** from the dropdown (populated from `agents.json`) and click **Load agent** — the engine and corpus load automatically.
2. Or expand **Load checkpoint manually** to load any `.pt` file directly.
3. Adjust the sampling sliders (temperature, top-k, top-p, max tokens).
4. Type a query and click **Send**. Click **Clear conversation** to reset history.

---

## 8 — Terminal chat CLI

```bash
python -m grimoire_ai.cli.chat \
    --checkpoint checkpoints/finetune/step_0000500.pt \
    --vocab      data/tokenizer/bpe.json \
    --corpus-dir data/corpus/saga/
```

All flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to `.pt` checkpoint |
| `--vocab` | required | Path to `bpe.json` tokenizer |
| `--corpus-dir` | — | Directory of `.txt` files to load as corpus |
| `--top-k-corpus` | `5` | Number of corpus passages to retrieve per query |
| `--max-turns` | `20` | Maximum conversation turns before oldest are dropped |
| `--temperature` | `0.8` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--max-new-tokens` | `128` | Maximum tokens to generate per response |

Commands during a session:

| Command | Effect |
|---|---|
| `/quit` | Exit |
| `/clear` | Reset conversation history |
| `/history` | Print all turns so far |

---

## 9 — Checkpoint compatibility

Any checkpoint produced by `train.py` or `finetune.py` is directly loadable. The checkpoint stores the full model configuration, so you never need to specify architecture parameters separately.

```python
# Works with both pre-trained and fine-tuned checkpoints.
engine = InferenceEngine(
    checkpoint_path="checkpoints/pretrain/step_0010000.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
```

Responses from a **pre-trained** (not yet fine-tuned) checkpoint will be text continuations rather than answers. Fine-tune first for chat-style responses.

---

## 10 — Troubleshooting

**`FileNotFoundError: checkpoint not found`**
Check the path. Checkpoints are only written at `save_every` step intervals — make sure training ran long enough.

**Responses are repetitive or cut off early**
Raise `max_new_tokens`, lower `repetition_penalty` (closer to 1.0), or increase `temperature`. If the model was only pre-trained and not fine-tuned, responses will drift — fine-tune first.

**Model continues the question instead of answering**
The checkpoint was not fine-tuned. Run `finetune_saga.py` (or your own fine-tuning script) to teach the model to respond after `<AST>`.

**CUDA out of memory**
The default ~25 M parameter model (~100 MB fp32) fits comfortably in 4 GB VRAM. If you hit OOM, force CPU: `InferenceEngine(..., device="cpu")`.

**Slow generation on CPU**
Expected — each token requires a full forward pass. The KV-cache reduces this from O(n²) to O(1) per step, but CPU throughput is limited. A CUDA GPU gives roughly 10–30× speedup.
