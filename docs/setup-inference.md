# Setting up Inference

This guide covers running the Grimoire model for inference — generating
responses via the Python API, the interactive Chat UI, or with a corpus
attached for retrieval-augmented generation.

A trained checkpoint is required.  If you do not have one yet, see
[setup-training.md](setup-training.md).

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| Python | 3.10 | |
| RAM | 4 GB | More if using a large corpus |
| GPU | — (CPU works) | CUDA auto-detected; speeds up generation |

---

## 1 — Install

### CPU only

```bash
git clone https://github.com/BasLinders/grimoire.git
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

## 2 — Minimal usage (Python API)

```python
from grimoire.llm.inference.engine import InferenceEngine

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)

response = engine.respond("What is the speed of a grappled creature?")
print(response)
```

The engine auto-detects CUDA and falls back to CPU when no GPU is available.

---

## 3 — Tuning generation behaviour

Pass a `GenerationConfig` to control how the model samples:

```python
from grimoire.llm.inference.engine import InferenceEngine
from grimoire.llm.inference.sampler import GenerationConfig

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)

config = GenerationConfig(
    max_new_tokens=256,   # maximum tokens to generate
    temperature=0.8,      # higher = more creative, lower = more deterministic
    top_k=50,             # keep only the 50 most likely next tokens
    top_p=0.9,            # nucleus sampling — cut off the tail of the distribution
    repetition_penalty=1.1,  # penalise repeating tokens (1.0 = disabled)
)

response = engine.respond("Explain advantage and disadvantage.", gen_config=config)
print(response)
```

**Parameter guidance:**

| Parameter | Conservative | Balanced | Creative |
|---|---|---|---|
| `temperature` | 0.5 | 0.8 | 1.2 |
| `top_k` | 20 | 50 | 100 |
| `top_p` | 0.8 | 0.9 | 0.95 |
| `repetition_penalty` | 1.2 | 1.1 | 1.0 |

---

## 4 — Corpus-grounded responses

Attaching a `GrimoireCorpus` grounds the model's answers in a defined
knowledge base.  Retrieved passages are injected into the prompt as context
before the model generates a response — no retraining required.

```python
from grimoire.corpus.corpus import GrimoireCorpus
from grimoire.llm.inference.engine import InferenceEngine

# Build a corpus from text.
corpus = GrimoireCorpus()
corpus.add_text(
    "A grappled creature has its speed reduced to zero. "
    "The condition ends if the grappler is incapacitated.",
    source="dnd_srd",
)
corpus.add_text(
    "Advantage means you roll two d20s and take the higher result. "
    "Disadvantage means you roll two d20s and take the lower result.",
    source="dnd_srd",
)

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
    corpus=corpus,
    max_context_tokens=512,   # token budget for retrieved passages
)

# The engine retrieves the most relevant passages and injects them.
response = engine.respond("What happens when a creature is grappled?", top_k_corpus=3)
print(response)
```

### Loading a corpus from files

For larger knowledge bases, build the corpus from text files at startup:

```python
from pathlib import Path
from grimoire.corpus.corpus import GrimoireCorpus

corpus = GrimoireCorpus()
for path in Path("data/corpus/").glob("*.txt"):
    corpus.add_text(path.read_text(encoding="utf-8"), source=path.stem)
```

### How retrieval works

The corpus engine stems and indexes 4-gram multi-tokens from every
added text.  At query time it scores candidate passages by Jaccard
similarity and returns the top-k ranked results.  Each result includes
an unstemmed 200-character excerpt from the original text, which is what
gets injected into the prompt.  This means the LLM always sees natural
prose, not stemmed tokens.

---

## 5 — Chat UI

The interactive chat tab lets you load a checkpoint and query the model
without writing any code.

```bash
python -m grimoire.ui
```

Open `http://localhost:7860` and go to the **Chat** tab:

1. Enter the path to your checkpoint (`.pt` file)
2. Enter the path to your vocabulary (`.json` file, default `data/tokenizer/bpe.json`)
3. Click **Load model** — wait for the status message
4. Adjust the sampling sliders (temperature, top-k, top-p, max tokens)
5. Type a query and click **Send**

The UI does not currently support corpus attachment — use the Python API
for retrieval-augmented responses.

---

## 6 — Checkpoint compatibility

Any checkpoint produced by either `train.py` (pre-training) or `finetune.py`
(fine-tuning) is loadable by `InferenceEngine`.  The checkpoint stores the
model configuration alongside the weights, so you do not need to specify
architecture parameters separately.

```python
# Works with both pre-trained and fine-tuned checkpoints.
engine = InferenceEngine(
    checkpoint_path="checkpoints/pretrain/step_0010000.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
```

Note: responses from a **pre-trained** (not yet fine-tuned) checkpoint will
be text continuations rather than conversational answers.  Fine-tune first
for chat-style responses.

---

## 7 — Troubleshooting

**`FileNotFoundError: checkpoint not found`**
The checkpoint path is wrong or the file does not exist.  Check the path
and make sure training completed (checkpoints are only written at
`save_every` intervals).

**Responses are repetitive or cut off early**
Try increasing `max_new_tokens`, lowering `repetition_penalty` (closer to
1.0), or raising `temperature`.  If the model was only pre-trained and not
fine-tuned, responses will tend to drift — fine-tune first.

**CUDA out of memory**
The default model (~25 M params, ~100 MB fp32) fits easily in 4 GB VRAM
for inference.  If you hit OOM, force CPU: `InferenceEngine(..., device="cpu")`.

**Slow generation on CPU**
Expected — each token requires a full forward pass.  The KV-cache reduces
this from O(n²) to O(1) per step, but CPU throughput is still limited.
A CUDA-capable GPU gives roughly 10–30× speedup.
