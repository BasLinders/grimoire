# Setting Up Training

This guide covers everything needed to prepare a corpus, pre-train a Grimoire model, and fine-tune it for conversational use — from environment setup through to a saved, chat-ready checkpoint.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11–3.13 (see CUDA note below) |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works) | NVIDIA RTX with 4 GB+ VRAM |
| CUDA | — | 12.4 (for RTX cards) |

> **For GPU use, keep Python within the range PyTorch ships CUDA wheels for**
> (currently up to 3.13). A newer Python still works CPU-only. See
> [§1 — CUDA](#cuda-windows--linux-rtx-card) for a dedicated-venv workaround
> if your default interpreter is too new.

---

## 1 — Install

### CPU only

```bash
git clone https://github.com/BasLinders/grimoire_ai.git
cd grimoire
pip install -e ".[dev]"
```

### CUDA (Windows / Linux, RTX card)

Install PyTorch with CUDA 12.4 support **before** the package:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

Verify the GPU is visible:

```python
import torch
print(torch.cuda.is_available())    # True
print(torch.cuda.get_device_name(0))
```

> **Python version matters for CUDA.** PyTorch's CUDA wheels lag the newest
> Python releases. If `pip install torch --index-url .../cu124` fails with
> `Could not find a version that satisfies the requirement torch`, your
> Python is too new — at the time of writing the CUDA wheels top out at
> **Python 3.13** (`cp313`), so Python 3.14 has no CUDA build and silently
> installs the CPU-only wheel instead (`torch.cuda.is_available()` →
> `False`). Check with `python --version`. If you're on a Python newer than
> the latest CUDA wheel, use a dedicated CUDA venv (next section) pinned to a
> supported version rather than changing your system Python.

#### A dedicated CUDA virtual environment

The cleanest way to get GPU acceleration without disturbing your main
environment — and the recommended approach if your default `python` is too
new for the CUDA wheels — is a separate venv pinned to a supported Python
version. Training/eval scripts auto-detect CUDA (`InferenceEngine` and the
trainers pick `cuda` when `torch.cuda.is_available()`), so **no script flags
change** — only which interpreter you run them with.

**Create it** (example pins Python 3.13 via the `py` launcher on Windows; on
Linux use `python3.13 -m venv`):

```bash
# From the repo root
py -3.13 -m venv .venv-cuda
source .venv-cuda/Scripts/activate     # Windows (Git Bash)
# source .venv-cuda/bin/activate       # Linux / macOS

pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

python -c "import torch; print(torch.cuda.is_available())"   # expect: True
```

While the venv is active, your prompt shows `(.venv-cuda)` and `python` /
`pip` resolve to it. Run any training or eval command exactly as documented
elsewhere in this guide — it will use the GPU automatically. CUDA embedding /
indexing is typically **10–30× faster** than CPU for this model size.

**Switch back** to your system Python at any time:

```bash
deactivate          # prompt's (.venv-cuda) prefix disappears
```

**Remove it** when you no longer need it — a venv is just a directory, so
deleting it fully uninstalls everything inside without touching your system
Python:

```bash
deactivate                  # if it's still active
rm -rf .venv-cuda           # Windows (Git Bash) / Linux / macOS
# Remove-Item -Recurse -Force .venv-cuda   # Windows PowerShell
```

> **Don't `pip uninstall torch` to "switch" between CPU and CUDA builds in
> one environment.** Uninstalling can leave an empty `torch` namespace
> directory behind that imports without error but fails at `import torch.nn`
> with `ModuleNotFoundError`. Keep the CPU and CUDA builds in separate venvs
> instead; if you hit that broken state, `pip install torch` (CPU) or the
> `cu124` index URL (CUDA) reinstalls cleanly over it.

### Training UI (optional)

```bash
pip install -e ".[ui]"
```

#### Running the UI on GPU

The UI does **not** switch compute environments at runtime — whether it uses
the GPU is fixed by **which interpreter launches it**. To run training and
evaluation on CUDA from the UI, launch the UI itself from your
[CUDA venv](#a-dedicated-cuda-virtual-environment):

```bash
source .venv-cuda/Scripts/activate     # Windows (Git Bash)
pip install -e ".[ui]"                 # the CUDA venv needs the UI extra too
python -m grimoire_ai.ui
```

The `[ui]` install is required separately in `.venv-cuda` — the venv setup
earlier only installed `.[dev]`, which does not include Gradio, so launching
the UI without this step fails with a missing-`gradio` import.

Behaviour once launched:

| Tab | Device control | On CPU-launched UI |
|---|---|---|
| Pre-train / Fine-tune | **none** — auto-detects `cuda` if available, else `cpu` | always CPU; no override |
| Evaluate | **Device** dropdown (`Auto` / `CPU` / `CUDA`) | selecting `CUDA` errors — the process has no GPU torch |

Confirm which device is actually in use from the on-screen log, not the fan
noise: training prints `Training on CUDA | …` / `Training on CPU | …`, and the
Evaluate tab prints `Model loaded on cuda` / `Model loaded on cpu`.

### Corpus scraper (optional — for ingesting web/PDF/DOCX sources)

```bash
pip install -e ".[scraper]"         # web, PDF, DOCX, Markdown
pip install -e ".[scraper-ocr]"     # + image OCR via Tesseract
```

---

## 2 — Prepare your corpus

### Option A — Use the Saga corpus (D&D 5e SRD)

Download and process the CC-BY 4.0 D&D 5e SRD in one step:

```bash
python scripts/build_saga_corpus.py
```

This downloads the SRD (~1.5 MB from GitHub), splits it into 24 rule sections, converts each to plain text, and copies four hand-authored probability and encounter-math reference files into `data/corpus/saga/`.

### Option B — Ingest your own sources

**Via the Ingest tab (UI):**

```bash
python -m grimoire_ai.ui
# → Ingest tab
```

Paste a URL, upload a file, or point at a directory. Choose a cleaning level (minimal / standard / thorough) and click **Start ingestion**.

**Via CLI:**

```bash
# Web page
python -m grimoire_ai.corpus.ingest --source https://example.com/rules --output data/raw/

# Raw Markdown URL (GitHub etc.) — detected automatically
python -m grimoire_ai.corpus.ingest \
    --source https://raw.githubusercontent.com/user/repo/main/doc.md \
    --output data/raw/

# Local file (PDF, DOCX, Markdown, plain text)
python -m grimoire_ai.corpus.ingest --source docs/phb_excerpt.pdf --output data/raw/

# Directory (batch, with cleaning level)
python -m grimoire_ai.corpus.ingest \
    --source docs/ --output data/raw/ --recursive --cleaning thorough
```

**Supported sources:**

| Source | Extra install | Notes |
|---|---|---|
| Web URL (HTML) | `pip install -e ".[scraper]"` | Strips nav/header/footer boilerplate |
| Raw Markdown URL | `pip install -e ".[scraper]"` | Detected by `.md` extension or `text/plain` response |
| PDF | `pip install -e ".[scraper]"` | Page-by-page text extraction |
| Word (.docx) | `pip install -e ".[scraper]"` | Paragraph extraction |
| Markdown file | no extra dep | Syntax stripped |
| Plain text | no extra dep | Returned as-is |
| Image (OCR) | `pip install -e ".[scraper-ocr]"` + Tesseract | Always uses `thorough` cleaning |

### Pre-process corpus for training

The trainer expects a memory-mapped binary of token IDs. The preprocessing script tokenises raw text files and writes that binary.

Expected directory layout:

```
data/
├── raw/          ← your .txt files go here
├── processed/    ← corpus.bin is written here
└── tokenizer/    ← bpe.json is written here (created on first run)
```

Run preprocessing:

```bash
python -m grimoire_ai.llm.data.preprocessing \
    --input  data/raw/ \
    --output data/processed/corpus.bin \
    --vocab  data/tokenizer/bpe.json
```

What it does:

- Trains a byte-level BPE vocabulary (16 384 tokens) if `bpe.json` does not exist
- Tokenises every `.txt` file under `--input`
- Writes a single `int32` memory-mapped binary to `--output`

---

## 3 — Pre-train

Pre-training teaches language patterns from raw text. It does **not** teach the model to answer questions — that is handled by fine-tuning (step 4).

### Via CLI

```bash
python -m grimoire_ai.llm.training.train \
    --corpus data/processed/corpus.bin
```

All flags are optional. Defaults are tuned for a 4 GB VRAM GPU:

| Flag | Default | Description |
|---|---|---|
| `--corpus` | `data/processed/corpus.bin` | Tokenised corpus binary |
| `--config` | — | Path to a JSON config file (see below) |
| `--resume` | — | Checkpoint `.pt` to resume from |

**JSON config** (all keys optional):

```json
{
    "corpus_path":    "data/processed/corpus.bin",
    "checkpoint_dir": "checkpoints/pretrain/",

    "model": {
        "vocab_size":  16384,
        "d_model":     512,
        "n_layers":    6,
        "n_heads":     8,
        "n_kv_heads":  2,
        "d_ff":        1408,
        "max_seq_len": 1024,
        "dropout":     0.1
    },

    "training": {
        "peak_lr":          3e-4,
        "warmup_steps":     500,
        "total_steps":      10000,
        "batch_size":       4,
        "accumulate_steps": 8,
        "log_every":        50,
        "save_every":       1000
    }
}
```

On a 4 GB GPU with defaults, expect ~2–4 hours for 10 000 steps depending on corpus size. On CPU, this takes much longer — reduce `total_steps` for experimentation.

### Via Training UI

```bash
python -m grimoire_ai.ui
```

Open `http://localhost:7860`, go to the **Pre-train** tab, fill in the corpus path and hyperparameters, then click **Start pre-training**. Loss updates stream live.

---

## 4 — Fine-tune (instruction tuning)

Fine-tuning is a short second pass on structured `{user, assistant}` examples. It teaches the model to respond after `<AST>` instead of continuing the user's text.

### Prepare a JSONL dataset

Each line must be a JSON object with `user` and `assistant` fields. The optional `context` field inserts a corpus excerpt between `<SEP>` markers, matching the format the model sees at inference:

```json
{"user": "What happens when a creature is grappled?", "assistant": "A grappled creature has its speed reduced to zero."}
{"user": "What is a cantrip?", "assistant": "A cantrip is a spell that can be cast at will without expending a spell slot.", "context": "Cantrips are simple spells that require no spell slot."}
```

**Validate your dataset before training:**

```bash
python scripts/validate_finetune_data.py \
    --data  data/finetune/examples.jsonl \
    --vocab data/tokenizer/bpe.json \
    --max-seq-len 512
```

This reports example count, token length statistics, and the proportion of examples that would be truncated.

### Saga fine-tuning dataset

A ready-to-use dataset is included at `scripts/finetune_data/saga_v1.jsonl` — 30 examples covering D&D 5e rules, encounter math, and probability/statistics. Use the dedicated script:

```bash
python scripts/finetune_saga.py \
    --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json \
    --output-dir checkpoints/saga/
```

Defaults: 300 steps, peak lr 5e-5, batch 4, gradient accumulation 4. Per-step loss is printed to stdout.

### Via CLI (general)

```bash
python -m grimoire_ai.llm.training.finetune \
    --resume  checkpoints/pretrain/step_0010000.pt \
    --data    data/finetune/examples.jsonl \
    --vocab   data/tokenizer/bpe.json \
    --output  checkpoints/finetune/
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--resume` | required | Pre-trained checkpoint to start from |
| `--data` | required | Path to JSONL fine-tuning dataset |
| `--vocab` | `data/tokenizer/bpe.json` | BPE vocabulary |
| `--output` | `checkpoints/finetune/` | Output checkpoint directory |
| `--total-steps` | `500` | Much shorter than pre-training |
| `--peak-lr` | `5e-5` | Lower LR — nudge weights, not overwrite |
| `--max-seq-len` | `512` | Sequence length cap for fine-tuning examples |

### Via Training UI

Open `http://localhost:7860`, go to the **Fine-tune** tab, point it at your pre-trained checkpoint and JSONL dataset, and click **Start fine-tuning**.

---

## 5 — Tune embeddings for retrieval (optional, experimental)

> **Status:** experimental. The model's *generation* quality does not depend
> on this step — it only affects the quality of the **semantic retrieval**
> backend (`--encoder model` / `--encoder lora`). External encoders
> (MiniLM / MPNet) remain the stronger option today; this stage exists to
> close that gap with a retriever you trained yourself. Skip it if you are
> happy using an external encoder or lexical retrieval.

The base and fine-tuned checkpoints are only ever trained with a
next-token-prediction objective, which gives no pressure to place
semantically similar passages near each other in embedding space. As a
result, the model's *own* pooled embeddings under-perform dedicated sentence
encoders for retrieval. `scripts/embed_tune.py` adds the missing signal with
a short, self-supervised **contrastive** training pass that produces a small
LoRA adapter specialised for embeddings — no labels and no domain-specific
setup, so the same recipe works on any corpus of `.txt` files.

### Hard-negative batching

Each batch is built from a handful of documents with several passages each
(controlled by `--passages-per-doc`), so the in-batch negatives include
**same-document near-misses** (hard) alongside cross-document negatives
(easy). Pure random batching across a large, diverse corpus only ever
produces easy negatives — the loss then saturates by learning coarse topic
separation and never learns to distinguish confusable concepts (an earlier
random-batched run retrieved "disadvantage" passages for "advantage" queries
despite a low training loss). `--batch-size` must be a multiple of
`--passages-per-doc`.

### Run it

```bash
python scripts/embed_tune.py \
    --checkpoint checkpoints/finetune/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json \
    --corpus-dir data/corpus/saga/ \
    --output     checkpoints/lora/embed-saga/embed.lora
```

Output is a small `.lora` file (a few MB). The base checkpoint's weights are
frozen — only the adapter is trained. On GPU this is fast; on CPU a few
hundred steps still completes in minutes. **Embedding a large corpus for
evaluation is the slow part, not this training pass** — see the
`--encoder lora` notes in [setup-inference.md](setup-inference.md).

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Checkpoint to tune (base weights stay frozen) |
| `--corpus-dir` | required | Directory of `.txt` files to train on |
| `--output` | required | Output `.lora` path |
| `--batch-size` | `8` | Passages per batch; must be a multiple of `--passages-per-doc` |
| `--passages-per-doc` | `4` | Passages per document per batch (hard-negative grouping) |
| `--total-steps` | `500` | Training steps |
| `--rank` / `--alpha` | `8` / `16.0` | LoRA capacity and scaling |
| `--lr` | `1e-4` | AdamW learning rate |
| `--temperature` | `0.05` | InfoNCE softmax temperature |

### Consuming the adapter

The adapter is for the **embedding model only**. It reroutes the same
`q_proj`/`v_proj` weights that generation uses, so it must be loaded into a
*separate* model instance dedicated to retrieval — never into the engine
that generates chat responses, or generation output changes too. The
evaluation harness handles this split for you:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/finetune/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json \
    --corpus-dir data/corpus/saga/ \
    --quiz       scripts/eval_data/saga_quiz.jsonl \
    --encoder lora --lora checkpoints/lora/embed-saga/embed.lora
```

---

## 6 — Checkpoints

Checkpoints are saved as `step_NNNNNNN.pt` files in the output directory. Each checkpoint stores:

- Model weights
- Optimizer state
- Training step number and most recent average loss
- **Full model configuration** — no need to track architecture settings separately

To resume an interrupted run:

```bash
python -m grimoire_ai.llm.training.train \
    --resume checkpoints/pretrain/step_0005000.pt
```

---

## 7 — Verify training worked

Quick sanity check after fine-tuning:

```python
from grimoire_ai.llm.inference.engine import InferenceEngine

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
print(engine.respond("What happens when a creature is grappled?"))
```

If the model produces a direct answer rather than continuing the question, fine-tuning succeeded.

### Verify semantic retrieval quality

After training, the model's embeddings encode domain knowledge. You can probe retrieval quality directly without a full chat session:

```python
from grimoire_ai.llm.inference.engine import InferenceEngine
from pathlib import Path

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)

documents = [
    (path.read_text(encoding="utf-8"), path.stem)
    for path in sorted(Path("data/corpus/saga/").glob("*.txt"))
]
retriever = engine.build_semantic_corpus(documents, attach=False)

results = retriever.query("grapple speed movement", top_k=3)
for r in results:
    print(f"[{r.score:.3f}] {r.source}: {r.excerpt[:80]}")
```

At early checkpoints (~1k steps), scores will be broadly similar across passages — the embeddings are not yet meaningful. By ~15k steps scores should clearly separate on-topic from off-topic passages. A well-trained model will return the grapple speed rule as the top result with a score noticeably above unrelated passages.

---

## 8 — Corpus updates after training

Training is **step-count based**, not corpus-size based. The trainer samples random windows from `corpus.bin` until `total_steps` is reached, so a larger corpus means more variety per step — not more training time.

### Workflow

```
1. Add new .txt files to  data/corpus/saga/   (or wherever your corpus lives)
2. Re-run preprocessing   →  corpus.bin is rebuilt with the new files included
3. Resume pre-training    →  point --resume at your latest checkpoint
```

Step 2 reuses the existing `bpe.json` — tokenizer training is skipped automatically if the file already exists, so preprocessing is fast.

Step 3 is the important one: **resume, don't restart**. The model continues from its current weights and now draws from the richer corpus. A few thousand additional steps is usually enough to absorb new material; you do not need to redo the full original run.

```bash
# Re-preprocess (fast — tokenizer already trained)
python -m grimoire_ai.llm.data.preprocessing \
    --input  data/corpus/saga/ \
    --output data/processed/corpus.bin \
    --vocab  data/tokenizer/bpe.json

# Resume pre-training for extra steps
python -m grimoire_ai.llm.training.train \
    --corpus data/processed/corpus.bin \
    --resume checkpoints/pretrain/step_0010000.pt
```

The same applies to fine-tuning: if you add new JSONL examples, resume from the existing fine-tuned checkpoint rather than the pre-trained one.

The semantic embedding index is built from corpus `.txt` files at inference startup — it does not live inside the checkpoint and does not need a training step when you add corpus files.

### When to retrain the tokenizer

Delete `bpe.json` before preprocessing only if the new content is from a very different domain whose vocabulary the current BPE cannot represent well (e.g., a non-Latin-script language). For additional D&D, mathematics, or plain-English content, the existing 16 384-token vocabulary is sufficient and retraining the tokenizer would require a full pre-training run from scratch.

| Change | Action |
|---|---|
| Add more same-domain `.txt` files | Reprocess → resume pre-training (a few k steps) |
| Add a completely new domain | Reprocess → resume (or full retrain if the domain shift is large) |
| Add fine-tune examples only | Skip preprocessing; re-run fine-tuning from the fine-tuned checkpoint |
| Force tokenizer retrain | Delete `bpe.json`, reprocess, full pre-train from scratch |

See [setup-inference.md](setup-inference.md) for the full inference and chat setup.
