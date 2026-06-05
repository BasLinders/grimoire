# Setting Up Training

This guide covers everything needed to prepare a corpus, pre-train a Grimoire model, and fine-tune it for conversational use — from environment setup through to a saved, chat-ready checkpoint.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11+ |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works) | NVIDIA RTX with 4 GB+ VRAM |
| CUDA | — | 12.4 (for RTX cards) |

---

## 1 — Install

### CPU only

```bash
git clone https://github.com/BasLinders/grimoire.git
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

### Training UI (optional)

```bash
pip install -e ".[ui]"
```

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
python -m grimoire.ui
# → Ingest tab
```

Paste a URL, upload a file, or point at a directory. Choose a cleaning level (minimal / standard / thorough) and click **Start ingestion**.

**Via CLI:**

```bash
# Web page
python -m grimoire.corpus.ingest --source https://example.com/rules --output data/raw/

# Raw Markdown URL (GitHub etc.) — detected automatically
python -m grimoire.corpus.ingest \
    --source https://raw.githubusercontent.com/user/repo/main/doc.md \
    --output data/raw/

# Local file (PDF, DOCX, Markdown, plain text)
python -m grimoire.corpus.ingest --source docs/phb_excerpt.pdf --output data/raw/

# Directory (batch, with cleaning level)
python -m grimoire.corpus.ingest \
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
python -m grimoire.llm.data.preprocessing \
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
python -m grimoire.llm.training.train \
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
python -m grimoire.ui
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
python -m grimoire.llm.training.finetune \
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

## 5 — Checkpoints

Checkpoints are saved as `step_NNNNNNN.pt` files in the output directory. Each checkpoint stores:

- Model weights
- Optimizer state
- Training step number and most recent average loss
- **Full model configuration** — no need to track architecture settings separately

To resume an interrupted run:

```bash
python -m grimoire.llm.training.train \
    --resume checkpoints/pretrain/step_0005000.pt
```

---

## 6 — Verify training worked

Quick sanity check after fine-tuning:

```python
from grimoire.llm.inference.engine import InferenceEngine

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
print(engine.respond("What happens when a creature is grappled?"))
```

If the model produces a direct answer rather than continuing the question, fine-tuning succeeded.

See [setup-inference.md](setup-inference.md) for the full inference and chat setup.
