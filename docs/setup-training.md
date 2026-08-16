# Setting Up Training

This guide covers everything needed to prepare a corpus, pre-train a Grimoire model, and fine-tune it for conversational use — from environment setup through to a saved, chat-ready checkpoint.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11–3.13 (see CUDA note below) |
| RAM | 8 GB | 16 GB |
| GPU | — (CPU works) | NVIDIA RTX with 4 GB+ VRAM, or Apple Silicon (M1+) |
| CUDA | — | 12.4 (for RTX cards) |

The goal is "runs on consumer hardware" as broadly as possible — CPU-only
always works, on any OS. GPU acceleration is available two ways: NVIDIA
CUDA (Windows/Linux) or Apple's Metal backend, **MPS** (macOS on Apple
Silicon). There's no CUDA on macOS at all, which is why MPS exists as its
dedicated GPU path.

> **For CUDA use, keep Python within the range PyTorch ships CUDA wheels for**
> (currently up to 3.13). A newer Python still works CPU-only. See
> [§1 — CUDA](#cuda-windows--linux-rtx-card) for a dedicated-venv workaround
> if your default interpreter is too new. MPS has no such constraint — it
> ships in the same PyPI `torch` wheel used for CPU, so there's no separate
> version matrix to track.

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

> **Python version matters for CUDA.** PyTorch's CUDA wheels lag the newest
> Python releases. If `pip install torch --index-url .../cu124` fails with
> `Could not find a version that satisfies the requirement torch`, your
> Python is too new — at the time of writing the CUDA wheels top out at
> **Python 3.13** (`cp313`), so Python 3.14 has no CUDA build and silently
> installs the CPU-only wheel instead (`torch.cuda.is_available()` →
> `False`). Check with `python --version`. If you're on a Python newer than
> the latest CUDA wheel, use a dedicated CUDA venv (next section) pinned to a
> supported version rather than changing your system Python.

### Apple Silicon (macOS, MPS)

No separate install step — the standard PyPI wheel already includes MPS
support:

```bash
git clone https://github.com/BasLinders/grimoire.git
cd grimoire
pip install -e ".[dev]"
```

Verify MPS is visible:

```python
import torch
print(torch.backends.mps.is_available())   # True on M1/M2/M3/M4 Macs
```

Training/eval scripts and the UI auto-detect MPS the same way they
auto-detect CUDA: CUDA first, then MPS, then CPU, so **no script flags
change** on Apple Silicon either. Mixed-precision (bf16/fp16 AMP) and
`torch.compile`, which are CUDA-specific optimizations in this codebase,
don't apply on MPS — training still runs on the GPU, just without those
extra speedups, so it's still meaningfully faster than CPU. See
[speed_optimization.md](speed_optimization.md) item #6 for scoping a
proper MPS throughput path (not yet implemented).

> **Docker note:** the provided `Dockerfile` targets CPU/CUDA — Docker
> Desktop on macOS runs containers inside a Linux VM with no Metal
> passthrough, so a container can never reach the GPU on a Mac. Run
> natively (as above) for MPS acceleration; use Docker only for CPU-only
> workflows on a Mac.

#### A dedicated CUDA virtual environment

The cleanest way to get GPU acceleration without disturbing your main
environment — and the recommended approach if your default `python` is too
new for the CUDA wheels — is a separate venv pinned to a supported Python
version. Training/eval scripts auto-detect CUDA (`InferenceEngine` and the
trainers pick `cuda` when `torch.cuda.is_available()`, falling back to `mps`
on Apple Silicon and `cpu` otherwise), so **no script flags change** — only
which interpreter you run them with.

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
| Pre-train / Fine-tune | **none** — auto-detects `cuda`, else `mps` (Apple Silicon), else `cpu` | always CPU; no override |
| Evaluate | **Device** dropdown (`Auto` / `CPU` / `CUDA` / `MPS`) | selecting `CUDA` or `MPS` errors — the process has no GPU torch |

Confirm which device is actually in use from the on-screen log, not the fan
noise: training prints `Training on CUDA | …` / `Training on MPS | …` /
`Training on CPU | …`, and the Evaluate tab prints `Model loaded on cuda` /
`Model loaded on mps` / `Model loaded on cpu`.

### Corpus scraper (optional — for ingesting web/PDF/DOCX sources)

```bash
pip install -e ".[scraper]"         # web, PDF, DOCX, Markdown
pip install -e ".[scraper-ocr]"     # + image OCR via Tesseract
```

---

## 2 — Prepare your corpus

### Option A — Use the Saga corpus (D&D 5e SRD seed)

Download and process the CC-BY 4.0 D&D 5e SRD in one step:

```bash
python scripts/build_saga_corpus.py
```

This downloads the SRD (~1.5 MB from GitHub), splits it into 24 rule sections, converts each to plain text, and copies four hand-authored probability and encounter-math reference files into `data/corpus/saga/`.

This is a **minimal seed**, not the corpus actually used in production — that corpus has grown to over a thousand files across many sources (Gutenberg, Stack Exchange RPG Q&A, official rulebooks/adventures, Wikipedia/Wikibooks, and more), deduplicated and source-weighted. To reproduce it:

1. Run whichever `scripts/scrape_*.py` scripts match the sources you want (e.g. `scrape_gutenberg_catalog.py` for bulk public-domain fiction, `scrape_stackexchange_rpg.py` for the official RPG Stack Exchange data dump).
2. Run `scripts/dedup_corpus.py` (MinHash + LSH) to catch near-duplicates across everything you've added.
3. Tag categories with `--weight-pattern GLOB:WEIGHT` during preprocessing (§2 below) so bulk/generic content doesn't dilute domain-specific content at training time.

See [expansion_PLAN.md](expansion_PLAN.md) for the current source list, scale, and the reasoning behind the weighting scheme in use.

#### The current full recipe (preprocess → weight → train)

Once your corpus directories are in place — `data/corpus/saga/` plus any derived/synthetic directories kept separate on disk (e.g. `data/corpus/saga_derived/`; see "Corpus updates after training" §8 for why they're kept apart rather than merged in) — this is the actual sequence used for every pretrain run against the Saga corpus:

**1. Preprocess.** Combines every corpus directory into one build, drops obvious junk documents, and tags every file with its training weight:

```bash
python -m grimoire_ai.llm.data.preprocessing \
    --input data/corpus/saga/ \
    --input data/corpus/saga_derived/ \
    --output data/processed/corpus.bin \
    --vocab data/tokenizer/bpe.json \
    --quality-filter \
    --quality-report data/processed/quality_report.jsonl \
    --weight-pattern "gutenberg_*:0.5" \
    --weight-pattern "wp_fantasy_*:0.5" \
    --weight-pattern "wp_math_*:1" \
    --weight-pattern "wp_dnd_*:1" \
    --weight-pattern "rpg_se_*:1" \
    --weight-pattern "fr_wiki_*:1" \
    --weight-pattern "dragon_*:1" \
    --weight-pattern "dnd_*:1" \
    --weight-pattern "synth_*:1" \
    --weight-pattern "entigraph_*:1" \
    --weight-pattern "*:1.75"
```

Check `quality_report.jsonl` for anything unexpected before moving on — see "Pre-process corpus for training" above for what `--quality-filter` catches. Weight values and categories are project decisions, not framework defaults — see [expansion_PLAN.md](expansion_PLAN.md) for the reasoning behind this exact list and its history of fixes (e.g. the `wotc-srd`-only re-scrapes some of these globs assume), and check there before assuming it's still current.

**2. Build sample weights.** Must be rebuilt any time the corpus, the `--weight-pattern` rules, `val_split`, or `val_stratified` change — a mismatch here silently produces the wrong window count at training time:

```bash
python scripts/build_source_weights.py \
    --corpus data/processed/corpus.bin \
    --seq-len 1024 --stride 512 \
    --val-split 0.01 \
    --val-stratified \
    --output data/processed/sample_weights.npy
```

**3. Train.** Fresh run (no `--resume`) whenever the corpus content changed materially rather than just grew (§8 covers when that applies). Point your config's `checkpoint_dir` at a new, unused directory each time instead of overwriting the previous run, so you can still compare against it:

```bash
python -m grimoire_ai.llm.training.train --config your_config.json --val-stratified
```

`total_steps` and the fields derived from it are Chinchilla-scaled from model size and batch/accum/seq_len (see `docs/PARAM_OPT.md`), not from the corpus's own token count — a corpus content change alone doesn't require recomputing them, only a change to model size or batch config does.

**4. Verify.** See "Verify a pretrain checkpoint before fine-tuning" under §3 below.

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

### Source-based sample weighting (optional)

If your corpus mixes domain-specific content with bulk/generic filler (e.g. curated D&D rules text alongside a large volume of general public-domain fiction added for language variety), tag files by filename glob so training samples them at different rates instead of uniformly:

```bash
python -m grimoire_ai.llm.data.preprocessing \
    --input  data/corpus/saga/ \
    --output data/processed/corpus.bin \
    --vocab  data/tokenizer/bpe.json \
    --weight-pattern "gutenberg_*:0.5" \
    --weight-pattern "srd_*:1.75" \
    --weight-pattern "*:1"
```

Rules are matched in order against each file's name (`fnmatch`), first match wins; unmatched files default to weight `1.0`. This writes `<output>.doc_end_offsets.npy` / `<output>.doc_weights.npy` sidecars — `corpus.bin` itself is unchanged.

Turn those into the per-window array training actually consumes:

```bash
python scripts/build_source_weights.py \
    --corpus   data/processed/corpus.bin \
    --seq-len  1024 --stride 512 \
    --output   data/processed/source_weights.npy
```

If you're training with a validation split (`val_split` > 0 below), pass the **same** `--val-split` value here — the corpus is concatenated in alphabetically-sorted file order, so leaving this out (or mismatching it) silently scores the wrong region and eventually causes a window-count mismatch when `Trainer` tries to use it:

```bash
python scripts/build_source_weights.py \
    --corpus     data/processed/corpus.bin \
    --seq-len    1024 --stride 512 \
    --val-split  0.01 \
    --output     data/processed/source_weights.npy
```

Then point `sample_weights_path` at the resulting file in your training config (see §3 below). Prefer the Pre-train tab's **"Build sample weights from tags"** button over this CLI when using the UI — it derives `--val-split` from whatever the tab's own Validation split field is set to, so the two can't drift out of sync.

**Validation coverage for thin tiers.** The default validation split scatters blocks uniformly by raw token position — representative of the corpus as a whole, but with no guarantee that a tier that's a small fraction of the corpus (e.g. official rulebooks at ~10%) ends up with *any* validation windows; a handful of random blocks can miss a thin category entirely. If you want to evaluate per-tier loss (e.g. to check whether up-weighted content actually improved) reliably, add `--val-stratified` to both commands — it holds out `val_split` fraction *within each weight tier separately*, so every tier gets validation coverage proportional to its own size:

```bash
python -m grimoire_ai.llm.training.train --config your_config.json --val-stratified

python scripts/build_source_weights.py \
    --corpus         data/processed/corpus.bin \
    --seq-len        1024 --stride 512 \
    --val-split      0.01 \
    --val-stratified \
    --output         data/processed/source_weights.npy
```

Both commands must agree on `--val-stratified` (same as `--val-split`) or the resulting window count won't align. In the UI, this is the Pre-train tab's **"Stratify validation by weight tags"** checkbox, next to Validation split.

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
    "corpus_path":         "data/processed/corpus.bin",
    "checkpoint_dir":      "checkpoints/pretrain/",
    "sample_weights_path": "data/processed/source_weights.npy",

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
        "save_every":       1000,
        "val_split":        0.01,
        "eval_every":       1000,
        "eval_batches":     50
    }
}
```

`sample_weights_path` is optional — omit it entirely for uniform sampling. `val_split` holds out that fraction of the corpus for a validation loss logged every `eval_every` steps; it's scattered across many small blocks rather than one contiguous chunk, so it stays a representative sample no matter how the corpus is structured.

On a 4 GB GPU with defaults, expect ~2–4 hours for 10 000 steps depending on corpus size. On CPU, this takes much longer — reduce `total_steps` for experimentation.

### Via Training UI

```bash
python -m grimoire_ai.ui
```

Open `http://localhost:7860`, go to the **Pre-train** tab, fill in the corpus path and hyperparameters, then click **Start pre-training**. Loss updates stream live.

### Verify a pretrain checkpoint before fine-tuning

A single aggregate validation loss can't tell you whether weighting is actually working, and it isn't comparable across runs that used different validation-split methods (contiguous-tail vs. scattered-block vs. `--val-stratified`) — see "Source-based sample weighting" above. Two scripts cover this gap for a *raw*, not-yet-fine-tuned checkpoint:

**Per-tier validation loss** (`scripts/eval_per_tier.py`) — requires the corpus to have `--weight-pattern` sidecars and the checkpoint to have trained with `--val-stratified` (reproduces that exact held-out split, just reported per tier instead of merged into one number):

```bash
python scripts/eval_per_tier.py \
    --checkpoint checkpoints/pretrain/<run>/step_XXXXXXX.pt \
    --corpus data/processed/corpus.bin \
    --val-split 0.01
```

Look for the loss ordering to match your intended weight prioritization (down-weighted tiers worst, up-weighted tiers best) — that's the real signal, not the absolute numbers.

**Qualitative completion check** (`scripts/qualitative_check.py`) — a raw pretrain checkpoint hasn't learned to follow a chat/instruction format yet, so `grimoire-chat`'s conversational template (and its missing `--repetition-penalty` support) isn't the right tool here. This generates fixed-prompt text completions instead, with full sampling control:

```bash
python scripts/qualitative_check.py \
    --checkpoint checkpoints/pretrain/<run>/step_XXXXXXX.pt \
    --vocab data/tokenizer/bpe.json
```

Eyeball the completions for coherence, on-topic terminology, and the absence of repetition loops (`does does does...`) or question-echoing — the same checks this project's prior ad-hoc qualitative reviews (`docs/expansion_PLAN.md`) looked for by hand.

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

`scripts/finetune_data/` has several ready-to-use JSONL sets, grown well past the original 30-example seed (`saga_v1.jsonl`): `saga_v2.jsonl` and `saga_dnd_math.jsonl` (D&D rules/encounter math, the latter rewritten to use `<TOOL:python>` tags instead of declining arithmetic — see the Math Tool item in [PLAN.md](PLAN.md)), `tool_call_examples.jsonl` (15 math-tool-call examples), and `general_conversations.jsonl` (64 pairs for base instruction fine-tuning, used by LoRA agent fine-tuning). The production checkpoint referenced in `agents.json` is fine-tuned on data built by `scripts/build_finetune_data_from_qa.py` from the cleaned Q&A corpus, not the seed dataset alone.

For a quick end-to-end run on the seed dataset:

```bash
python scripts/finetune_saga.py \
    --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json \
    --output-dir checkpoints/saga/
```

Defaults: 300 steps, peak lr 5e-5, batch 4, gradient accumulation 4. Per-step loss is printed to stdout.

For LoRA-based agent fine-tuning (freezes base weights, trains ~0.5% of parameters as a small `.lora` adapter) rather than a full fine-tune, use the Fine-tune tab's **Mode** dropdown (Base instruction fine-tune vs. Agent LoRA adapter) or the `--lora-rank`/`--lora-alpha` flags on `grimoire-finetune`.

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
    --encoder lora --lora checkpoints/lora/embed-saga/embed.lora \
    --quiz-repetition-penalty 1.3
```

> **Always pass `--quiz-repetition-penalty 1.3` for quiz evaluation.** The
> flag defaults to `1.0` (no penalty), which leaves greedy decoding free to
> fall into repetition loops ("a critical hit is a critical hit") on a small
> model — degenerate output that depresses quiz scores and does **not**
> reflect real use, since the chat/agent runtime already generates with a
> `1.3` penalty (see `agents.json`). The default is left at `1.0` so the
> harness measures *unmodified* generation by default, but the standing
> recipe for any quiz comparison should match the deployment setting. Keep
> the penalty **identical across every run you compare** — it shifts absolute
> pass-rates, so mixing `1.0` baselines with `1.3` candidates is not a fair
> comparison.

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

Step 3 is usually **resume, don't restart**: the model continues from its current weights and now draws from the richer corpus. A few thousand additional steps is usually enough to absorb a modest, incremental addition; you do not need to redo the full original run.

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

**When to restart from scratch instead.** Resuming only works well when the checkpoint still has meaningful learning rate left in its cosine schedule and the corpus change is incremental, not substantial. Prefer a fresh run when either is true:

- **The corpus content changed materially**, not just grew — e.g. a cleanup pass rewrote a large fraction of existing text (markup stripped, boilerplate removed), not just new files appended. The old checkpoint was shaped by the *previous* version of that text.
- **The existing checkpoint is near the end of its schedule** (LR has decayed close to `min_lr`, i.e. ~10% of peak). Resuming there applies the new material at a learning rate too small to meaningfully absorb it — you'd get a checkpoint that's barely different from before, not one that's actually learned from the change.

If either applies, start over with `--resume` omitted (or blank in the UI) rather than continuing a run whose weights and LR schedule were shaped by a meaningfully different corpus.

The same reasoning applies to fine-tuning: if you add new JSONL examples, resume from the existing fine-tuned checkpoint rather than the pre-trained one — unless the fine-tune data changed substantially enough that the same restart-vs-resume tradeoff applies.

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
