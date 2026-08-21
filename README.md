# Grimoire

**Grimoire** is a hybrid small language model (SLM) engine built entirely from scratch, designed to run on consumer hardware with optional GPU acceleration. It pairs a semantic retrieval engine with a scratch-built transformer LLM — the retrieval engine grounds answers in your corpus, the LLM provides coherent conversational language, and a retrieval-score router decides per-query whether grounding is needed or the model should answer from its own knowledge.

Grimoire is intentionally domain-agnostic. Agents are named configurations: each agent declares its corpus, its checkpoint, and its generation defaults. The engine stays the same.

The first agent built on Grimoire is **Saga**: a focused domain chatbot covering Dungeons & Dragons rules, encounter mathematics, and probability / data science.

## Goals

- Run on consumer hardware (CPU, NVIDIA CUDA, or Apple Silicon MPS — no cloud required)
- Grounded domain facts from semantic retrieval; coherent conversation from the LLM
- Automatic routing: only inject corpus context when retrieval finds a confident match
- Full conversational coherence with multi-turn context tracking
- Modular: swap corpora, agents, and checkpoints independently
- One model powers both retrieval and generation — no separate embedding server
- Built from scratch as a learning project: tokenizer, transformer, training loop, and fine-tuning pipeline are all hand-written

## Repository Structure

```
grimoire/
├── grimoire_ai/
│   ├── agents/             # Agent registry — loads agents.json, builds InferenceEngine
│   ├── corpus/             # Corpus ingestion, stemming, n-gram index, Jaccard retrieval
│   ├── llm/                # Scratch-built transformer LLM
│   │   ├── tokenizer/      # Byte-level BPE tokenizer (default vocab size 16 384; extend() grows it without retraining)
│   │   ├── model/          # Decoder-only transformer — GQA or MLA attention, RoPE, SwiGLU, RMSNorm,
│   │   │                   #   optional multi-token prediction, optional RETRO-style chunked cross-attention
│   │   ├── data/           # TokenizedDataset, PaddingCollator, ConversationDataset, quality filter,
│   │   │                   #   neighbor-augmented dataset for RETRO-style training
│   │   ├── training/       # Trainer, checkpointing, pretrain + finetune entry points
│   │   └── inference/      # PromptBuilder, KV-cache sampler, InferenceEngine, SemanticRetriever,
│   │                       #   cross-encoder reranker, CRAG corrective filter, constrained decoding
│   ├── state/              # ConversationState — rolling multi-turn history + prompt build
│   ├── cli/                # Interactive terminal chat loop
│   └── ui/                 # Two Gradio apps: training/eval (Preprocess/Pre-train/Fine-tune/Scale/Evaluate/Ingest/Corpus) + chat
├── agents.json             # Named agent configurations (checkpoint, vocab, corpus, gen defaults)
├── scripts/
│   ├── scrape_*.py                 # Per-source corpus scrapers: Wikipedia, Wikibooks, arXiv,
│   │                               #   Gutenberg (+ catalog-based bulk variant), D&D Wiki, GitHub
│   │                               #   D&D repos, Fandom wikis, Open5e, 5etools, Internet Archive
│   │                               #   Dragon/Dungeon magazines, Stack Exchange RPG
│   ├── dedup_corpus.py             # MinHash + LSH near-duplicate detection across the corpus
│   ├── clean_stackexchange_markup.py  # Strip vote-score/tag/Markdown scaffolding from SE dumps
│   ├── build_saga_corpus.py        # Seed corpus: D&D 5e SRD sections + math references
│   ├── build_source_weights.py     # Per-window sample_weights.npy from --weight-pattern tags
│   ├── build_finetune_data_from_qa.py  # Build fine-tune JSONL from cleaned Q&A corpus data
│   ├── finetune_saga.py            # Fine-tune a checkpoint on a Saga JSONL dataset
│   ├── validate_finetune_data.py   # Pre-flight check on any JSONL dataset
│   ├── embed_tune.py               # Contrastive LoRA fine-tuning for retrieval embeddings
│   ├── score_difficulty.py         # Difficulty-based per-window sample weighting
│   ├── generate_open5e_entigraph.py  # EntiGraph-style entity recombination from Open5e's structured data
│   ├── evaluate.py                 # Perplexity / retrieval hit-rate / Q&A quiz CLI
│   ├── export_gguf.py              # Export a GQA checkpoint to GGUF for llama.cpp
│   ├── finetune_data/              # JSONL fine-tune datasets (Saga Q&A, math tool calls, general chat)
│   └── saga_references/            # Hand-authored math/probability reference .txt files
├── docs/                   # Setup guides, roadmap, and corpus-expansion history
├── data/                   # Runtime data — gitignored (corpus bins, tokenizer, checkpoints)
│   ├── raw/                # Source .txt files for pre-training corpus
│   ├── processed/          # Tokenised corpus.bin
│   ├── tokenizer/          # bpe.json — trained BPE vocabulary
│   ├── corpus/
│   │   └── saga/           # Populated by build_saga_corpus.py + the scrape_*.py scripts
│   └── finetune/           # Any additional fine-tuning datasets
├── checkpoints/            # Saved model checkpoints — gitignored
└── tests/                  # unit tests + integration tests — all green
```

## Architecture

### Flow

```mermaid
flowchart TD
    A([User message]) --> B[ConversationState\nrolling turn history]

    B --> C[SemanticRetriever\ncosine similarity over\nmodel embeddings]
    C --> R{Retrieval router\nscore ≥ threshold?}
    R -->|yes — inject context| E
    R -->|no — pure-chat| E

    B -->|packed prompt ids| E[GrimoireTransformer\nPromptBuilder → KV-cache sampler]

    E --> F([Response])
    F --> B
```

### How the hybrid works

Every user query is embedded by the same transformer that will generate the reply. That embedding is compared against pre-indexed corpus passage vectors (also embedded by the same model). If the top match exceeds the **retrieval threshold**, the passage is injected into the prompt as grounding context before generation. If no passage clears the threshold — because the query is conversational rather than domain-factual — the model answers from its own knowledge, without cluttering the prompt.

Both halves run the same model, in the same learned representation space. There is no separate embedding server and no external vector database.

### Component Table

| Component | Role | Status |
|---|---|---|
| **BPE Tokenizer** | Byte-level Byte-Pair Encoding; default vocab 16 384; lossless round-trip for any Unicode; opt-in `extend()` grows an existing vocabulary with new merges while preserving every existing token id, so old checkpoints stay loadable | ✓ done |
| **Corpus Scrapers** | `ingest()` dispatcher (web / PDF / DOCX / Markdown / OCR); dedicated scrapers for Wikipedia, Wikibooks, arXiv abstracts, Gutenberg (curated lists, plus a catalog-CSV-based bulk variant that filters Gutenberg's official bulk feed locally rather than scraping search pages), D&D Wiki, GitHub D&D repos, Fandom wikis, Open5e, 5etools, Internet Archive Dragon/Dungeon magazines, and Stack Exchange RPG (official data dump, not live scraping) | ✓ done |
| **Near-Duplicate Dedup** | MinHash + LSH near-duplicate removal; word 5-gram shingling, SHA-1 hashing, union-find clustering, longest-kept policy | ✓ done |
| **Source-Based Sample Weighting** | `--weight-pattern GLOB:WEIGHT` on `grimoire-preprocess` tags documents by filename glob; `scripts/build_source_weights.py` (or the Pre-train tab's "Build sample weights from tags" button) turns those into a per-window `sample_weights.npy` consumed by `Trainer`'s `WeightedRandomSampler` — upweight domain-specific content, downweight bulk filler, without touching `corpus.bin` itself | ✓ done |
| **Heuristic Quality Filter** | `grimoire_ai/llm/data/quality_filter.py` — dependency-free, Gopher/C4-style document screening (char/word minimums, alpha ratio, symbol-junk ratio, mean word length, short-line ratio for nav-menu detection, 3-gram repetition); opt-in `--quality-filter` on `grimoire-preprocess` with an optional report | ✓ done |
| **EntiGraph Entity Recombination** | `scripts/generate_open5e_entigraph.py` synthesizes training text by connecting entities from Open5e's structured API data, rather than general-purpose LLM rephrasing (evaluated and ruled out — model-collapse risk) | ✓ done |
| **Corpus Engine** | Ingests text, indexes stemmed 4-gram multi-tokens, retrieves top-k passages by Jaccard similarity (lexical fallback) | ✓ done |
| **Semantic Retriever** | Chunks documents into passages, embeds each with the model's own representations, and ranks by cosine similarity — the primary retrieval path | ✓ done |
| **Persistent RAG Index** | Chunk embeddings pre-computed once and cached as a numpy memmap (optional FAISS `IndexFlatIP` for larger corpora), with MD5-hash staleness checks against source files and checkpoint — replaces per-session recomputation; "Build/Rebuild index" button in the Corpus tab | ✓ done |
| **Cross-Encoder Reranker** | Second-stage rescoring of a widened first-stage candidate pool via a cross-encoder (`ms-marco-TinyBERT-L-2-v2` or `ms-marco-MiniLM-L-12-v2`, `sentence-transformers`) before the final top-k slice; opt-in, off by default | ✓ done |
| **CRAG Corrective Filter** | `CragFilter` classifies retrieved passages Correct / Ambiguous / Incorrect against two confidence thresholds, demoting (not dropping) Ambiguous passages; optional `corrective_retry` issues one widened local re-query when nothing clears the confidence bar; layers on top of, not in place of, the retrieval router below; wired in for Saga | ✓ done |
| **Retrieval Router** | Compares the top retrieval score against a configurable threshold; routes to grounded or pure-chat generation per query | ✓ done |
| **GrimoireTransformer** | Scratch-built decoder-only transformer (25 M / 85 M / 250 M params); RoPE, SwiGLU, RMSNorm, weight-tied output head; `embed()` for sentence embeddings; gradient checkpointing support | ✓ done |
| **Attention: GQA / MLA** | `TransformerConfig.attention_type` selects Grouped Query Attention (default) or Multi-Head Latent Attention — MLA compresses K/V into a shared low-rank latent with decoupled RoPE, ~45% smaller KV-cache at ~0.3% val-loss cost; old checkpoints without the setting default to GQA transparently | ✓ done |
| **Multi-Token Prediction (MTP)** | Optional auxiliary prediction heads (`TransformerConfig.n_predict`, default `0`) predict several future tokens during pretraining to improve sample efficiency; excluded from `val_loss` on purpose so MTP-on/off runs stay comparable | ✓ done |
| **RETRO-style Chunked Cross-Attention** | Optional cross-attention sublayers (`TransformerConfig.retro_layers`) let the model attend directly to retrieved neighbor chunks inside the transformer, instead of only via prompt concatenation; trained with a neighbor-augmented dataset that pairs precomputed neighbor ids with training windows | ✓ done |
| **Training Pipeline** | AdamW + cosine-warmup LR, bf16/fp16 AMP (bf16 preferred on Ampere+), Flash Attention (SDPA), `torch.compile`, gradient accumulation, gradient checkpointing, SWA, bootstrap early stopping, difficulty- or source-weighted sampling, scattered-block or weight-stratified validation split | ✓ done |
| **Instruction Fine-tuning** | `ConversationDataset` on `{user, assistant, context?}` JSONL; response-only loss masking | ✓ done |
| **LoRA / Adapter Fine-Tuning** | Rank-decomposition adapters fused into frozen attention projections (both GQA and MLA target sets); ~0.5% of parameters trained; each domain persona becomes a 2–5 MB `.lora` file, hot-swappable without reloading the base model; separate contrastive-embedding LoRA path for retrieval quality (`embed_tune.py`) | ✓ done |
| **Inference Engine** | PromptBuilder (corpus → prompt), KV-cache autoregressive sampler with adaptive entropy temperature, `respond()`, `chat()`, `chat_stream()`, `embed()`, `build_semantic_corpus()`; optional int8 quantization | ✓ done |
| **Constrained Decoding** | Opt-in logit-masking guards: a stat-block constraint restricting tokens after a recognized field label to well-formed continuations, and a repetition-loop guard that hard-bans the token that would extend an established exact repeat cycle (catches exact repeats only, not templated loops with varying content) | ✓ done |
| **KV-Cache** | Caches K/V projections: O(n²) → O(1) per generation step; sliding-window truncation at `max_seq_len` | ✓ done |
| **int8 Quantization** | `InferenceEngine(quantize=True)` replaces all Linear layers with dynamic int8 equivalents; ~4× smaller, faster on CPU; uses `torchao` when available, falls back to `torch.ao` | ✓ done |
| **GGUF Export** | Binary GGUF v3 writer + CLI export a GQA checkpoint for llama.cpp deployment, optional Q4_K_M quantization; MLA checkpoints aren't supported yet (`NotImplementedError` — llama.cpp's tensor/metadata conventions for that architecture weren't verifiable without a real binary to test against) | ✓ done |
| **Conversation State** | `ConversationState` packs rolling history newest-first within the token budget, then fills remaining space with corpus context | ✓ done |
| **Evaluation Harness** | Perplexity / BPC on held-out corpus, retrieval hit-rate over a fixed query set, keyword-recall + best-matching-window token-F1 Q&A quiz (searches all contiguous response windows against the reference, excludes question-shared vocabulary); `run_eval()` harness writes timestamped JSON to `data/eval/`; CLI at `scripts/evaluate.py` | ✓ done |
| **Math Tool** | `MathTool` detects arithmetic in queries, evaluates safely via pure-`ast` visitor (no `eval()`), injects result as context; resolves `<TOOL:python>…</TOOL>` tags from fine-tuned models; stdlib functions (factorial, exp, comb, hypot, trig, …) + scipy stats (norm_cdf, binom_pmf, t_ppf, …) with graceful fallback; `--math-tool` CLI flag; checkbox in the chat UI | ✓ done |
| **Training/Eval UI** | Gradio app: Preprocess (BPE training, `--extend-vocab`, weight-pattern tagging), Pre-train (attention-type + MLA dimension fields, size presets, gradient checkpointing, `torch.compile` mode, Chinchilla-optimal step suggestion from corpus size, sample-weight building), Fine-tune (LoRA rank/alpha/targets, step suggestion from dataset example count), Scale (Chinchilla calculator), Evaluate, Ingest, Corpus (pre-build/rebuild semantic index) tabs | ✓ done |
| **Chat UI** | Separate Gradio app: scrolling `gr.Chatbot` transcript, pinned input, agent selector or manual checkpoint loading, int8 toggle, adaptive temperature, retrieval controls (including reranker + CRAG threshold sliders, repetition-loop guard), math tool, dataset builder for saving exchanges as fine-tune pairs | ✓ done |
| **Agent Registry** | `AgentRegistry` reads `agents.json`; `build_engine(key, quantize=)` returns a ready `InferenceEngine` with corpus (and, where configured, CRAG filtering) auto-loaded | ✓ done |
| **Agent Routing** | `AgentRouter` scores each query against every registered agent's corpus and dispatches automatically when the top score clears a threshold; `MultiAgentEngine` hot-swaps the active LoRA adapter and corpus per turn (sub-second vs. minutes for a full model reload) behind one shared `InferenceEngine` | ✓ done |
| **Saga Corpus** | `scripts/build_saga_corpus.py` builds a minimal SRD + math-reference seed; the corpus actually in use has grown far beyond that seed via the scrapers above (Gutenberg, Stack Exchange RPG, Forgotten Realms wiki, official rulebooks/adventures, Wikipedia/Wikibooks) plus MinHash dedup and source-based weighting — see [docs/expansion_PLAN.md](docs/expansion_PLAN.md) for current scale and composition | ✓ done |
| **Saga Fine-tune Dataset** | Multiple JSONL sets in `scripts/finetune_data/` (D&D rules/math Q&A, math-tool-call examples, general conversation) plus `scripts/build_finetune_data_from_qa.py` to derive fine-tune examples from the cleaned Q&A corpus data; the production checkpoint in `agents.json` is fine-tuned on the latter, not the seed dataset alone | ✓ done |
| **Integration Tests** | End-to-end: BPE train → corpus build → model pretrain → checkpoint → engine load → multi-turn `chat()` | ✓ done |

### Multi-turn prompt format

```
<BOS> [<SEP> {corpus context} <SEP>] <USR> q1 <AST> a1
                                     <USR> q2 <AST> a2
                                     …
                                     <USR> current query <AST>
```

History is packed newest-first within the token budget. When retrieval clears the threshold, the corpus context fills whatever space remains after history. If the budget is exhausted the oldest turns are dropped first, then the context is trimmed. The current query is always preserved.

### Why Hybrid

| Approach | Strength | Weakness |
|---|---|---|
| LLM alone | Fluent, coherent, conversational | Hallucinates domain facts; expensive to specialise |
| Corpus alone | Accurate, deterministic, explainable | Cannot track context or produce natural sentences |
| **Grimoire (hybrid)** | Grounded facts when the corpus is relevant + coherent language always + pure-chat when it isn't | Slightly more complex setup |

New domain knowledge requires only adding `.txt` files to the corpus — no retraining. Because retrieval uses the model's own embeddings, the index improves as training progresses.

### LLM Architecture

The transformer uses four improvements over the GPT-2 baseline:

| Technique | Replaces | Benefit |
|---|---|---|
| **RMSNorm** | LayerNorm | Removes mean-centering; marginally faster, equally stable |
| **RoPE** | Learned positional embeddings | Encodes relative position via rotation; zero extra parameters |
| **SwiGLU** | GELU feed-forward | Gated activation; consistently outperforms GELU at equal parameter budget |
| **Grouped Query Attention** | Multi-head attention | `n_kv_heads=2` vs `n_heads=8`; 4× smaller KV cache at inference |

**Attention is configurable** via `TransformerConfig.attention_type`. GQA (above) is the default and the only attention type any shipped checkpoint or size preset uses. **Multi-Head Latent Attention (MLA)** is an opt-in alternative that compresses K/V into a shared low-rank latent with decoupled RoPE instead of caching full per-head K/V — reported ~45% smaller KV-cache at ~0.3% validation-loss cost, which matters more here than in a typical LLM because retrieved corpus excerpts inflate prompt length every turn. Checkpoints saved before `attention_type` existed default to `"gqa"` on load.

Two further opt-in training-time additions, both off by default (`0` / `None`) so they reproduce prior behaviour exactly when unset: **Multi-Token Prediction** (`n_predict` auxiliary heads predicting several future tokens per step, for sample efficiency) and **RETRO-style chunked cross-attention** (`retro_layers`, letting select transformer blocks attend directly to retrieved neighbor chunks rather than only via prompt concatenation).

Three size presets (all share `vocab_size=16384`, `max_seq_len=1024`, GQA attention):

| Preset | d_model | n_layers | n_heads | n_kv_heads | d_ff | Params | fp32 size |
|---|---|---|---|---|---|---|---|
| **small-25M** | 512 | 6 | 8 | 2 | 1408 | ~25 M | ~100 MB |
| **medium-85M** | 768 | 12 | 12 | 3 | 2048 | ~85 M | ~340 MB |
| **large-250M** | 1024 | 20 | 16 | 4 | 2816 | ~250 M | ~1 GB |

With `InferenceEngine(quantize=True)` the fp32 size shrinks ~4× (int8 Linear layers).

**Training optimisations:** Flash Attention (SDPA), `torch.compile`, `cudnn.benchmark`, non-blocking GPU transfers, bf16/fp16 AMP (bf16 preferred on Ampere+, fp16 fallback on older CUDA), gradient accumulation, gradient checkpointing (halves VRAM at ~20% speed cost), Stochastic Weight Averaging, bootstrap-confidence early stopping, difficulty-weighted sampling. `torch.compile`/`cudnn.benchmark`/AMP are CUDA-specific; on Apple Silicon (MPS) training still runs on the GPU, just without those extra speedups.

## Agents

### Saga

Saga is the first Grimoire agent. It focuses on:

- **D&D 5e rules**: conditions, classes, spells, monsters, combat mechanics — sourced from the CC-BY 4.0 Systems Reference Document, official rulebooks/adventures, and Stack Exchange RPG Q&A
- **Encounter mathematics**: XP budgets, CR scaling, multipliers, action economy
- **Probability & statistics**: dice distributions, DPR calculations, hit probabilities, death save odds

**Minimal setup** (SRD-only seed corpus — enough to get the pipeline running, far smaller than the corpus actually in production):

```bash
# 1. Build the seed corpus (downloads D&D 5e SRD ~1.5 MB, takes ~30 s)
python scripts/build_saga_corpus.py

# 2. Pre-train a model (or use an existing checkpoint)
python -m grimoire_ai.llm.training.train

# 3. Validate a fine-tuning dataset
python scripts/validate_finetune_data.py \
    --data  scripts/finetune_data/saga_v1.jsonl \
    --vocab data/tokenizer/bpe.json

# 4. Fine-tune on it
python scripts/finetune_saga.py \
    --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json

# 5. Update agents.json with the fine-tuned checkpoint path, then load
#    Saga from the agent dropdown in the chat app.
python -m grimoire_ai.ui.chat_app
```

**Reproducing the full production corpus** is a larger, ongoing effort — see [docs/expansion_PLAN.md](docs/expansion_PLAN.md) for the current source list, scale, dedup process, source-weighting scheme, and open decisions. In short: run the `scrape_*.py` scripts for whichever sources you want, `dedup_corpus.py` to catch near-duplicates, tag categories with `--weight-pattern` during `grimoire-preprocess`, and build fine-tune data with `scripts/build_finetune_data_from_qa.py` rather than the minimal `saga_v1.jsonl` alone.

## Usage

### Ingest a corpus source

```bash
# Web page (HTML)
python -m grimoire_ai.corpus.ingest --source https://example.com/rules --output data/raw/

# Raw Markdown URL (e.g. GitHub) — detected automatically, no HTML parsing
python -m grimoire_ai.corpus.ingest --source https://raw.githubusercontent.com/user/repo/main/doc.md --output data/raw/

# Local file (PDF, DOCX, Markdown, plain text)
python -m grimoire_ai.corpus.ingest --source docs/phb_excerpt.pdf --output data/raw/

# Directory (batch)
python -m grimoire_ai.corpus.ingest --source docs/ --output data/raw/ --recursive
```

Or from the **Ingest** tab in `python -m grimoire_ai.ui`.

### Pre-process corpus for training

```bash
python -m grimoire_ai.llm.data.preprocessing \
    --input  data/raw/ \
    --output data/processed/corpus.bin \
    --vocab  data/tokenizer/bpe.json
```

When the corpus grows and `data/tokenizer/bpe.json` already exists, add `--extend-vocab --vocab-size <larger>` to grow the vocabulary with new merges instead of retraining from scratch — existing token ids are preserved, so old checkpoints stay loadable after resizing their embedding/output layers (`--resume` in `train.py`/`finetune.py` does this automatically).

### Pre-train

```bash
python -m grimoire_ai.llm.training.train
# or via UI:  python -m grimoire_ai.ui  → Pre-train tab
```

### Fine-tune

```bash
python -m grimoire_ai.llm.training.finetune \
    --resume  checkpoints/step_0010000.pt \
    --data    data/finetune/examples.jsonl \
    --vocab   data/tokenizer/bpe.json \
    --output  checkpoints/finetune/
# or via UI:  python -m grimoire_ai.ui  → Fine-tune tab
```

### Chat (terminal)

```bash
# Ungrounded
python -m grimoire_ai.cli.chat \
    --checkpoint checkpoints/finetune/step_0000500.pt \
    --vocab      data/tokenizer/bpe.json

# Semantic retrieval with routing threshold
python -m grimoire_ai.cli.chat \
    --checkpoint         checkpoints/finetune/step_0000500.pt \
    --vocab              data/tokenizer/bpe.json \
    --corpus-dir         data/corpus/saga/ \
    --semantic \
    --retrieval-threshold 0.0
```

Commands: `/clear` (reset history), `/history` (review turns), `/quit`.

### Chat (Python API)

```python
from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.state.conversation import ConversationState

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
# Build a semantic corpus and attach it; set a routing threshold.
engine.build_semantic_corpus(
    [("data/corpus/saga/srd.txt", "srd")],  # (text, source) tuples
)
engine.retrieval_threshold = 0.0  # route to pure-chat when no passage clears 0.0

state = ConversationState()
r1 = engine.chat("What happens when a creature is grappled?", state)
r2 = engine.chat("How do I escape the grapple?", state)  # model sees prior turn
```

### Load a named agent (Python API)

```python
from grimoire_ai.agents.registry import AgentRegistry

registry = AgentRegistry("agents.json")
engine = registry.build_engine("saga")  # loads checkpoint + corpus from agents.json
```

### Training/eval UI

```bash
pip install -e ".[ui]"
python -m grimoire_ai.ui
# open http://localhost:7860
```

Seven tabs: **Preprocess** (BPE training, `--extend-vocab`, weight-pattern tagging for source-based sample weighting), **Pre-train** (model size presets, gradient checkpointing, `torch.compile` mode, "Build sample weights from tags" button), **Fine-tune** (LoRA rank/alpha/targets), **Scale** (Chinchilla scaling calculator), **Evaluate** (perplexity, retrieval hit-rate, Q&A quiz), **Ingest** (multi-file upload), **Corpus** (pre-build the semantic embedding index).

### Chat UI

```bash
pip install -e ".[ui]"
python -m grimoire_ai.ui.chat_app
# open http://localhost:7861
```

A separate, dedicated app — not a tab in the training UI. Streaming responses in a scrolling transcript, agent selector or manual checkpoint loading, corpus directory + embedding backend + retrieval threshold, generation controls (temperature, top-k/top-p, adaptive temperature, repetition-loop guard), math tool, and a dataset builder for turning good exchanges into fine-tune pairs.

### Console scripts

`pip install -e .` registers entry points for the package's CLI tools, so the `python -m ...` invocations above can be replaced with short commands on `PATH`:

| Command | Equivalent to |
|---|---|
| `grimoire-chat` | `python -m grimoire_ai.cli.chat` |
| `grimoire-ui` | `python -m grimoire_ai.ui` |
| `grimoire-chat-ui` | `python -m grimoire_ai.ui.chat_app` |
| `grimoire-train` | `python -m grimoire_ai.llm.training.train` |
| `grimoire-finetune` | `python -m grimoire_ai.llm.training.finetune` |
| `grimoire-preprocess` | `python -m grimoire_ai.llm.data.preprocessing` |

Standalone scripts in `scripts/` (corpus scrapers, `evaluate.py`, `export_gguf.py`, `build_saga_corpus.py`, …) are invoked directly with `python scripts/<name>.py` — they live outside the installed package and are not registered as console scripts.

## Deployment

### Docker

One image, both apps — `docker-compose.yml` runs them side by side:

```bash
docker compose up --build
# training/eval: http://localhost:7860
# chat:          http://localhost:7861
```

Or run just one directly with `docker run`, overriding `CMD` for chat:

```bash
docker build -t grimoire-ai .
docker run --rm -p 7860:7860 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/agents.json:/app/agents.json" \
    grimoire-ai
# open http://localhost:7860

docker run --rm -p 7861:7861 \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/checkpoints:/app/checkpoints" \
    -v "$(pwd)/agents.json:/app/agents.json" \
    grimoire-ai grimoire-chat-ui
# open http://localhost:7861
```

The image is CPU-only (`python:3.11-slim` base, CPU build of torch), bound to `0.0.0.0` instead of localhost so it's reachable from outside the container. Mount `data/`, `checkpoints/`, and `agents.json` from the host so corpora, checkpoints, and agent configs persist across container restarts and survive image rebuilds. For CUDA, see the comments at the top of the `Dockerfile`.

### GGUF export (llama.cpp)

```bash
python scripts/export_gguf.py \
    --checkpoint checkpoints/pretrain/step_0010000.pt \
    --output     models/grimoire-f16.gguf \
    --vocab      data/tokenizer/bpe.json

# Optional: quantize to 4-bit
llama-quantize models/grimoire-f16.gguf models/grimoire-q4km.gguf Q4_K_M

# Run with llama.cpp — Grimoire's architecture (GQA, RoPE, SwiGLU, RMSNorm)
# matches the "llama" GGUF architecture exactly, so no custom llama.cpp
# build is required.
llama-cli -m models/grimoire-q4km.gguf -p "You are a D&D assistant." -n 256
```

Only GQA checkpoints export today — a checkpoint trained with `attention_type="mla"` raises `NotImplementedError`, since llama.cpp's tensor/metadata conventions for that architecture couldn't be verified without a real binary to test an export against.

## Development

```bash
pip install -e ".[dev]"
pytest                            # unit tests
pytest tests/test_integration.py  # end-to-end integration tests
```

See [docs/setup-training.md](docs/setup-training.md) and [docs/setup-inference.md](docs/setup-inference.md) for detailed setup guides, and [docs/useful_commands.md](docs/useful_commands.md) for check/verify/loop commands that don't belong to a single pipeline stage (corpus token counts, multi-seed comparison loops, quality-filter previews, and the like).

## References

- Granville, V. (2026). *LLMs Without Deep Neural Networks — New Architecture, Benefits & Case Study*. BondingAI.io.
- Granville, V. (2026). *No-Blackbox, Secure, Efficient AI and XLLM Solutions*. MLT.
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv.
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv.
- Ainslie, J. et al. (2023). *GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.

## License
This repository is not licensed for use, modification, or distribution.
All rights reserved.
