# Grimoire

**Grimoire** is a hybrid small language model (SLM) engine built entirely from scratch, designed to run on consumer hardware with optional GPU acceleration. It pairs a corpus retrieval engine with a scratch-built transformer LLM — the retrieval engine provides grounded domain facts, the LLM provides coherent language and conversational flow.

Grimoire is intentionally domain-agnostic. Agents are named configurations: each agent declares its corpus, its checkpoint, and its generation defaults. The engine stays the same.

The first agent built on Grimoire is **Saga**: a focused domain chatbot covering Dungeons & Dragons rules, encounter mathematics, and probability / data science.

## Goals

- Run on consumer hardware (CPU or CUDA GPU — no cloud required)
- Zero domain training cost: corpus retrieval handles facts; the LLM handles language and conversation
- Full conversational coherence with multi-turn context tracking
- Modular: swap corpora, agents, and checkpoints independently
- Deterministic, explainable retrieval — no hallucination on domain facts
- Built from scratch as a learning project: tokenizer, transformer, training loop, and fine-tuning pipeline are all hand-written

## Repository Structure

```
grimoire/
├── grimoire/
│   ├── agents/             # Agent registry — loads agents.json, builds InferenceEngine  ✓
│   ├── corpus/             # Corpus ingestion, stemming, n-gram index, Jaccard retrieval  ✓
│   ├── llm/                # Scratch-built transformer LLM                                ✓
│   │   ├── tokenizer/      # Byte-level BPE tokenizer (vocab size 16 384)
│   │   ├── model/          # Decoder-only transformer (GQA, RoPE, SwiGLU, RMSNorm)
│   │   ├── data/           # TokenizedDataset, PaddingCollator, ConversationDataset
│   │   ├── training/       # Trainer, checkpointing, pretrain + finetune entry points
│   │   └── inference/      # PromptBuilder, KV-cache sampler, InferenceEngine
│   ├── state/              # ConversationState — rolling multi-turn history + prompt build  ✓
│   ├── cli/                # Interactive terminal chat loop                                ✓
│   └── ui/                 # Gradio app — Pre-train, Fine-tune, Ingest, Chat tabs          ✓
├── agents.json             # Named agent configurations (checkpoint, vocab, corpus, gen defaults)
├── scripts/
│   ├── build_saga_corpus.py        # Download D&D 5e SRD + copy math references
│   ├── finetune_saga.py            # Fine-tune a checkpoint on the Saga dataset
│   ├── validate_finetune_data.py   # Pre-flight check on any JSONL dataset
│   ├── finetune_data/
│   │   └── saga_v1.jsonl           # 30 Q&A examples (D&D rules, encounter math, probability)
│   └── saga_references/            # Hand-authored math/probability reference .txt files
├── docs/                   # Setup guides (training, inference)
├── data/                   # Runtime data — gitignored (corpus bins, tokenizer, checkpoints)
│   ├── raw/                # Source .txt files for pre-training corpus
│   ├── processed/          # Tokenised corpus.bin
│   ├── tokenizer/          # bpe.json — trained BPE vocabulary
│   ├── corpus/
│   │   └── saga/           # Populated by scripts/build_saga_corpus.py
│   └── finetune/           # Any additional fine-tuning datasets
├── checkpoints/            # Saved model checkpoints — gitignored
└── tests/                  # 227 unit tests + 15 integration tests — all green
```

## Architecture

### Flow

```mermaid
flowchart TD
    A([User message]) --> B[ConversationState\nrolling turn history]

    B --> C[GrimoireCorpus\nJaccard retrieval]
    C --> D[(Corpus Index\nn-gram hash map)]
    D -->|top-k excerpts| E

    B -->|packed prompt ids| E[GrimoireTransformer\nPromptBuilder → KV-cache sampler]

    E --> F([Response])
    F --> B
```

### Component Table

| Component | Role | Status |
|---|---|---|
| **BPE Tokenizer** | Byte-level Byte-Pair Encoding; vocab 16 384; lossless round-trip for any Unicode | ✓ done |
| **Corpus Engine** | Ingests text, indexes stemmed 4-gram multi-tokens, retrieves top-k passages by Jaccard similarity with unstemmed excerpts | ✓ done |
| **Corpus Scraper** | `ingest()` dispatcher for web URLs (HTML + Markdown), PDFs, DOCX, Markdown files, plain text, and images (OCR) | ✓ done |
| **GrimoireTransformer** | Scratch-built decoder-only transformer (~25 M params); GQA, RoPE, SwiGLU, RMSNorm, weight-tied output head | ✓ done |
| **Training Pipeline** | AdamW + cosine-warmup LR, fp16 AMP, gradient accumulation, checkpointing; `on_log` callback for live loss streaming | ✓ done |
| **Instruction Fine-tuning** | `ConversationDataset` on `{user, assistant, context?}` JSONL; response-only loss masking | ✓ done |
| **Inference Engine** | PromptBuilder (corpus → prompt), KV-cache autoregressive sampler (temperature / top-k / top-p / repetition penalty), `respond()`, `chat()`, and `chat_stream()` (token-by-token generator) | ✓ done |
| **KV-Cache** | Caches K/V projections: O(n²) → O(1) per generation step; sliding-window truncation at `max_seq_len` | ✓ done |
| **Conversation State** | `ConversationState` packs rolling history newest-first within the token budget, then fills remaining space with corpus context | ✓ done |
| **Training UI** | Gradio app: Pre-train, Fine-tune, Ingest, and Chat tabs with live loss streaming | ✓ done |
| **Agent Registry** | `AgentRegistry` reads `agents.json`; `build_engine(key)` returns a ready `InferenceEngine` with corpus auto-loaded | ✓ done |
| **Saga Corpus** | 24 D&D 5e SRD sections + 4 hand-authored math/probability reference files; built by `scripts/build_saga_corpus.py` | ✓ done |
| **Saga Fine-tune Dataset** | 30 Q&A examples (D&D rules, encounter math, probability) in `scripts/finetune_data/saga_v1.jsonl` | ✓ done |
| **Integration Tests** | End-to-end: BPE train → corpus build → model pretrain → checkpoint → engine load → multi-turn `chat()` | ✓ done |

### Multi-turn prompt format

```
<BOS> [<SEP> {corpus context} <SEP>] <USR> q1 <AST> a1
                                     <USR> q2 <AST> a2
                                     …
                                     <USR> current query <AST>
```

History is packed newest-first within the token budget. The corpus context fills whatever space remains after history. If the budget is exhausted the oldest turns are dropped first, then the context is trimmed. The current query is always preserved.

### Why Hybrid

| Approach | Strength | Weakness |
|---|---|---|
| LLM alone | Fluent, coherent, conversational | Hallucinates domain facts; expensive to specialise |
| Corpus alone | Accurate, deterministic, explainable | Cannot track context or produce natural sentences |
| **Grimoire (hybrid)** | Grounded facts from corpus + coherent language from LLM | Slightly more complex setup |

New domain knowledge requires only adding `.txt` files to the corpus and calling `corpus.add_text()` — no retraining.

### LLM Architecture

The transformer uses four improvements over the GPT-2 baseline:

| Technique | Replaces | Benefit |
|---|---|---|
| **RMSNorm** | LayerNorm | Removes mean-centering; marginally faster, equally stable |
| **RoPE** | Learned positional embeddings | Encodes relative position via rotation; zero extra parameters |
| **SwiGLU** | GELU feed-forward | Gated activation; consistently outperforms GELU at equal parameter budget |
| **Grouped Query Attention** | Multi-head attention | `n_kv_heads=2` vs `n_heads=8`; 4× smaller KV cache at inference |

Default configuration: `vocab_size=16384`, `d_model=512`, `n_layers=6`, `n_heads=8`, `n_kv_heads=2`, `d_ff=1408`, `max_seq_len=1024` → ~25 M parameters, ~100 MB fp32.

## Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | BPE tokenizer (byte-level, 16 384 vocab, 6 special tokens) | ✓ done |
| **2** | Corpus retrieval engine (stemmer, n-gram index, Jaccard scoring, unstemmed excerpts) | ✓ done |
| **3** | Transformer architecture (GQA, RoPE, SwiGLU, RMSNorm) + training pipeline | ✓ done |
| **4** | Inference pipeline: PromptBuilder, sampler (temperature/top-k/top-p/repetition), InferenceEngine | ✓ done |
| **5** | KV-cache (O(n²) → O(1) per step) + richer corpus context (unstemmed excerpts) | ✓ done |
| **6** | Instruction fine-tuning: `ConversationDataset`, response-only loss masking, `finetune.py` | ✓ done |
| **6.5** | Training UI: Gradio app (Pre-train, Fine-tune, Chat tabs) + `Trainer.on_log` live streaming | ✓ done |
| **7** | Conversation state: `ConversationState`, `InferenceEngine.chat()`, terminal chat CLI | ✓ done |
| **2.5** | Corpus scraper: web URLs (HTML + Markdown), PDF, DOCX, Markdown, images (OCR) | ✓ done |
| **7.5** | Integration test suite: full pipeline from BPE training to multi-turn `chat()` | ✓ done |
| **8** | Agent registry (`agents.json`) + agent selector dropdown in Chat UI | ✓ done |
| **9** | Saga corpus: D&D 5e SRD (24 sections) + encounter math + probability references | ✓ done |
| **10** | Saga instruction fine-tuning: 30-example JSONL dataset + fine-tune + validation scripts | ✓ done |

### Why two training phases?

Pre-training on raw text teaches the model language statistics — grammar, facts, style. It gives the model no reason to *respond* to a question rather than continue it.

Instruction fine-tuning is a short second pass on `{user, assistant}` pairs. After fine-tuning the model reliably produces a response after `<AST>` instead of continuing the user's sentence.

Domain knowledge lives in the corpus, not the model weights. Fine-tuning teaches *conversation behaviour*; it does not need to be repeated when corpora are updated.

## Agents

### Saga

Saga is the first Grimoire agent. It focuses on:

- **D&D 5e rules**: conditions, classes, spells, monsters, combat mechanics — sourced from the CC-BY 4.0 Systems Reference Document
- **Encounter mathematics**: XP budgets, CR scaling, multipliers, action economy
- **Probability & statistics**: dice distributions, DPR calculations, hit probabilities, death save odds

**Setup:**

```bash
# 1. Build the Saga corpus (downloads D&D 5e SRD ~1.5 MB, takes ~30 s)
python scripts/build_saga_corpus.py

# 2. Pre-train a model (or use an existing checkpoint)
python -m grimoire_ai.llm.training.train

# 3. Validate the fine-tuning dataset
python scripts/validate_finetune_data.py \
    --data  scripts/finetune_data/saga_v1.jsonl \
    --vocab data/tokenizer/bpe.json

# 4. Fine-tune on the Saga dataset
python scripts/finetune_saga.py \
    --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \
    --vocab      data/tokenizer/bpe.json

# 5. Update agents.json with the fine-tuned checkpoint path, then load
#    Saga from the Chat tab dropdown in the UI.
python -m grimoire_ai.ui
```

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
python -m grimoire_ai.cli.chat \
    --checkpoint checkpoints/finetune/step_0000500.pt \
    --vocab      data/tokenizer/bpe.json \
    --corpus-dir data/corpus/saga/
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

### Training UI

```bash
pip install -e ".[ui]"
python -m grimoire_ai.ui
# open http://localhost:7860
```

Six tabs: **Preprocess**, **Pre-train** (with model size presets), **Fine-tune**, **Ingest** (multi-file upload), **Chat** (streaming responses + dataset builder), **Scale** (Chinchilla scaling calculator).

## Development

```bash
pip install -e ".[dev]"
pytest                            # 227 unit tests + 2 skipped (PDF container issue)
pytest tests/test_integration.py  # 15 end-to-end integration tests
```

See [docs/setup-training.md](docs/setup-training.md) and [docs/setup-inference.md](docs/setup-inference.md) for detailed setup guides.

## References

- Granville, V. (2026). *LLMs Without Deep Neural Networks — New Architecture, Benefits & Case Study*. BondingAI.io.
- Granville, V. (2026). *No-Blackbox, Secure, Efficient AI and XLLM Solutions*. MLT.
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv.
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv.
- Ainslie, J. et al. (2023). *GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
