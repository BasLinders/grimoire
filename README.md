# Grimoire

**Grimoire** is a hybrid small language model (SLM) engine built from scratch, designed to run on a home machine with GPU acceleration. It pairs a corpus retrieval engine (based on dr. Vincent Granville's Radial Basis Function interpolator) with a scratch-built transformer LLM for coherent, context-aware response generation. The result is an agent engine that is accurate, explainable, and grounded in a defined corpus — no hallucination on domain facts.

Grimoire is intentionally domain-agnostic. Agents are configurations of the engine: each agent declares which corpora it uses, which tools it has access to, and its conversational persona. The engine stays the same.

The first agent built on Grimoire is **Saga**: a domain-specialised chatbot covering Dungeons & Dragons, mathematics, and data science, with Google Calendar integration.

## Goals

- Run on consumer hardware (CPU or CUDA GPU — no cloud required)
- Zero *domain* training cost: corpus retrieval handles facts; the LLM handles language and conversation flow
- Full conversational coherence with multi-turn context tracking
- Modular engine: swap corpora, agents, and tool integrations independently
- Deterministic, explainable retrieval grounded in a defined corpus — no hallucination on domain facts
- Built from scratch as a learning project: tokenizer, transformer architecture, training loop, and fine-tuning pipeline are all hand-written

## Repository Structure

```
grimoire/
├── grimoire/
│   ├── corpus/             # Corpus ingestion, stemming, multi-token indexing  ✓
│   ├── llm/                # Scratch-built transformer LLM                     ✓
│   │   ├── tokenizer/      # Byte-level BPE tokenizer (16 384 vocab)
│   │   ├── model/          # Decoder-only transformer (GQA, RoPE, SwiGLU, RMSNorm)
│   │   ├── data/           # TokenizedDataset, PaddingCollator, ConversationDataset
│   │   ├── training/       # Trainer, checkpointing, pretrain + finetune entry points
│   │   └── inference/      # PromptBuilder, KV-cache sampler, InferenceEngine
│   ├── state/              # Conversation state and rolling multi-turn history  ✓
│   ├── cli/                # Interactive terminal chat loop                     ✓
│   ├── ui/                 # Gradio training/fine-tuning/chat UI                ✓
│   ├── router/             # Intent detection and tool routing                  planned
│   └── tools/              # Tool integrations (Google Calendar, etc.)          planned
├── agents/
│   └── saga/               # Saga agent — D&D, math, data science, calendar    planned
├── data/                   # Corpus data files (gitignored)
├── docs/                   # Setup guides (training, inference)
└── tests/                  # 144 tests — all green
```

## Architecture

### Flow

```mermaid
flowchart TD
    A([User message]) --> B[Conversation State Manager]

    B --> C{Intent Router}

    C -->|knowledge query| D[Grimoire Corpus Engine]
    C -->|calendar intent| E[Google Calendar API]

    D --> F[(Corpus Index\nnested hash map)]
    F -->|top-k passages + scores| G

    B -->|rolling history| G[GrimoireTransformer\nPromptBuilder → Sampler]

    G --> H([Coherent response])
    E --> H
```

### Component Roles

| Component | Role | Status |
|---|---|---|
| **Corpus Engine** | Ingests text, indexes stemmed 4-gram multi-tokens in a hash map, retrieves top-k passages by Jaccard similarity | ✓ done |
| **BPE Tokenizer** | Byte-level Byte-Pair Encoding; vocab size 16 384; lossless round-trip for any Unicode input | ✓ done |
| **GrimoireTransformer** | Scratch-built decoder-only transformer (~25 M params); GQA, RoPE, SwiGLU, RMSNorm, weight-tied output head | ✓ done |
| **Training Pipeline** | AdamW + cosine-warmup LR, fp16 AMP, gradient accumulation, checkpointing; `on_log` callback for live UI streaming | ✓ done |
| **Inference Engine** | PromptBuilder (corpus → prompt), KV-cache autoregressive sampler (temperature / top-k / top-p / repetition penalty), `respond()` and `chat()` API | ✓ done |
| **KV-Cache** | Caches K/V projections so each generation step costs O(1) instead of O(n²); sliding-window truncation at `max_seq_len` | ✓ done |
| **Instruction Fine-tuning** | Second training pass on `{user, assistant, context?}` JSONL examples; response-only loss masking so only answer tokens contribute to the loss | ✓ done |
| **Training UI** | Gradio web app: Pre-train tab, Fine-tune tab, and multi-turn Chat tab with live loss streaming and conversation history | ✓ done |
| **Conversation State** | `ConversationState` maintains a rolling turn history; `build_prompt_ids()` packs history newest-first within the token budget, then fills remaining space with corpus context | ✓ done |
| **Corpus Scraper** | Ingestion utility for web URLs, PDFs, DOCX, Markdown, and images (OCR) — converts sources to corpus `.txt` files | phase 2.5 |
| **Integration Tests** | End-to-end test suite covering the full pipeline: pre-train → fine-tune → load → corpus attach → multi-turn `chat()`. Catches wiring bugs that unit tests miss. | phase 7.5 |
| **Intent Router** | Routes calendar intents to Google Calendar API; all knowledge queries to the corpus engine | planned |
| **Tool Integrations** | Google Calendar read/write | planned |

### Multi-turn prompt format

```
<BOS> [<SEP> {corpus context} <SEP>] <USR> q1 <AST> a1
                                     <USR> q2 <AST> a2
                                     …
                                     <USR> current query <AST>
```

The first turn of every session is identical to the single-turn fine-tuning format — no distribution shift from the training data.

### Why Hybrid

The corpus engine and the LLM cover each other's weaknesses:

- **LLM alone** — fluent and coherent, but hallucinates domain facts and requires expensive fine-tuning to specialise
- **Corpus alone** — accurate and deterministic, but cannot track conversational context or generate natural sentences
- **Together** — the corpus retrieves grounded passages; the LLM reads them and produces a coherent, context-aware response. No domain-specific fine-tuning required — new knowledge comes from adding to the corpus, not retraining.

### LLM Architecture

The transformer uses four improvements over the GPT-2 baseline, all from openly published research:

| Technique | Replaces | Benefit |
|---|---|---|
| **RMSNorm** | LayerNorm | Removes mean-centering; marginally faster, equally stable (Zhang & Sennrich, 2019) |
| **RoPE** | Learned positional embeddings | Encodes relative position via rotation; zero extra parameters (Su et al., 2021) |
| **SwiGLU** | GELU feed-forward | Gated activation; consistently outperforms GELU at the same parameter budget (Shazeer, 2020) |
| **Grouped Query Attention** | Standard multi-head attention | `n_kv_heads=2` vs `n_heads=8`; 4× smaller KV cache at inference (Ainslie et al., 2023) |

Default configuration: `vocab_size=16384`, `d_model=512`, `n_layers=6`, `n_heads=8`, `n_kv_heads=2`, `d_ff=1408`, `max_seq_len=1024` → ~25 M parameters, ~100 MB fp32.

## Development Roadmap

| Phase | Scope | Status |
|---|---|---|
| **1** | BPE tokenizer (byte-level, 16 384 vocab, special tokens) | ✓ done |
| **2** | Corpus retrieval engine (stemmer, n-gram index, Jaccard scoring) | ✓ done |
| **3** | Transformer architecture (GQA, RoPE, SwiGLU, RMSNorm) + training pipeline | ✓ done |
| **4** | Inference pipeline: PromptBuilder, sampler (temperature/top-k/top-p), InferenceEngine | ✓ done |
| **5** | KV-cache (O(n²) → O(1) per generation step) + richer corpus context (unstemmed excerpts) | ✓ done |
| **6** | Instruction fine-tuning: `ConversationDataset`, response-only loss masking, `finetune.py` entry point | ✓ done |
| **6.5** | Training UI: Gradio app (Pre-train, Fine-tune, Chat tabs) with live loss streaming via `Trainer.on_log` callback | ✓ done |
| **7** | Conversation state manager: `ConversationState`, `InferenceEngine.chat()`, terminal chat loop | ✓ done |
| **2.5** | Corpus scraper: web URLs, PDF, DOCX, Markdown, Images (OCR) → corpus `.txt` files | next |
| **7.5** | Integration test suite: full pipeline (train → checkpoint → load → corpus → multi-turn chat) in a single fast test run | planned |
| **8** | Intent router + tool integrations (Google Calendar) | planned |
| **9** | Saga agent: D&D / math / data science corpus + calendar assistant | planned |

### Why two training phases?

Pre-training on raw text teaches the model statistics of language — grammar, facts, style. But it gives the model no reason to *respond* to a question rather than continue the question. Instruction fine-tuning is a lightweight second pass on a small dataset of `(query, response)` pairs formatted in the prompt template the model will see at inference. After fine-tuning the model reliably produces a response after `<AST>` instead of continuing the user's sentence.

This is also why adding domain knowledge to Grimoire does **not** require retraining: the corpus handles domain facts at inference time by injecting retrieved passages as prompt context. Fine-tuning only needs to happen once to teach conversation behaviour; it is not repeated when corpora are updated.

## Agents

### Saga *(planned)*

Saga is the first Grimoire-based agent. It operates as both a domain chatbot and a personal assistant:

- **Knowledge domains**: Dungeons & Dragons rules and lore, mathematics, data science
- **Assistant features**: Google Calendar integration — reads appointments to answer scheduling questions
- **Design principle**: corpus-grounded answers only. Saga will say it does not know rather than guess.

## Usage

### Preprocessing your corpus

```bash
python -m grimoire.llm.data.preprocessing \
    --input  data/raw/ \
    --output data/processed/corpus.bin \
    --vocab  data/tokenizer/bpe.json
```

This tokenizes all `.txt` files under `data/raw/`, trains the BPE vocabulary if it does not yet exist, and writes a memory-mapped binary corpus ready for training.

### Training

```bash
python -m grimoire.llm.training.train
```

Or resuming from a checkpoint:

```bash
python -m grimoire.llm.training.train --resume checkpoints/step_0001000.pt
```

**GPU setup (Windows, RTX, CUDA 12.4):**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
```

The trainer auto-detects CUDA and enables fp16 AMP when available.

### Training UI

```bash
pip install -e ".[ui]"
python -m grimoire.ui
# open http://localhost:7860
```

Three tabs: **Pre-train** (corpus → checkpoint), **Fine-tune** (checkpoint + JSONL → fine-tuned checkpoint), **Chat** (multi-turn conversation with any checkpoint).

### Fine-tuning

```bash
python -m grimoire.llm.training.finetune \
    --resume  checkpoints/step_0010000.pt \
    --data    data/finetune/examples.jsonl \
    --vocab   data/tokenizer/bpe.json \
    --output  checkpoints/finetune/
```

See [docs/setup-training.md](docs/setup-training.md) for the full JSONL format and all flags.

### Multi-turn chat (terminal)

```bash
python -m grimoire.cli.chat \
    --checkpoint checkpoints/finetune/step_0000500.pt \
    --vocab      data/tokenizer/bpe.json \
    --corpus-dir data/corpus/
```

Commands during chat: `/clear` (reset history), `/history` (review turns), `/quit`.

### Single-turn inference (Python API)

```python
from grimoire.corpus.corpus import GrimoireCorpus
from grimoire.llm.inference.engine import InferenceEngine
from grimoire.llm.inference.sampler import GenerationConfig

corpus = GrimoireCorpus()
corpus.add_text("A grappled creature has its speed reduced to zero.", source="dnd_srd")

engine = InferenceEngine(
    checkpoint_path="checkpoints/step_0005000.pt",
    tokenizer_path="data/tokenizer/bpe.json",
    corpus=corpus,
    gen_config=GenerationConfig(temperature=0.8, top_p=0.9, top_k=50),
)
print(engine.respond("What happens when a creature is grappled?"))
```

### Multi-turn chat (Python API)

```python
from grimoire.llm.inference.engine import InferenceEngine
from grimoire.state.conversation import ConversationState

engine = InferenceEngine(
    checkpoint_path="checkpoints/finetune/step_0000500.pt",
    tokenizer_path="data/tokenizer/bpe.json",
)
state = ConversationState()

response1 = engine.chat("What is grapple?", state)
response2 = engine.chat("And how do I escape it?", state)  # model sees prior turn
```

The engine auto-detects CUDA and falls back to CPU when no GPU is available.

## Development

```bash
pip install -e ".[dev]"
pytest
```

All 144 tests pass across corpus, tokenizer, model, data pipeline, training, inference, conversation state, and fine-tuning modules.

See [docs/setup-training.md](docs/setup-training.md) and [docs/setup-inference.md](docs/setup-inference.md) for detailed setup guides.

## References

- Granville, V. (2026). *LLMs Without Deep Neural Networks — New Architecture, Benefits & Case Study*. BondingAI.io.
- Granville, V. (2026). *No-Blackbox, Secure, Efficient AI and XLLM Solutions*. MLT.
- Zhang, B. & Sennrich, R. (2019). *Root Mean Square Layer Normalization*. NeurIPS.
- Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv.
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. arXiv.
- Ainslie, J. et al. (2023). *GQA: Training Generalised Multi-Query Transformer Models from Multi-Head Checkpoints*. EMNLP.
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR.
