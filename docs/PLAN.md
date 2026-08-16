# Grimoire Roadmap

---

## Phase 1 — Foundation (Complete)

| Step | Scope | Status |
|---|---|---|
| **1** | BPE tokenizer (byte-level, 16 384 vocab, 6 special tokens) | ✓ done |
| **2** | Corpus retrieval engine (stemmer, n-gram index, Jaccard scoring, unstemmed excerpts) | ✓ done |
| **2.5** | Corpus scraper: web URLs (HTML + Markdown), PDF, DOCX, Markdown, images (OCR) | ✓ done |
| **3** | Transformer architecture (GQA, RoPE, SwiGLU, RMSNorm) + training pipeline | ✓ done |
| **4** | Inference pipeline: PromptBuilder, sampler (temperature/top-k/top-p/repetition), InferenceEngine | ✓ done |
| **5** | KV-cache (O(n²) → O(1) per step) + richer corpus context (unstemmed excerpts) | ✓ done |
| **6** | Instruction fine-tuning: `ConversationDataset`, response-only loss masking, `finetune.py` | ✓ done |
| **6.5** | Training UI: Gradio app (Pre-train, Fine-tune, Chat tabs at the time; later split into separate training/eval and chat apps — see README.md) + `Trainer.on_log` live streaming | ✓ done |
| **7** | Conversation state: `ConversationState`, `InferenceEngine.chat()`, terminal chat CLI | ✓ done |
| **7.5** | Integration test suite: full pipeline from BPE training to multi-turn `chat()` | ✓ done |
| **8** | Agent registry (`agents.json`) + agent selector dropdown in Chat UI | ✓ done |
| **9** | Saga corpus: D&D 5e SRD (24 sections) + encounter math + probability references — this is the *initial seed*; see "Corpus & Data Quality" below for what it grew into | ✓ done |
| **10** | Saga instruction fine-tuning: 30-example JSONL dataset + fine-tune + validation scripts — likewise superseded as the primary fine-tune data source; see [expansion_PLAN.md](expansion_PLAN.md) | ✓ done |
| **11** | Training optimisations: Flash Attention (SDPA), `torch.compile`, `cudnn.benchmark`, non-blocking transfers | ✓ done |
| **12** | Semantic retrieval: `GrimoireTransformer.embed()`, `SemanticRetriever` (cosine over model embeddings), `InferenceEngine.build_semantic_corpus()` | ✓ done |
| **13** | Retrieval router: score-threshold gate in `InferenceEngine._retrieve()` — routes per-query to grounded or pure-chat generation | ✓ done |

### Why two training phases?

Pre-training on raw text teaches the model language statistics — grammar, facts, style. It gives the model no reason to *respond* to a question rather than continue it.

Instruction fine-tuning is a short second pass on `{user, assistant}` pairs. After fine-tuning the model reliably produces a response after `<AST>` instead of continuing the user's sentence.

Domain knowledge lives in the corpus, not the model weights. Fine-tuning teaches *conversation behaviour*; it does not need to be repeated when corpora are updated.

---

## Phase 2 — Expansion

### Context

Grimoire's foundation is solid. Phase 2 makes it meaningfully more capable and deployable: better memory efficiency, evaluation infrastructure, tool-calling, adapter-based fine-tuning, automatic agent routing, persistent retrieval, and portable export.

Current strengths: KV-cache, mixed-precision (AMP), Flash SDPA, torch.compile, hybrid lexical+semantic RAG, multi-turn conversation state, BPE tokenizer, full pre-train → fine-tune pipeline, agent registry, int8 quantization, gradient checkpointing.

Current gaps: no LoRA, no evaluation harness, no tool-calling, recomputed RAG index per session, no export format beyond `.pt`.

### Items

### 1. int8 Quantization (Memory — CPU viability) ✓ done

Dynamic int8 quantization via `torch.quantization.quantize_dynamic` / `torchao`. Zero-shot, cuts model size ~4×, speeds up CPU inference. Tries `torchao` first; falls back to legacy API with deprecation warning suppressed. `InferenceEngine(quantize=True)` + checkbox in the chat UI.

### 2. Gradient Checkpointing (Memory — Training) ✓ done

`GrimoireTransformer.enable_gradient_checkpointing()` wraps each `TransformerBlock` with `torch.utils.checkpoint.checkpoint(use_reentrant=False)`. Halves peak VRAM at ~20% speed cost. `Trainer(gradient_checkpointing=True)`, `--gradient-checkpointing` CLI flag, checkbox in Pre-train tab.

### 3. Evaluation Harness ✓ done

**Why third:** Without domain-specific evals, there is no feedback signal for any other roadmap item. Every downstream decision — does quantization hurt quality? did fine-tuning improve accuracy? — requires this first.

- **Perplexity eval:** bits-per-character on a held-out corpus slice (`grimoire_ai/llm/eval/perplexity.py`)
- **Retrieval hit-rate:** keyword-in-top-1 over a fixed 20-query Saga set (`grimoire_ai/llm/eval/retrieval.py`)
- **D&D rules quiz:** 20 Q&A pairs with keyword-recall + token-F1 scoring (`grimoire_ai/llm/eval/quiz.py`, `scripts/eval_data/saga_quiz.jsonl`)
- **Harness:** orchestrates all three; writes timestamped JSON to `data/eval/` (`grimoire_ai/llm/eval/harness.py`)
- **CLI:** `python scripts/evaluate.py --checkpoint ... --vocab ...` (perplexity, retrieval, quiz flags)
- **UI:** Evaluate tab (after Scale) — checkpoint/vocab/corpus/quiz inputs, Run button, live log stream

### 4. Math → Python CLI (Tool Calling) ✓ done

**Why fourth:** The D&D fine-tuning data originally trained the model to decline arithmetic and defer to a tool that didn't exist yet. This closes that loop — `saga_v1.jsonl` and `saga_dnd_math.jsonl` were since rewritten to use `<TOOL:python>` tags instead of declining.

- **Query-side detection:** `MathTool.detect()` identifies arithmetic in queries via regex (operators, percent-of, Unicode multiply/power); dice notation excluded (`grimoire_ai/tools/math_tool.py`)
- **Safe evaluator:** pure `ast`-based visitor — no `eval()`, no subprocess needed; supports arithmetic operators, parentheses, and whitelisted math functions (sqrt, sin, cos, log, …)
- **Context injection:** result prepended as a synthetic `QueryResult` with excerpt `[Math] expr = result`; flows through the existing `PromptBuilder` context slot
- **Response-side tag resolution:** `MathTool.process_response()` replaces `<TOOL:python>…</TOOL>` tags emitted by fine-tuned models with evaluated results
- **Fine-tune data:** 15 examples with `<TOOL:python>` format in `scripts/finetune_data/tool_call_examples.jsonl`
- **CLI:** `--math-tool` flag on `python -m grimoire_ai.cli.chat`
- **UI:** "Enable math tool" checkbox in the chat UI (wired to both agent load and manual load)

### 5. LoRA / Adapter Fine-Tuning ✓ done

**Why fifth:** Full fine-tuning on 29–36 examples risks catastrophic forgetting. LoRA freezes base weights and trains ~0.5% of parameters — better regularization, faster, and each domain persona becomes a 2–5 MB `.lora` file rather than a full checkpoint.

- `LoRALinear` wrapper: rank-decomposition `A × B` path fused into frozen `q_proj`/`v_proj` via registered buffer `base_weight`; `apply_lora()` patches the transformer in-place, `merge_and_unload()` absorbs adapter into weights for export (`grimoire_ai/llm/model/lora.py`)
- `save_lora()` iterates `named_modules()` directly (never clones frozen `base_weight` buffers — avoids OOM); `load_lora()` matches by module name; both validated by 29-test suite
- `--lora-rank` / `--lora-alpha` args on `finetune.py`; `InferenceEngine.load_lora()` hot-swaps adapter without reloading base weights
- Fine-tune UI: **Mode dropdown** (Base instruction fine-tune vs Agent LoRA adapter) — switches defaults, shows/hides agent name field, clears stale resume path; saves named `<agent>.lora` file on completion
- 64 general-conversation pairs (`scripts/finetune_data/general_conversations.jsonl`) for base instruction fine-tuning

### 6. Agent Routing ✓ done

**Why sixth:** `AgentRegistry` already loads multiple agents. What's missing is automatic dispatch.

- `AgentRouter` scores a query against each agent's `GrimoireCorpus` (top-1 Jaccard) and routes to the highest-scoring agent when it strictly exceeds the threshold; falls back to `default_key` otherwise (`grimoire_ai/agents/router.py`)
- `MultiAgentEngine` wraps one shared `InferenceEngine`; `_switch_to()` hot-swaps the active LoRA adapter and corpus per turn — sub-second vs minutes if reloading the full model; `top_k_corpus` forwarded so corpus retrieval depth is configurable in the multi-agent path
- `_RoutingStateWrapper` proxy intercepts `add_turn()` to inject `agent_key` / `routing_score`; `__setattr__` delegates public-name writes to the wrapped `ConversationState` so no state mutation is silently dropped; `add_turn` signature accepts `**_` to tolerate future keyword arguments without crashing
- `InferenceEngine.unload_lora()` public method consolidates the merge+reload sequence; `checkpoint_path` property exposes the path without callers accessing `_checkpoint_path` directly
- `ConversationState.routing_log` property returns `[(turn_index, agent_key, score), …]` for all routed turns; unrouted turns excluded
- `AgentRegistry._scan_corpus_dirs()` helper eliminates the duplicate corpus-scan block from `build_engine` and `build_router`; `build_multi_agent_engine()` now calls `engine.unload_lora()` instead of duplicating the merge+reload sequence
- UI: **Auto-route** option prepended to agent selector; **Routing threshold** slider revealed on Auto-route selection and passed to `build_multi_agent_engine(threshold=)`; routing label in "Routed to" textbox preserved mid-stream via `gr.update()` no-op yields (was cleared to `""` on every token)
- 18-test suite covering `AgentRouter`, `_RoutingStateWrapper`, `ConversationState.routing_log`, and `MultiAgentEngine` (`tests/agents/test_router.py`)

Saga is the only agent actually built on this infrastructure so far — see [tools.md](tools.md) for the planned `tools/` directory of developer utilities meant to be reused across agents rather than re-written per agent.

### 7. Persistent RAG Index ✓ done

**Why seventh:** Semantic embeddings are recomputed from scratch every session. As corpus grows toward 500M tokens this becomes a minutes-long blocking startup.

- Pre-compute chunk embeddings after preprocessing; persist as a flat numpy memmap alongside `corpus.bin`
- On session start, load the index from disk; recompute only for new/changed chunks (hash-based staleness check)
- Optional FAISS index for approximate nearest-neighbour at scale (replaces brute-force cosine)
- Expose "Rebuild index" button in the Corpus tab

### 8. GGUF Export ✓ done

**Why last:** The long-term deployment story — no Python dependency, 4-bit quantization, widest hardware support via llama.cpp.

- `grimoire_ai/llm/export/gguf_writer.py`: `GGUFWriter` class — GGUF v3 binary format (header, KV metadata, tensor info, tensor data); `grimoire_to_gguf_name()` maps all GrimoireTransformer state_dict keys to GGUF tensor names
- `scripts/export_gguf.py`: CLI — loads a `.pt` checkpoint, writes F16 or F32 GGUF with architecture metadata and optional BPE tokenizer embedding; Q4_K_M quantization is done post-export via `llama-quantize grimoire-f16.gguf grimoire-q4km.gguf Q4_K_M`
- RoPE buffers (`_cos`, `_sin`) and attention mask are not exported; llama.cpp recomputes them from `rope_theta` and context length
- Weight-tied `output_head.weight` exported once as `output.weight`; 1-D norm weights always stored as F32 for numerical stability
- 42-test suite: name mapping, binary header correctness, dtype selection, alignment, end-to-end export with synthetic checkpoints (`tests/llm/test_export_gguf.py`)
- Deploy: `llama-cli -m grimoire-f16.gguf -p "You are a D&D assistant." -n 256`

---

### Priority Table

| # | Item | Effort | Status | Unblocks |
|---|---|---|---|---|
| 1 | int8 quantization | Low | ✓ done | CPU inference at medium/large scale |
| 2 | Gradient checkpointing | Low | ✓ done | Training medium/large on consumer GPU |
| 3 | Evaluation harness | Medium | ✓ done | All quality-sensitive decisions |
| 4 | Math → Python CLI | Medium | ✓ done | Closing the tool-call loop from fine-tune data |
| 5 | LoRA adapters | Medium | ✓ done | Per-agent specialization without catastrophic forgetting |
| 6 | Agent routing | Low (after 5) | ✓ done | Automatic multi-domain dispatch |
| 7 | Persistent RAG index | Medium | ✓ done | Startup time at 500M token corpus scale |
| 8 | GGUF export | High | ✓ done | Widest deployment, no Python dependency |

---

## Corpus & Data Quality (parallel track, ongoing)

### Context

Fine-tuning experiments converged on a conclusion that shifted priorities: the model's failure modes (context-copying, question-echoing, factual hallucination) look like data-scarcity symptoms, not architecture-too-small symptoms — see [expansion_PLAN.md](expansion_PLAN.md) for the full analysis. This track runs alongside the engineering phases above rather than sequentially after them; it's about corpus scale and quality, not new features.

### Items

- **Gutenberg expansion** (hand-curated, then catalog-based): grew the corpus from a handful of hand-picked public-domain texts to hundreds of files via `scripts/scrape_gutenberg_catalog.py`, which filters Gutenberg's official bulk CSV catalog locally rather than scraping search-result pages (which explicitly warn against it).
- **Stack Exchange RPG Q&A**: `scripts/scrape_stackexchange_rpg.py` downloads the official rpg.stackexchange.com data dump (not live scraping) and pairs each question with its top answers.
- **Near-duplicate dedup**: `scripts/dedup_corpus.py` (MinHash + LSH, word 5-gram shingling) checks new additions against the existing corpus before merging.
- **Derived-adventure pilot**: hand-written D&D adventures derived from public-domain source texts, with all cited monster stats verified against the actual SRD/Open5e stat blocks rather than invented — a deliberately slow, quality-over-volume complement to bulk scraping. Not yet wired into training (`saga_derived/` sits outside `agents.json`'s `corpus_dirs`).
- **Source-based sample weighting**: `--weight-pattern` on `grimoire-preprocess` tags documents by filename glob; `scripts/build_source_weights.py` turns tags into a per-window `sample_weights.npy` consumed by `Trainer`'s `WeightedRandomSampler`. Validated with a paired before/after pretrain comparison — reduced validation loss by ~5.5% relative at no wall-clock cost.
- **Stack Exchange markup cleanup**: `scripts/clean_stackexchange_markup.py` strips vote-score/tag/Markdown-header scaffolding from the SE dump files — this scaffolding was observed bleeding into model generations verbatim (e.g. a literal `## Answer (score: 4)` fragment in output).
- **Validation-split fix**: `_build_datasets`'s `val_split` previously held out the corpus's alphabetically-last files as "the tail," which for this corpus meant validating almost entirely on short Wikipedia/Wikibooks stub articles — not a representative sample. Fixed to hold out many small blocks scattered across the whole corpus instead (`train.py`'s `_split_blocks`).

Full source list, current scale, and open decisions (target model size vs. actual token count, whether to fold `saga_derived/` into training, finer-grained weight tiers) live in [expansion_PLAN.md](expansion_PLAN.md) — that document is updated as the corpus changes; this section is a stable summary of what shipped.

---

## Phase 2.5 — Further Optimization (Planned)

### Context

GGUF export covers one deployment path (llama.cpp), but it's an outlier in scope — a single export format rather than a general optimization. The items below are the remaining levers for compute, payload size, and training efficiency that Phase 2 didn't cover. None are started; this section is a reference for prioritizing future work, not a commitment.

### Items

#### 1. Knowledge Distillation

Train Grimoire against a larger teacher model's logits (or soft targets) instead of, or alongside, raw corpus text. Highest potential quality-per-parameter payoff of the three, since it transfers a teacher's learned distribution rather than just token statistics — but requires access to a larger teacher model and a heavier training pipeline (forward pass through both models, KL-divergence loss term). Worth prioritizing only if teacher access becomes available.

#### 2. Pruning / Sparsity

Zero out low-magnitude weights post-training to shrink the payload further than int8 quantization alone. Low effort — a single post-training pass, no architecture changes — but gains are modest at this model scale (25M–250M params) and accuracy can degrade without careful calibration (iterative pruning + fine-tune recovery). Lowest-effort next step if a smaller checkpoint is needed soon.

#### 3. Speculative Decoding

Use a small draft model to propose multiple tokens, verified in a single forward pass by the full model, to speed up autoregressive generation. Real wall-clock latency win at inference time, but adds a second model to build/maintain and only pays off if generation speed (not retrieval or model load time) is the actual bottleneck — not yet measured as one.

### Priority Table

| # | Item | Effort | Status | Unblocks |
|---|---|---|---|---|
| 1 | Knowledge distillation | High | not started | Quality-per-parameter, if a teacher model is available |
| 2 | Pruning / sparsity | Low | not started | Smaller payload beyond int8 quantization |
| 3 | Speculative decoding | Medium | not started | Faster generation, if latency becomes the bottleneck |
