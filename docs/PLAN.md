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
| **6.5** | Training UI: Gradio app (Pre-train, Fine-tune, Chat tabs) + `Trainer.on_log` live streaming | ✓ done |
| **7** | Conversation state: `ConversationState`, `InferenceEngine.chat()`, terminal chat CLI | ✓ done |
| **7.5** | Integration test suite: full pipeline from BPE training to multi-turn `chat()` | ✓ done |
| **8** | Agent registry (`agents.json`) + agent selector dropdown in Chat UI | ✓ done |
| **9** | Saga corpus: D&D 5e SRD (24 sections) + encounter math + probability references | ✓ done |
| **10** | Saga instruction fine-tuning: 30-example JSONL dataset + fine-tune + validation scripts | ✓ done |
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

Dynamic int8 quantization via `torch.quantization.quantize_dynamic` / `torchao`. Zero-shot, cuts model size ~4×, speeds up CPU inference. Tries `torchao` first; falls back to legacy API with deprecation warning suppressed. `InferenceEngine(quantize=True)` + UI checkbox in Chat tab.

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

**Why fourth:** The D&D fine-tuning data already trains the model to decline arithmetic and defer to a tool. This closes that loop.

- **Query-side detection:** `MathTool.detect()` identifies arithmetic in queries via regex (operators, percent-of, Unicode multiply/power); dice notation excluded (`grimoire_ai/tools/math_tool.py`)
- **Safe evaluator:** pure `ast`-based visitor — no `eval()`, no subprocess needed; supports arithmetic operators, parentheses, and whitelisted math functions (sqrt, sin, cos, log, …)
- **Context injection:** result prepended as a synthetic `QueryResult` with excerpt `[Math] expr = result`; flows through the existing `PromptBuilder` context slot
- **Response-side tag resolution:** `MathTool.process_response()` replaces `<TOOL:python>…</TOOL>` tags emitted by fine-tuned models with evaluated results
- **Fine-tune data:** 15 examples with `<TOOL:python>` format in `scripts/finetune_data/tool_call_examples.jsonl`
- **CLI:** `--math-tool` flag on `python -m grimoire_ai.cli.chat`
- **UI:** "Enable math tool" checkbox in Chat tab (wired to both agent load and manual load)

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

### 7. Persistent RAG Index

**Why seventh:** Semantic embeddings are recomputed from scratch every session. As corpus grows toward 500M tokens this becomes a minutes-long blocking startup.

- Pre-compute chunk embeddings after preprocessing; persist as a flat numpy memmap alongside `corpus.bin`
- On session start, load the index from disk; recompute only for new/changed chunks (hash-based staleness check)
- Optional FAISS index for approximate nearest-neighbour at scale (replaces brute-force cosine)
- Expose "Rebuild index" button in the Corpus tab

### 8. GGUF Export

**Why last:** The long-term deployment story — no Python dependency, 4-bit quantization, widest hardware support via llama.cpp.

- Write `scripts/export_gguf.py`: map GrimoireTransformer weight names to GGUF tensor layout, write header + tensors
- Support Q4_K_M quantization in the export (strongest quality/size tradeoff in llama.cpp)
- Validate round-trip: load exported GGUF in llama.cpp, confirm perplexity within 5% of fp32 baseline
- Document deployment instructions (`llama-cli`, `llama-server`)

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
| 7 | Persistent RAG index | Medium | — | Startup time at 500M token corpus scale |
| 8 | GGUF export | High | — | Widest deployment, no Python dependency |
