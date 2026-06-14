# Grimoire Roadmap V3

## Context

Grimoire is a domain-specific language model toolkit — ~25M parameter transformer with GQA, RoPE, SwiGLU, RMSNorm — targeting specific corpora (mathematics, D&D, mythology/fantasy). The architecture is solid; what follows is the plan to make it meaningfully more capable and deployable.

Current strengths: KV-cache, mixed-precision (AMP), Flash SDPA, torch.compile, hybrid lexical+semantic RAG, multi-turn conversation state, BPE tokenizer, full pre-train → fine-tune pipeline, agent registry.

Current gaps: no quantization, no LoRA, no evaluation harness, no tool-calling, recomputed RAG index per session, no export format beyond `.pt`.

---

## Roadmap

### 1. int8 Quantization (Memory — CPU viability)

**Why first:** Moving to medium-85M or large-250M breaks the "runs on CPU" claim without it. Dynamic int8 quantization via `torch.quantization.quantize_dynamic` is zero-shot (no retraining), cuts model size ~4×, and makes CPU inference genuinely faster.

- Apply `quantize_dynamic` to linear layers at inference load time
- Add `--quantize` flag to the inference engine and Gradio UI
- Verify output quality vs. fp32 baseline on held-out validation set
- Document memory footprint before/after per size preset

### 2. Gradient Checkpointing (Memory — Training)

**Why second:** Unblocks training medium-85M and large-250M on consumer GPUs (8–16 GB VRAM). Trades ~20% training speed for roughly half the activation memory.

- Add `model.gradient_checkpointing_enable()` toggle in the pre-training loop
- Expose as checkbox in the Pre-train tab ("Reduce VRAM (slower)")
- Confirm compatibility with `torch.compile()` (may need to disable compile when checkpointing is on)

### 3. Evaluation Harness

**Why third:** Without domain-specific evals, there is no feedback signal for any other roadmap item. Every downstream decision — does quantization hurt quality? did fine-tuning improve accuracy? — requires this first.

- **Perplexity eval:** compute bits-per-character on a held-out slice of each corpus group (math, fantasy)
- **Retrieval hit-rate:** for a fixed query set, measure what fraction of top-1 RAG results are relevant
- **D&D rules quiz:** a curated set of 50–100 factual Q&A pairs; score exact/fuzzy match of model responses
- **Math explanation quality:** rubric-scored sample (can be manual for v1)
- Expose a "Run evals" button in the UI; log results to `data/eval/` with timestamp

### 4. Math → Python CLI (Tool Calling)

**Why fourth:** The D&D fine-tuning data already trains the model to decline arithmetic and defer to a tool. This closes that loop.

- Detect math-intent queries via pattern matching on the generated `<AST>` preamble (e.g., model emits `<TOOL:python>expr</TOOL>` before its narrative response)
- Pass expression to a sandboxed evaluator (SymPy or `asteval` — not `eval()`) in a subprocess
- Inject the result as a `<SEP>`-delimited context line before the model continues generation
- Extend fine-tuning data with tool-call examples (`<TOOL:python>`, result, narrative)
- Add opt-in toggle in Chat tab ("Enable math tool")

### 5. LoRA / Adapter Fine-Tuning

**Why fifth:** Full fine-tuning on 29–36 examples risks catastrophic forgetting. LoRA freezes base weights and trains ~0.5% of parameters — better regularization, faster, and each domain persona becomes a 2–5 MB `.lora` file rather than a full checkpoint.

- Implement `LoRALinear` wrapper (rank-decomposition `A × B` bypass on `q_proj`, `v_proj`)
- Add `--lora-rank` and `--lora-alpha` args to the fine-tune pipeline
- Save/load LoRA weights separately from base checkpoint
- Merge adapters into base weights for export (`merge_and_unload()`)

### 6. Agent Routing

**Why sixth:** `AgentRegistry` already loads multiple agents. What's missing is automatic dispatch.

- Lightweight intent classifier: score query against each agent's domain corpus using the existing lexical retriever; route to highest-scoring agent
- Fallback: if no agent exceeds a confidence threshold, use the default agent
- Log routing decisions to the conversation trace for debugging
- UI: show which agent handled each turn in the chat history

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

## Priority Table

| # | Item | Effort | Unblocks |
|---|---|---|---|
| 1 | int8 quantization | Low | CPU inference at medium/large scale |
| 2 | Gradient checkpointing | Low | Training medium/large on consumer GPU |
| 3 | Evaluation harness | Medium | All quality-sensitive decisions |
| 4 | Math → Python CLI | Medium | Closing the tool-call loop from fine-tune data |
| 5 | LoRA adapters | Medium | Per-agent specialization without catastrophic forgetting |
| 6 | Agent routing | Low (after 5) | Automatic multi-domain dispatch |
| 7 | Persistent RAG index | Medium | Startup time at 500M token corpus scale |
| 8 | GGUF export | High | Widest deployment, no Python dependency |
