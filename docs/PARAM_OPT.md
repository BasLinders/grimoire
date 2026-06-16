# Parameter Auto-Detection Reference

Assessment of ~75 UI parameters across Chat, Pre-train, Fine-tune, Scale, Evaluate, Ingest, and Corpus tabs. Documents which parameters can be auto-detected, from what source, and at what priority.

**Currently auto-detected**: device type only (`engine.py:133–134`).

| Phase | Status |
|-------|--------|
| 1 — Zero-effort pre-fills | **Complete** |
| 2 — Filesystem discovery | Pending |
| 3 — agents.json config | Pending |
| 4 — Device-aware suggestions | Pending |
| 5 — Corpus-size-aware model preset | Pending |
| 6 — Query-aware generation | Pending |

---

## Phase 1 — Zero-effort pre-fills ✓

No runtime probing required; values are always constant or trivially derived.

| Parameter | Tab(s) | Fix |
|-----------|--------|-----|
| `chat_vocab` | Chat | Pre-fill `"data/tokenizer/bpe.json"` or hide |
| `ft_vocab` | Fine-tune | Pre-fill `"data/tokenizer/bpe.json"` or hide |
| `ev_vocab` | Evaluate | Pre-fill `"data/tokenizer/bpe.json"` or hide |
| `ci_vocab` | Corpus | Pre-fill `"data/tokenizer/bpe.json"` or hide |
| `pt_ckpt_dir` | Pre-train | Pre-fill `"checkpoints/pretrain/"` |
| `ft_ckpt_dir` | Fine-tune | Pre-fill `"checkpoints/finetune/"` |
| `sc_corpus` | Scale | Pre-fill `"data/processed/corpus.bin"` |
| `pt_corpus` | Pre-train | Pre-fill `"data/processed/corpus.bin"` |
| `pt_warmup` | Pre-train | Derive as `round(pt_steps * 0.05)` |
| `ft_warmup` | Fine-tune | Derive as `round(ft_steps * 0.02)` |
| `pt_log` | Pre-train | Derive as `round(pt_steps * 0.005)` |
| `ft_log` | Fine-tune | Derive as `round(ft_steps * 0.05)` |
| `pt_save` | Pre-train | Derive as `round(pt_steps * 0.10)` |
| `ft_save` | Fine-tune | Derive as `round(ft_steps * 0.20)` |
| `pt_eval_every` | Pre-train | Mirror `pt_save` |
| `ft_eval_every` | Fine-tune | Mirror `ft_save` |
| `ft_lora_alpha` | Fine-tune | Set equal to `ft_lora_rank` |
| `ft_lora_targets` | Fine-tune | Default to `"q_proj,v_proj"` (never user-tuned) |
| `ing_cleaning` | Ingest | Default `"standard"` (safe, no detection needed) |
| `ing_recursive` | Ingest | Auto-enable when mode is `"Directory"` |

**Implementation notes:**
- Vocab paths, standard directory defaults, `ing_cleaning`, `ing_recursive`, and `ft_lora_targets` were already pre-filled/handled in the original code.
- Added `_derive_pt_step_params` and `_derive_ft_step_params` helper functions; wired to `pt_steps.change` and `ft_steps.change` events so warmup, log, save, and eval_every auto-update (`app.py`).
- Added `_derive_lora_alpha`; wired to `ft_lora_rank.change`. Also updated `_toggle_finetune_mode` to sync `ft_lora_alpha` when mode switches between Base and Agent.

---

## Phase 2 — Filesystem discovery

Scan local paths on tab load; replace freetext fields with dropdowns populated from found files.

| Parameter | Tab(s) | Scan Path | Filter |
|-----------|--------|-----------|--------|
| `chat_ckpt` | Chat | `checkpoints/` | `*.pt` |
| `pt_resume` | Pre-train | `checkpoints/pretrain/` | `*.pt` |
| `ft_pretrain_ckpt` | Fine-tune | `checkpoints/pretrain/` | `*.pt` |
| `ft_resume` | Fine-tune | `checkpoints/finetune/` | `*.pt` |
| `ev_checkpoint` | Evaluate | `checkpoints/` | `*.pt` |
| `sc_checkpoint` | Scale | `checkpoints/` | `*.pt` |
| `ci_checkpoint` | Corpus | `checkpoints/` | `*.pt` |
| `chat_lora` | Chat | `checkpoints/finetune/` | `*.lora` |
| `chat_corpus_dir` | Chat | `data/corpus/` | subdirectories |
| `ci_corpus_dir` | Corpus | `data/corpus/` | subdirectories |
| `ev_corpus_dir` | Evaluate | `data/corpus/` | subdirectories |
| `ft_data` | Fine-tune | `data/finetune/` | `*.jsonl` |

Default selection: most recently modified file in each list.

---

## Phase 3 — Config-driven (agents.json)

When the user selects an agent from `agent_dropdown`, auto-populate generation and retrieval fields from the agent's entry in `agents.json`. The registry already loads this data (`registry.py:140–151`); it needs to be wired to the UI components.

| UI Parameter | agents.json field |
|--------------|-------------------|
| `chat_temp` | `gen_config.temperature` |
| `chat_top_k` | `gen_config.top_k` |
| `chat_top_p` | `gen_config.top_p` |
| `chat_tokens` | `gen_config.max_new_tokens` |
| `chat_corpus_dir` | `corpus_dirs[0]` |
| `chat_lora` | `lora_path` (if present) |
| `chat_ckpt` | `checkpoint` (if present) |

**Implementation note**: wire a `gr.change` event on `agent_dropdown` that calls a function loading the selected agent's config and returning updated component values.

---

## Phase 4 — Device-aware suggestions

Use `torch.cuda.is_available()` and `torch.cuda.get_device_properties()` at startup. Already partially done in `engine.py:133–134`.

| Parameter | Tab(s) | Logic |
|-----------|--------|-------|
| `chat_quantize` | Chat | Auto-enable if `device == "cpu"` |
| `pt_grad_ckpt` | Pre-train | Suggest `True` if GPU VRAM < 16 GB |
| `pt_batch` | Pre-train | See batch sizing table below |
| `ft_batch` | Fine-tune | See batch sizing table below |
| `pt_accum` | Pre-train | `max(1, target_effective_batch // pt_batch)` |
| `ft_accum` | Fine-tune | `max(1, target_effective_batch // ft_batch)` |

**Batch size heuristic** (target effective batch = 32):

| VRAM | Suggested `batch` | Notes |
|------|-------------------|-------|
| CPU / < 4 GB | 1 | Use high `accum` |
| 4–8 GB | 2 | |
| 8–16 GB | 4 | Current default |
| > 16 GB | 8–16 | Reduce `accum` accordingly |

---

## Phase 5 — Corpus-size-aware model preset

On Pre-train tab load (or when `pt_corpus` changes), read the binary's token count and suggest a preset. Rules already documented in `config.py:173–182`.

| Corpus token count | Suggested `pt_preset` |
|--------------------|----------------------|
| < 100 M | `small-25M` |
| 100 M – 500 M | `medium-85M` |
| > 500 M | `large-250M` |

When `pt_preset` changes, the following fields auto-update (already handled by `app.py` preset logic):

`pt_d_model`, `pt_n_layers`, `pt_n_heads`, `pt_n_kv_heads`, `pt_d_ff`

Also derive `pt_steps` via Chinchilla scaling (reference: Scale tab logic):

```
optimal_tokens = 20 × n_params
optimal_steps  = optimal_tokens / (batch × accum × seq_len)
```

---

## Phase 6 — Query-aware generation (optional / runtime)

Analyse the user's message text before generation and adjust generation defaults. Lower priority; can be done without UI changes via a pre-processing hook.

| Heuristic | Condition | Action |
|-----------|-----------|--------|
| Factual query | Starts with "What", "How many", "List", "When", "Define" | Suggest `temp ≤ 0.5` |
| Creative query | Starts with "Write", "Create", "Design", "Imagine" | Suggest `temp ≥ 0.9` |
| Short query (< 10 tokens) | — | Suggest `max_tokens = 256` |
| Long query (> 50 tokens) | — | Suggest `max_tokens = 128` |

---

## Parameter count summary

| Phase | Parameters addressed | Effort |
|-------|---------------------|--------|
| 1 — Zero-effort pre-fills | ~20 | Trivial |
| 2 — Filesystem discovery | ~13 | Low |
| 3 — agents.json config | ~7 | Low |
| 4 — Device-aware | ~6 | Low–Medium |
| 5 — Corpus-size preset | ~7 | Medium |
| 6 — Query-aware (runtime) | Optional | Medium |

**Total reducible**: ~50 of ~75 parameters can be hidden, pre-filled, or derived without user input.

---

## Key source files

| File | Relevance |
|------|-----------|
| `grimoire_ai/ui/app.py` | All UI components (2,468 lines) |
| `grimoire_ai/llm/model/config.py` | `TransformerConfig`, preset definitions |
| `grimoire_ai/llm/inference/sampler.py` | `GenerationConfig` dataclass |
| `grimoire_ai/llm/inference/engine.py` | Device detection (`lines 133–134`) |
| `grimoire_ai/agents/registry.py` | Agent loading, gen_config wiring (`lines 140–151`) |
| `grimoire_ai/llm/training/trainer.py` | Trainer hyperparameter defaults (`lines 100–130`) |
| `agents.json` | Per-agent gen_config, corpus_dirs, lora_path |
