# Parameter Auto-Detection Reference

Assessment of ~75 UI parameters across Chat, Pre-train, Fine-tune, Scale, Evaluate, Ingest, and Corpus tabs. Documents which parameters can be auto-detected, from what source, and at what priority.

**Currently auto-detected**: device type only (`engine.py:133–134`).

| Phase | Status |
|-------|--------|
| 1 — Zero-effort pre-fills | **Complete** |
| 2 — Filesystem discovery | **Complete** |
| 3 — agents.json config | **Complete** |
| 4 — Device-aware suggestions | **Complete** |
| 5 — Corpus-size-aware model preset | **Complete** |
| 6 — Query-aware generation | **Complete** |

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

## Phase 2 — Filesystem discovery ✓

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

Default selection: most recently modified file in each list. `ci_corpus_dir` auto-selects the newest corpus directory; all other path fields default to empty (user selects).

**Implementation notes:**
- Added `_scan_files(base_dir, pattern, recursive)` and `_scan_subdirs(base_dir)` helpers in `app.py`.
- All 12 path fields converted from `gr.Textbox` to `gr.Dropdown(allow_custom_value=True)` — custom paths can still be typed freely.
- Six choice lists pre-computed once at app startup inside `build_app()`: `_ckpts_all`, `_ckpts_pretrain`, `_ckpts_finetune`, `_lora_choices`, `_corpus_dirs`, `_jsonl_choices`.
- `chat_lora` and `chat_ckpt` scan `checkpoints/` recursively; resume fields scope to their respective subdirectory.

---

## Phase 3 — Config-driven (agents.json) ✓

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

**Implementation notes:**
- Added `_preview_agent_config(display_name)` in `app.py`; replaces the previous `agent_dropdown.change` lambda that only toggled `chat_routing_threshold`.
- "Auto-route" leaves all sliders and paths unchanged (no single agent to derive from); only the routing threshold visibility is toggled.
- Individual `gen_config` keys are applied only when present in the agent's entry — missing keys leave the corresponding slider at its current value.
- Registry errors (missing `agents.json`, malformed JSON) are caught silently; UI falls back to current values.

---

## Phase 4 — Device-aware suggestions ✓

Use `torch.cuda.is_available()` and `torch.cuda.get_device_properties()` at startup. Already partially done in `engine.py:133–134`.

| Parameter | Tab(s) | Logic |
|-----------|--------|-------|
| `chat_quantize` | Chat | Auto-enable if `device == "cpu"` |
| `pt_grad_ckpt` | Pre-train | Suggest `True` if GPU VRAM < 16 GB |
| `pt_batch` | Pre-train | See batch sizing table below |
| `ft_batch` | Fine-tune | See batch sizing table below |
| `pt_accum` | Pre-train | `max(1, target_effective_batch // pt_batch)` |
| `ft_accum` | Fine-tune | `max(1, target_effective_batch // ft_batch)` |

**Batch size heuristic** (target effective batch = 32 for pre-train, 16 for fine-tune):

| VRAM | Suggested `batch` | Notes |
|------|-------------------|-------|
| CPU / < 4 GB | 1 | Use high `accum` |
| 4–8 GB | 2 | |
| 8–16 GB | 4 | Previous hardcoded default |
| > 16 GB | 8 | Reduce `accum` accordingly |

**Implementation notes:**
- Added `_detect_device_profile()` helper in `app.py`; called once at the top of `build_app()`, result stored in `_dp`.
- `sc_batch` and `sc_accum` (Scale tab) are synced to `_dp["pt_batch"]` / `_dp["pt_accum"]` so the Chinchilla calculator always reflects the actual training config.
- CPU with no CUDA: `quantize=True`, `batch=1`, `accum=32/16`. CUDA: `quantize=False`, batch/accum derived from VRAM, `grad_ckpt=True` when VRAM < 16 GB.
- MPS (Apple Metal) falls into the CPU path — int8 quantization suggested, batch=1. Acceptable for now.
- If `torch` is unavailable or CUDA probe fails, returns neutral defaults identical to the previous hardcoded values.

---

## Phase 5 — Corpus-size-aware model preset ✓

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

**Implementation notes:**
- Added `_suggest_preset_from_corpus(corpus_path, pt_batch, pt_accum)` helper in `app.py`.
- Token count is read from file size (`int32` = 4 bytes each) — no numpy or torch import needed; very cheap.
- Added `_PRESET_PARAMS` dict mapping preset names to approximate param counts for Chinchilla math.
- Returns 7 `gr.update()` values covering `pt_preset`, all 5 arch fields, and `pt_steps` atomically — bypasses the `pt_preset.change` cascade since arch fields are returned directly.
- Wired to `pt_corpus.change` (inputs: `pt_corpus`, `pt_batch`, `pt_accum`) and `app.load` using a shared `_pt_corpus_outputs` list.
- `pt_warmup`, `pt_log`, `pt_save`, `pt_eval_every` are computed directly inside `_suggest_preset_from_corpus` via `_derive_pt_step_params(optimal_steps)` and included in the 11-element return list — Gradio does not guarantee that programmatic `gr.update()` outputs re-fire a component's own `.change()` handlers, so the cascade is made explicit.
- Guards: missing path, non-existent file, zero batch/accum → all-no-op returns; no crash.

---

## Phase 6 — Query-aware generation ✓

Analyse the user's message text before generation and adjust generation defaults. Lower priority; can be done without UI changes via a pre-processing hook.

| Heuristic | Condition | Action |
|-----------|-----------|--------|
| Factual query | Starts with "What", "How many", "List", "When", "Define" | `temp = min(temp, 0.5)` |
| Creative query | Starts with "Write", "Create", "Design", "Imagine" | `temp = max(temp, 0.9)` |
| Short query (< 10 words) | — | `max_tokens = 256` |
| Long query (> 50 words) | — | `max_tokens = 128` |

**Implementation notes:**
- Added `_query_gen_hints(query, temperature, max_new_tokens, adaptive_temperature)` in `app.py`; called inside `chat()` before `GenerationConfig` is constructed — no UI changes required.
- Temperature hints use clamp semantics (min/max) rather than hard overrides, so a user who has already tuned temperature below 0.5 for a factual query isn't pushed back up.
- Temperature hints are skipped entirely when `adaptive_temperature=True` (the adaptive schedule supersedes them).
- Token-budget hints use word count (whitespace split) as a fast approximation; no tokenizer import needed.
- `_FACTUAL_PREFIXES` and `_CREATIVE_PREFIXES` are module-level constants so they compile once.

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
