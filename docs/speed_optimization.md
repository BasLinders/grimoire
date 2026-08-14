# Training Throughput Optimization — Candidate Improvements

Candidate changes to make the training loop itself (pretrain, fine-tune,
embedding contrastive tuning) run faster in wall-clock time. This is
deliberately scoped to *throughput only* — it does not touch model quality,
corpus size, or architecture. Those are tracked separately in
[`architecture_optimization.md`](architecture_optimization.md) (items #6-9
in particular are quality/architecture, not speed) and
[`expansion_PLAN.md`](expansion_PLAN.md). Throughput is the priority right
now because pretrain/fine-tune/embed-tune runs take long enough that
iterating on corpus size and fine-tuning quality is bottlenecked on wall
time, not on ideas.

Written after auditing `Trainer` (`grimoire_ai/llm/training/trainer.py`)
against `EmbedTuner` (`grimoire_ai/llm/training/embed_tune.py`) and finding
they've drifted: `Trainer` already has mixed precision and `torch.compile`;
`EmbedTuner` has neither. Every item below is a config/flag change or a
small, mechanical port of infrastructure that already exists elsewhere in
the codebase — none of it is new capability.

## Status

| # | Item | Where | Status |
|---|---|---|---|
| 1 | Port AMP + `torch.compile` into `EmbedTuner` | `training/embed_tune.py` | ✓ shipped — [PR #183](https://github.com/BasLinders/grimoire/pull/183) |
| 2 | Switch `Trainer`'s AMP dtype from fp16 to bf16 | `training/trainer.py` | ✓ shipped — [PR #184](https://github.com/BasLinders/grimoire/pull/184) |
| 3 | `torch.compile(mode="max-autotune")` for pretraining | `training/trainer.py` | ✓ shipped as opt-in `compile_mode`/`--compile-mode` — [PR #185](https://github.com/BasLinders/grimoire/pull/185) — needs A/B on real hardware |
| 4 | Rebalance `batch_size` vs `accumulate_steps` | trainer configs (`train.py`, `finetune.py`) | not started — needs empirical VRAM headroom check |
| 5 | Skip no-op padding in `PaddingCollator` for fixed-length windows | `data/collator.py` | not started |
| 6 | Scope MPS mixed precision / compile correctly instead of blanket-disabling | `device.py`, `trainer.py`, `embed_tune.py` | not started — needs Apple Silicon hardware to validate |

## 1. `EmbedTuner` has no AMP or `torch.compile` — the biggest gap

`Trainer.__init__`/`train()` runs every step through `torch.autocast` +
`GradScaler`, and wraps the model in `torch.compile` on CUDA
([`trainer.py:632-690`](../grimoire_ai/llm/training/trainer.py#L632-L690),
[`trainer.py:409`](../grimoire_ai/llm/training/trainer.py#L409)).
`EmbedTuner.train_step` / `train_step_pairs`
([`embed_tune.py:415-428`](../grimoire_ai/llm/training/embed_tune.py#L415-L428)) have none of that — plain fp32,
eager mode. Worse, `train_step` runs the model twice per step (`emb_a`,
`emb_b`, the SimCSE self-pair). On Ampere (RTX 3050) tensor cores, fp16/bf16
autocast alone is typically ~2x on matmul-heavy work, so this loop is
currently paying full fp32 cost, twice, every single step, for no reason.

This is a straight port of the pattern already proven in `Trainer.__init__`
(autocast context, `GradScaler`, `torch.compile` wrapping) into
`EmbedTuner.__init__`/`train_step`/`train_step_pairs`. Contained to one
file, no interface changes for callers (`scripts/embed_tune.py`).

## 2. fp16 → bf16 for `Trainer`'s AMP

`Trainer` hardcodes `dtype=torch.float16` with a `GradScaler`
([`trainer.py:633-637`](../grimoire_ai/llm/training/trainer.py#L633-L637)). Ampere supports bf16 natively. bf16
keeps fp32's exponent range, so it never underflows — no loss-scaling
retries. Today, whenever `GradScaler` overshoots its scale factor, that step
is silently skipped: a full forward + backward pass computed and thrown
away with no optimizer update. bf16 removes that failure mode entirely and
lets `GradScaler` be dropped from the bf16 code path. Raw matmul throughput
vs fp16 is roughly a wash on the same tensor cores — the win here is fewer
wasted steps and less moving machinery, not raw FLOPs.

## 3. `torch.compile` mode for pretraining

`Trainer` compiled with the default mode unconditionally. Pretraining runs
are long (thousands of steps) at fixed shapes every step (same
`batch_size`, same `seq_len`) — exactly the case where `mode="max-autotune"`'s
extra warmup cost amortizes well. Not worth changing for short fine-tune
runs (a few hundred steps) where the extra warmup may not pay back — so
this is exposed as an opt-in `compile_mode` constructor arg on `Trainer`
(default `None`, unchanged behavior) and a `--compile-mode` flag on
`train.py`'s pretrain CLI specifically, not switched on by default: the
actual crossover point where the extra autotuning pays for itself depends
on hardware, not just step count, and there's no GPU available in this dev
environment to establish that empirically. Run pretraining once with
`--compile-mode max-autotune` and once without, and compare wall-clock per
step after the (slower) first few steps.

## 4. `batch_size` vs `accumulate_steps` rebalance

Current defaults: pretrain `batch_size=4, accumulate_steps=8` (effective
batch 32); fine-tune `batch_size=4, accumulate=4`. Every accumulation
micro-step re-enters the Python loop, re-slices the RoPE/causal-mask
buffers, rebuilds the attention bias — fixed per-step overhead that matters
more on a small model where the actual matmul is cheap relative to that
dispatch cost. If VRAM has headroom (check with `nvidia-smi` mid-run),
doubling `batch_size` and halving `accumulate_steps` keeps the same
effective batch size and memory-checkpointing tradeoff but roughly halves
that overhead. Not guaranteed free on a 4 GB card (`finetune.py`'s own
docstring targets an RTX 3050 4 GB) — this one needs measuring, not assuming.

## 5. `PaddingCollator` pads sequences that are never actually variable-length

`PaddingCollator.__call__` does flip → `pad_sequence` → flip on every batch
([`collator.py:95-98`](../grimoire_ai/llm/data/collator.py#L95-L98)), but `TokenizedDataset` always yields
fixed-length windows during pretraining — there is nothing to pad. Because
`num_workers=0`, this runs synchronously on the main thread, so it's never
overlapped with GPU work. Cheap per call, but it happens on every
micro-batch (`batch_size × accumulate_steps` times per optimizer step) for
the full length of a run. Lowest priority — worth a quick "all lengths
equal → plain `torch.stack`" fast path once the bigger items are measured.

## 6. MPS is a first-class device already — the throughput gating isn't scoped to match

`select_device()`, `InferenceEngine`, `Trainer`, `EmbedTuner`, and the UI's
device dropdown all already treat MPS as a real, selected device today
(`torch.mps.recommended_max_memory()` even backs the UI's VRAM display in
`ui/shared.py:95`), and every op the model uses — SDPA, RMSNorm, SwiGLU,
GQA/RoPE — is a standard PyTorch op with MPS coverage. Training and
inference already run correctly on MPS. What's missing is throughput
parity, and the current code doesn't attempt it in a scoped way — it hangs
every optimization off one hardcoded `device == "cuda"` check, justified by
one blanket comment repeated three times (`device.py:7-10`,
[`trainer.py:342-345`](../grimoire_ai/llm/training/trainer.py#L342-L345),
[`embed_tune.py:362-367`](../grimoire_ai/llm/training/embed_tune.py#L362-L367)):
*"GradScaler is CUDA-only, MPS autocast is immature."*

That sentence bundles three genuinely independent facts that don't all
point the same way:

- **`torch.amp.GradScaler` has no MPS backend.** True, and permanent — not
  a maturity issue, an API gap.
- **fp16 autocast needs `GradScaler`.** Also true, on any device — fp16's
  narrow exponent range underflows without loss scaling.
- **bf16 autocast needs no loss scaling at all**, on any device — same
  exponent range as fp32, so it never underflows. This is the exact
  reasoning behind item #2 above for CUDA, and it applies to MPS
  *unchanged*: bf16 autocast without `GradScaler` sidesteps the one real
  MPS blocker entirely. The "GradScaler is CUDA-only" fact is true but
  irrelevant to a dtype that was never going to use `GradScaler` anyway.
- **MPS autocast op coverage and `torch.compile`'s MPS backend being
  "immature"** was a true snapshot of some earlier PyTorch version, not a
  permanent architectural ceiling — it needs re-checking against whatever
  PyTorch ships on the actual Mac this runs on, not assumed indefinitely
  from a comment that predates several PyTorch releases.

Scoped as three independent sub-items, in order of confidence:

1. **bf16 autocast on MPS, no `GradScaler`.** Replace the CUDA-only
   `_use_amp` boolean with a dtype-aware check: enable bf16 autocast on
   `device_type="mps"`, keep the scaler permanently disabled (bf16 never
   needs it, on any device). Highest confidence of the three — needs no
   CUDA-only API at all, and bf16 autocast op coverage on MPS has been
   stable for a while.
2. **`torch.compile` on MPS.** Attempt it behind the same
   `hasattr(torch, "compile")` + `suppress_errors=True` fallback pattern
   already used for CUDA, instead of gating MPS out of compilation
   entirely. Lower confidence — MPS compile support is real but has
   historically had rougher edges (missing ops, silent eager fallback)
   than CUDA's Inductor path. Needs to be tried and timed on real hardware,
   not assumed to work or assumed to help.
3. **cuDNN benchmark / pinned memory.** Genuinely CUDA-specific, no MPS
   equivalent — correctly excluded today, nothing to change here.

Source: [PyTorch MPS backend docs](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Recommendation

Tackle in status-table order: **#1 first** — contained to one file, and the
only item on this list fixing something with *zero* existing optimization
rather than tuning one that's already partially applied. **#2 and #3** are
cheap flag flips, worth A/B testing back-to-back once #1 lands. **#4**
needs empirical VRAM headroom testing before committing to it, so it comes
after the flag flips. **#5** is real but marginal — clean up last, once the
bigger wins are measured and it's clear what's left on the table. **#6**
is a distinct case: implementing it is low-risk (same fallback-on-failure
pattern already used for CUDA compile), but there is no Apple Silicon
hardware available to run it against right now, so "implemented" and
"verified" are different things here until it's actually run on a Mac.
