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
| 2 | Switch `Trainer`'s AMP dtype from fp16 to bf16 | `training/trainer.py` | not started |
| 3 | `torch.compile(mode="max-autotune")` for pretraining | `training/trainer.py` | not started |
| 4 | Rebalance `batch_size` vs `accumulate_steps` | trainer configs (`train.py`, `finetune.py`) | not started — needs empirical VRAM headroom check |
| 5 | Skip no-op padding in `PaddingCollator` for fixed-length windows | `data/collator.py` | not started |

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

`Trainer` compiles with the default mode
([`trainer.py:409`](../grimoire_ai/llm/training/trainer.py#L409)). Pretraining runs are long (thousands of
steps) at fixed shapes every step (same `batch_size`, same `seq_len`) —
exactly the case where `mode="max-autotune"`'s extra warmup cost amortizes
well. Worth an A/B on your hardware since the gain is compile-time-vs-run-length
dependent; not worth changing for short fine-tune runs (a few hundred steps)
where the extra warmup may not pay back.

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

## Recommendation

Tackle in status-table order: **#1 first** — contained to one file, and the
only item on this list fixing something with *zero* existing optimization
rather than tuning one that's already partially applied. **#2 and #3** are
cheap flag flips, worth A/B testing back-to-back once #1 lands. **#4**
needs empirical VRAM headroom testing before committing to it, so it comes
after the flag flips. **#5** is real but marginal — clean up last, once the
bigger wins are measured and it's clear what's left on the table.
