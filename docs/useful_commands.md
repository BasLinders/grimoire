# Useful Commands

A scratchpad of check/verify/test/loop commands that come up repeatedly
but don't have a natural home in another doc — either because they're
one-off diagnostics rather than part of a documented pipeline stage
(`setup-training.md`, `setup-inference.md`), or because they're a
multi-seed/multi-run *pattern* around an existing script rather than a
single invocation worth putting in that script's own usage docstring.

Not a replacement for those docs — if a command belongs to a specific
pipeline stage with its own doc, it's cross-referenced here rather than
duplicated. Add to this file when a command proves useful more than
once and doesn't already fit elsewhere.

## Corpus

**Quick token count, without touching the production corpus.** Useful
for checking progress against the ~500M Chinchilla-optimal target (see
`expansion_PLAN.md`) after adding new sources, before committing to a
real preprocess + retrain. Writes to scratch filenames so it never
overwrites `data/processed/corpus.bin`/`quality_report.jsonl` — read-only
in effect, even though the underlying script always writes a `.bin`
file:

```bash
python -m grimoire_ai.llm.data.preprocessing \
    --input data/corpus/saga/ \
    --input data/corpus/saga_derived/ \
    --input data/corpus/general_qa/ \
    --output data/processed/corpus_count_scratch.bin \
    --vocab data/tokenizer/bpe.json \
    --quality-filter \
    --quality-report data/processed/quality_report_scratch.jsonl
```

No `--weight-pattern` needed — weighting only tags sidecar files for
`build_source_weights.py`, it doesn't affect the total token count.
Prints a running per-file count and a final `Total: N tokens` line.
Delete the two scratch output files afterward; nothing downstream reads
them.

**Preview what `--quality-filter` would drop, before trusting it on a
new source.** Non-destructive — never modifies or deletes anything, just
reports what the automatic filter in `grimoire-preprocess` would catch:

```bash
python scripts/score_corpus_quality.py --corpus-dir data/corpus/general_qa/
python scripts/score_corpus_quality.py --corpus-dir data/corpus/general_qa/ --report quality_preview.jsonl
```

Run this against any freshly-scraped directory before its first real
`--quality-filter` preprocess run — cheapest way to catch a scraper
producing systematically bad output (e.g. nav-menu-only pages, OCR junk)
before it silently gets dropped (or worse, silently kept) at scale.

**Near-duplicate check before merging new content into the corpus:**

```bash
python scripts/dedup_corpus.py --corpus-dir data/corpus/saga/ --new-glob "gutenberg_*.txt"
python scripts/dedup_corpus.py --corpus-dir data/corpus/saga/ --new-glob "gutenberg_*.txt" --threshold 0.3
```

Lower `--threshold` = stricter (catches more near-duplicates, more false
positives); see `expansion_PLAN.md`'s derived-adventure pilot for a case
that used a looser 0.2-0.25 threshold deliberately (shared vocabulary
between adventures was expected, not a duplication signal).

## Checkpoint comparison and verification

**Multi-seed qualitative comparison.** A single seed's sampling output
is not a reliable signal on its own (`compare_checkpoints.py`'s own
docstring: an unseeded run once showed 4/6 prompts collapsing, a rerun
on the same prompts showed none of those four recurring) — this
project's convention is always a few seeds, never one.

PowerShell:

```powershell
0..4 | ForEach-Object {
    python scripts/compare_checkpoints.py `
        --checkpoint-a checkpoints/finetune/A/step_XXXX.pt --label-a A `
        --checkpoint-b checkpoints/finetune/B/step_XXXX.pt --label-b B `
        --vocab data/tokenizer/bpe.json `
        --seed $_
}
```

Bash:

```bash
for seed in 0 1 2 3 4; do
    python scripts/compare_checkpoints.py \
        --checkpoint-a checkpoints/finetune/A/step_XXXX.pt --label-a A \
        --checkpoint-b checkpoints/finetune/B/step_XXXX.pt --label-b B \
        --vocab data/tokenizer/bpe.json \
        --seed "$seed"
done
```

**Same-checkpoint before/after comparison** (isolating one decode-time
flag's effect, rather than comparing two different checkpoints) — pass
the *same* checkpoint as both `--checkpoint-a` and `--checkpoint-b`, and
vary only the flag under test between two separate runs. Used this way
to validate the `RepetitionLoopGuard` templated-loop fix against the
real bug (`training_PLAN.md`'s Step 7): one run with the old guard
settings, one with the new, same checkpoint/prompts/seed, diffed by
hand. `compare_checkpoints.py` applies one set of loop-guard flags to
*both* sides of a single invocation, so this pattern needs two separate
invocations, not one:

```bash
python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/finetune/X/step_XXXX.pt --label-a same \
    --checkpoint-b checkpoints/finetune/X/step_XXXX.pt --label-b same \
    --vocab data/tokenizer/bpe.json --seed 4 \
    --loop-guard-max-repeats 3 --loop-guard-max-period 4 \
    > before.txt

python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/finetune/X/step_XXXX.pt --label-a same \
    --checkpoint-b checkpoints/finetune/X/step_XXXX.pt --label-b same \
    --vocab data/tokenizer/bpe.json --seed 4 \
    --loop-guard-max-repeats 3 --loop-guard-max-period 16 --loop-guard-template-match-ratio 0.6 \
    > after.txt
```

**Multi-seed quiz-eval comparison** (aggregate pass-rate/kw-recall/F1
across seeds, matching a real deployment's sampling instead of the
quiz's own greedy default — see `evaluate.py`'s `--quiz-temperature`
etc.):

PowerShell:

```powershell
0..4 | ForEach-Object {
    python scripts/evaluate.py `
        --checkpoint checkpoints/finetune/X/step_XXXX.pt `
        --vocab data/tokenizer/bpe.json `
        --quiz-repetition-penalty 1.3 `
        --quiz-temperature 0.8 --quiz-top-k 50 --quiz-top-p 0.9 `
        --quiz-seed $_
}
```

Bash:

```bash
for seed in 0 1 2 3 4; do
    python scripts/evaluate.py \
        --checkpoint checkpoints/finetune/X/step_XXXX.pt \
        --vocab data/tokenizer/bpe.json \
        --quiz-repetition-penalty 1.3 \
        --quiz-temperature 0.8 --quiz-top-k 50 --quiz-top-p 0.9 \
        --quiz-seed "$seed"
done
```

Match `--quiz-repetition-penalty`/`--quiz-temperature`/`--quiz-top-k`/
`--quiz-top-p` to whatever's actually in `agents.json`'s `gen_config`
for the checkpoint under test, or the comparison isn't representative of
what users actually see.

**Per-tier validation loss** (requires the checkpoint to have trained
with `--val-stratified`, and the same `--val-split` it used):

```bash
python scripts/eval_per_tier.py \
    --checkpoint checkpoints/pretrain/X/step_XXXX.pt \
    --corpus data/processed/corpus.bin \
    --val-split 0.01
```

**Single-checkpoint raw completion check** (no fine-tune instruction
format, no comparison — just "does this pretrain checkpoint produce
coherent text"):

```bash
python scripts/qualitative_check.py \
    --checkpoint checkpoints/pretrain/X/step_XXXX.pt \
    --vocab data/tokenizer/bpe.json
```

## Fine-tune data

**Validate a JSONL fine-tune file** (schema + tokenizer round-trip)
before training on it:

```bash
python scripts/validate_finetune_data.py \
    --data data/finetune/X.jsonl \
    --vocab data/tokenizer/bpe.json \
    --max-seq-len 512
```

**Downsample an oversized JSONL source** (e.g. one Stack Exchange site
outweighing the rest of the fine-tune mix):

```bash
python scripts/downsample_jsonl.py \
    --input data/finetune/general_se_qa.jsonl \
    --output data/finetune/general_se_qa_downsampled.jsonl \
    --n 40000
```

Sanity-check the resulting mix ratio with `wc -l` on each input file
before combining — see `training_PLAN.md`'s Step 2 note on this.

## Training utilities

**Learning-rate range test** (Smith, 2015) — finds a good peak LR for a
specific model/corpus combination rather than trusting the trainer's
hard-coded default; builds a throwaway freshly-initialized model, never
touches a real checkpoint:

```bash
python scripts/lr_range_test.py \
    --corpus data/processed/corpus.bin \
    --output lr_range_test.csv
```

**Generate synthetic corpus filler** (deterministic per seed, idempotent
re-runs — mirrors D&D sourcebook / textbook-encyclopedia style; see
`synth_*` in the `--weight-pattern` list in `setup-training.md`):

```bash
python scripts/generate_lore.py --group lore --count 3000
python scripts/generate_lore.py --group datascience --count 2000
```

## Tests

```bash
pytest                            # full unit test suite
pytest tests/test_integration.py  # end-to-end integration test
pytest tests/llm/test_constrained_decoding.py -q  # one file, quiet output
```

**Known Windows-only noise, not a regression signal:** ~34 local pytest
failures on Windows are `PermissionError`s from `tempfile.TemporaryDirectory`
cleanup racing an open file handle (seen in `test_dataset.py`,
`test_trainer.py`, `test_saga_corpus.py`) — CI runs on Linux only, where
this doesn't reproduce. Confirm a failure is this kind before treating it
as real: check the traceback ends in
`PermissionError: [WinError 32] The process cannot access the file
because it is being used by another process` during `tempfile` cleanup,
not inside the test body itself. If you need a clean run to isolate a
real regression from this noise, run just the file(s) you actually
changed rather than the full suite.
