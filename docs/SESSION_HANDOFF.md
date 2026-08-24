# Session Handoff — 2026-08-25

Snapshot for picking up this work cold in a new session. This is a
point-in-time note, not a permanent doc — safe to delete once you've
read it and are oriented, or once its "in progress" item is resolved
and recorded properly in `training_PLAN.md`.

For the full narrative history behind everything below, read
`docs/training_PLAN.md`'s **Steps 7-10** (repetition-guard fix, corpus
rebalancing attempt, fine-tune dehedging, pattern-list widening) and
`docs/known_bugs.md`'s **"Residual Stack-Exchange-answer register
drift"** entry, which cross-references all of them.

## In progress right now

A pretrain rerun is either running or about to be launched:

```bash
python -m grimoire_ai.llm.training.train --config configs/train_config_general_expansion_v3.json --val-stratified
```

~5.6 hours on this machine (same config previously took 20,306-20,219s).
This reproduces `general_expansion_v3`'s pretrain checkpoint — its
weights were kept (`data/processed/corpus.bin` /
`data/processed/sample_weights.npy`, dated 2026-08-22, confirmed
unchanged since) but the checkpoint itself was deleted in an earlier
disk cleanup, so it needs to be rebuilt from scratch.

**Why**: testing a combination never tried — pairing `v3`'s cleaned/
reweighted pretrain corpus (Q&A dominance pulled down to ~75.4% of
effective sampling exposure, specifically to fix a register-drift tic)
with the dehedged fine-tune data (which independently fixed the same
tic, but only partially, from the fine-tune side). Step 8 proved
pretrain-reweighting alone doesn't survive fine-tuning on an all-Q&A
mix; Step 9 proved dehedging the fine-tune data alone gives a real but
partial win. Neither was tried together. This run tests whether they
compound.

### Once the pretrain run finishes

1. **Per-tier eval + qualitative check**, same as `v3` originally got
   (`docs/training_PLAN.md`'s Step 8 has the exact commands) — confirm
   the pretrain-level fix reproduces (expect ~0/6 forum-voice samples on
   the qualitative pass, matching last time).
2. **Fine-tune off this new checkpoint using the wider dehedge pattern
   list** (Step 10's `general_se_qa_dehedged_v2.jsonl` — regenerate if
   it's not still on disk, via `scripts/dehedge_finetune_data.py`) —
   this is a genuinely new combination, not a repeat of any prior run.
   `saga_se_qa.jsonl`/`open5e_qa.jsonl` stay undehedged (Step 9 found
   touching `saga_se_qa.jsonl` hurts quiz scores).
3. **Full comparison** against the currently-shipped
   `general-expansion-v1-dehedged` — quiz eval (5 seeds), qualitative
   (5 seeds, `compare_checkpoints.py`), and the full harness
   (perplexity/retrieval/quiz together via `evaluate.py`). Ship only if
   it's a clear win on at least one axis with no regression on others —
   same bar every checkpoint swap this session was held to.
4. **Record the result either way** in `training_PLAN.md` as Step 11 —
   this repo's convention is to document negative/mixed results as
   thoroughly as wins (see Steps 8 and 10, both negative/neutral
   findings kept for the record).

## Current production state

`agents.json`'s `saga.checkpoint` → `checkpoints/finetune/general-expansion-v1-dehedged/step_0013213.pt`.
`gen_config` includes the templated-repetition-loop-guard fix
(`loop_guard_max_period: 16`, `loop_guard_template_match_ratio: 0.6`).
This is the checkpoint to beat, not raw `general-expansion-v1`.

## Checkpoint disk state (all pruned to just what's live/needed)

- `checkpoints/pretrain/general_expansion_v1/` — base for current production
- `checkpoints/finetune/general-expansion-v1-dehedged/` (just the final `step_0013213.pt`) — **current production**
- `checkpoints/lora/` — 9 embedding-adapter dirs, untouched, negligible size
- Everything else from this session's experiments (`general_expansion_v2`/`v3` pretrain, `general-expansion-v1` fine-tune, `-dehedged-v2`, `-dehedged-v3`) was deleted after being superseded/discarded — **~120GB freed total this session**. `checkpoints/` is now explicitly gitignored (`.gitignore`) and was never tracked in git at all — nothing to reconcile with GitHub.

## Known open issues (not blocking, not attempted further this session)

- **`RepetitionLoopGuard` doesn't catch templated loops** — fixed and shipped (`known_bugs.md`), not open.
- **Residual SE-answer-voice register drift** — partially fixed (Step 9's dehedge), the in-progress work above is the next attempt.
- **New minor tic found in Step 8's pretrain-only qualitative check**: raw `5etools_*`/`open5e_*` markdown stat-block syntax (`# Monster Revival`, `## Traits`) leaking verbatim into `mechanics`/`cantrip`-style completions, once that tier's pretrain weight went up. Not confirmed present in current production (which uses `v1`'s pretrain, not `v3`'s), not investigated further. Would need something like `clean_stackexchange_markup.py` but for the reference-material files if it turns out to matter.
- **`docs/corpus_index_scaling.md`**: `CorpusIndex` has no memory ceiling, scoped but not started — a real architectural item for whenever corpus growth continues.
- **Corpus is at ~470M tokens (~94% of the ~500M Chinchilla-optimal target for `small-25M`)** — close enough that this session stopped chasing more scraping and moved to weighting/fine-tune-quality work instead; revisit only if pretrain-scale ever becomes the bottleneck again.

## Session-specific process notes (useful if continuing this style of work)

- **Git workflow this session**: feature branch + PR for actual code changes (scripts, `Trainer`, tests); direct-to-`main` push for docs-only/config-only changes, per explicit user instruction each time — always ask if unstated. After any push, **verify it actually landed** (`git fetch origin && git log origin/main --oneline` and/or `git show origin/main:<file> | grep ...`) — a PR merge race cost one fix commit that had to be cherry-picked back in earlier this session. Clean up local *and* remote branches after every confirmed merge.
- **This worktree's own branch (`claude/repo-status-next-steps-b7d2cc`) is stale** — it predates most of this session's commits. Always `git fetch origin && git checkout -b <new-branch> origin/main` before starting new work, never edit directly on the stale branch (checking it out again discards uncommitted changes silently — happened once this session).
- **`checkpoints/finetune/` fine-tune runs now auto-prune to the last 3 checkpoints** (`--keep-last-n-checkpoints`, default 3, merged this session) — no more manual cleanup needed after a fine-tune run. Pretrain (`train.py`) is unaffected, its `save_every` is already sparse.
- **`data/corpus/general_qa_source/`** holds the pre-markup-cleanup backup of the general-content Stack Exchange scrape — `build_finetune_data_from_qa.py` needs this directory, not the cleaned `data/corpus/general_qa/`, since the cleanup strips the structural markers Q&A extraction depends on (same as the older `saga_se_qa_source/` pattern for `rpg_se_*`).
- Archive.org is blocked on this network for Stack Exchange dumps; `scrape_huggingface_stackexchange.py` is the working fallback (same output shape).
