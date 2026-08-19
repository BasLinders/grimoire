# Training Resume Plan — New Fine-Tune Data Sources

Picks up after [PR #175](https://github.com/BasLinders/grimoire/pull/175)
(merged), which added `scripts/scrape_stackexchange.py` (any Stack Exchange
site, not just rpg.SE) and `scripts/generate_open5e_qa.py` (template-based,
code-verified Q&A from Open5e monster/spell fields). Neither has been run
yet — no new data exists on disk, only the scripts to produce it. This
document is the checklist for turning them into an actual updated Saga
checkpoint, in order.

See [expansion_PLAN.md](expansion_PLAN.md) for the corpus/weighting work this
builds on and [PLAN.md](PLAN.md)'s Phase 2 item 5 for the LoRA implementation
referenced in step 3.

## Step 1 — Run the new scrapers

Commands only — these hit the network and download multi-hundred-MB dumps,
run them yourself rather than through an agent session:

```bash
python scripts/scrape_stackexchange.py --site history
python scripts/scrape_stackexchange.py --site travel
python scripts/scrape_stackexchange.py --site skeptics
python scripts/generate_open5e_qa.py --output data/finetune/open5e_qa.jsonl
```

The three `--site` picks are a starting default (factual-explainer,
practical-advice, evidence-reasoning registers), not a fixed list — any slug
from stackexchange.com/sites works except Stack Overflow itself (split
dump). Add or swap sites here if a different conversational register seems
more valuable once the first batch is evaluated (step 4).

`generate_open5e_qa.py` now defaults to `--document-slug wotc-srd` (the
official 5e SRD) — added after a real run of the command above produced
23,210 examples, 46% of which belonged to a question asked more than once
with directly contradictory answers (e.g. "What is the challenge rating of
the Aboleth?" answered as both CR 11 and CR 10), because Open5e blends the
official document with unrelated third-party rulesets under the same
endpoints and the script had no filter. If `data/finetune/open5e_qa.jsonl`
on disk predates this fix, regenerate it before step 2.

## Step 2 — Build and combine fine-tune JSONL

```bash
python scripts/build_finetune_data_from_qa.py \
    --corpus-dir data/corpus/general_qa/ \
    --pattern    "*_se_*.txt" \
    --output     data/finetune/general_se_qa.jsonl

# Regenerate the existing D&D Q&A data too if data/finetune/saga_se_qa.jsonl
# isn't already present locally (data/ is gitignored, so a fresh clone or
# worktree won't have it) — see scripts/build_finetune_data_from_qa.py's own
# docstring for the saga_se_qa_source/ --corpus-dir and --min-score 1 flags
# that produced the currently-shipped checkpoint.

cat data/finetune/general_se_qa.jsonl \
    data/finetune/open5e_qa.jsonl \
    data/finetune/saga_se_qa.jsonl \
    > data/finetune/combined_v1.jsonl

python scripts/validate_finetune_data.py \
    --data  data/finetune/combined_v1.jsonl \
    --vocab data/tokenizer/bpe.json
```

Sanity-check the mix before training: print a few counts per source
(`wc -l` on each input file) so the combined dataset's D&D-vs-general ratio
is a deliberate choice, not an accident of whatever `cat` order was used.

## Step 3 — Open decision: full fine-tune or LoRA

The currently-shipped checkpoint
(`checkpoints/finetune/saga-se-qa-weighted-clean-v2/step_0007288.pt`, per
`agents.json`) was produced by **full fine-tuning**
(`scripts/finetune_saga.py`), not LoRA — despite LoRA being fully
implemented and marked done in `PLAN.md`'s Phase 2 item 5. That item's
stated rationale for LoRA was regularizing against catastrophic forgetting
on a *small* dataset (29–36 hand-authored examples). `combined_v1.jsonl`
from step 2 is several orders of magnitude larger, which weakens that
specific rationale — full fine-tuning on a large, diverse dataset is less
prone to catastrophic forgetting in the first place. This is a real decision
to make before training, not a default to skip past:

- **Full fine-tune** (`scripts/finetune_saga.py` or
  `python -m grimoire_ai.llm.training.finetune` with `--lora-rank 0`):
  continues the existing checkpoint lineage, simplest to compare directly
  against `saga-se-qa-weighted-clean-v2`.
- **LoRA** (`python -m grimoire_ai.llm.training.finetune --lora-rank 8`):
  produces a small `.lora` file (`agents.json`'s `lora_path` field) layered
  on top of a *pre-trained* (not previously fine-tuned) checkpoint instead of
  baking into the weights — cheaper to iterate on, easy to swap out if a
  batch of new source data turns out to hurt quality, matches the
  agent-per-persona design `AgentRegistry`/`AgentRouter` already assume for
  future non-Saga agents.

Recorded here so the choice doesn't get made implicitly by whichever script
happens to be copy-pasted from a previous run.

## Step 4 — Train

Full fine-tune (same shape as the run that produced the current checkpoint):

```bash
python scripts/finetune_saga.py \
    --checkpoint checkpoints/pretrain/<weighted_clean checkpoint>.pt \
    --vocab      data/tokenizer/bpe.json \
    --data       data/finetune/combined_v1.jsonl \
    --output-dir checkpoints/finetune/saga-combined-v1/ \
    --total-steps <scale with dataset size — see finetune_saga.py's docstring>
```

LoRA, if step 3 goes that way:

```bash
python -m grimoire_ai.llm.training.finetune \
    --resume     checkpoints/pretrain/<weighted_clean checkpoint>.pt \
    --vocab      data/tokenizer/bpe.json \
    --data       data/finetune/combined_v1.jsonl \
    --output     checkpoints/lora/saga-combined-v1/ \
    --lora-rank  8 \
    --total-steps <scale with dataset size>
```

`--total-steps` needs to scale with `combined_v1.jsonl`'s size the same way
`expansion_PLAN.md` scaled it for the rpg-only data (300 steps was tuned for
~30 examples; 7,288 steps for 77,740) — recompute the ratio for the actual
combined example count rather than reusing either number directly.

## Step 5 — Evaluate against the current production checkpoint

Same methodology as `expansion_PLAN.md`'s `weighted` vs `weighted_clean`
comparison — a single aggregate metric isn't enough to catch a regression
hiding in one slice of the data:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/finetune/saga-combined-v1/<final>.pt \
    --vocab      data/tokenizer/bpe.json \
    --quiz-repetition-penalty 1.3
```

Run the same command against the current production checkpoint for a
side-by-side (perplexity, BPC, retrieval hit-rate, quiz pass-rate,
kw-recall, token-F1). Two specific things to check given past history in
this repo, not just the aggregate numbers:

- **Degenerate collapse** (question-echoing, `does does does...`-style
  repetition loops) — this exact failure mode shipped to production once
  before (`expansion_PLAN.md`'s `--accepted-only` finding) and passed
  unnoticed until a qualitative check caught it. Sample generations on a
  handful of prompts by hand before trusting the aggregate metrics.
- **D&D fact recall on the up-weighted tier** — `expansion_PLAN.md` found a
  real (if narrow) CR/XP-recall regression between two prior checkpoints
  that the aggregate per-tier loss didn't fully explain. Diluting the
  fine-tune mix with general (non-D&D) data changes the question/answer
  ratio the model sees during this phase; confirm D&D-specific quiz
  questions didn't get worse as a side effect of the general-data addition.

### `saga-combined-v1` vs. production (2026-08-16)

Quiz: `saga-combined-v1` pass-rate 24.5%, kw-recall 13.61%, token-F1
0.1748. Production (`saga-se-qa-weighted-clean-v2`) pass-rate 22.4%,
kw-recall 12.24%, token-F1 0.1961 — a mixed result, `combined-v1` ahead
on two metrics, production ahead on F1.

Investigated both the degenerate-collapse rate and the token-F1 gap.

**Token-F1 gap**: root cause found and fixed. Plain unigram F1 assumes the
prediction is already a short extracted span (the standard SQuAD
assumption); this quiz scores full free-form generations against it
instead, producing three symptoms — precision penalized response length
regardless of correctness, punctuation-stripping collapsed `"+3"`/`"-3"`
and fragmented `"1/4"` into `"1"`+`"4"`, and restating the question's own
wording bought free overlap credit. Replaced with best-matching-window
F1: search all contiguous windows of the response for the best-scoring
span against the reference instead of scoring the whole response,
exclude vocabulary shared with the question from the overlap count on
both sides, and keep signs/fractions/decimals/possessives intact in the
tokenizer (`grimoire_ai/llm/eval/quiz.py`, 2026-08-16). Fully
agent-agnostic — no D&D-specific logic anywhere in the fix, so it applies
to any future agent's quiz eval, not just Saga's.

**Re-verified against the real quiz eval, twice, under different
retrieval conditions** — before the fix, token-F1 and keyword-recall
*disagreed* on which checkpoint was better (F1 favored production,
kw-recall favored `combined-v1`, the numbers quoted above). After the
fix:
- No `--corpus-dir`: `combined-v1` pass-rate 26.5%, kw-recall 13.95%,
  F1 0.1033; production pass-rate 20.4%, kw-recall 11.22%, F1 0.1005 —
  both metrics now agree, favoring `combined-v1`.
- `--corpus-limit 200` (same 200 sampled files for both checkpoints,
  fixed seed): `combined-v1` pass-rate 6.1%, kw-recall 3.06%, F1 0.1463;
  production pass-rate 10.2%, kw-recall 5.10%, F1 0.1606 — both metrics
  again agree, this time favoring production. (Pass-rate/kw-recall are
  much lower in this run for both checkpoints — a random 200/1469-file
  sample gives poor retrieval hit-rate, 10-15%, not a checkpoint
  regression; see `docs/corpus_index_scaling.md` for why `--corpus-limit`
  exists at all.)

Which checkpoint "wins" flips between the two post-fix conditions — that
reflects the checkpoints' differing sensitivity to retrieval context, a
separate question from whether the metric agrees with itself, which it
now consistently does. Moved out of `known_bugs.md` (2026-08-16).

**Degenerate collapse**: root cause found and fixed. `repetition_penalty`
is a flat, non-escalating discount (standard CTRL-style penalty) that a
confident-enough model can override indefinitely — inherent to the
penalty type, not a bug. `RepetitionLoopGuard` (a hard structural ban,
already implemented, already wired into `grimoire-chat`/the Chat tab)
fixes it: quantified before/after on `combined-v1` at
`repetition_penalty=1.3`, 5 seeds × 12 prompts, showed severe collapses
drop from 6/60 (10%) to 0/60 (0%) with `loop_guard_max_repeats=3`, no
speed cost. **Deployed**: added `loop_guard_max_repeats: 3` /
`loop_guard_max_period: 4` to `agents.json`'s `saga.gen_config`
(2026-08-16) — no code change needed, the config-loading path already
passed these through. Re-verified via the actual quiz eval
(`scripts/evaluate.py --quiz-loop-guard-max-repeats 3`, now supported):
pass-rate and kw-recall unchanged for both checkpoints (quiz uses greedy
decoding, which loops far less often than the stochastic sampling used
in the 5-seed test), but the guard still fired on 7/49 questions for
`combined-v1` and 1/49 for production — every triggered case was a
genuine severe collapse (`"to to to to..."`, `"own own own..."`,
`"prevent prevent..."`) successfully broken, confirmed by diffing
responses before/after. This is also a real, reproducible (non-seed-
dependent) difference favoring production's stability under greedy
decoding specifically — 7/49 vs. 1/49 — worth keeping in mind
alongside the earlier tied stochastic-sampling result, not a
contradiction of it (different decoding strategy, different question).

## Step 6 — Ship it

If the new checkpoint clears step 5 without regressions:

- Full fine-tune: update `agents.json`'s `saga.checkpoint` to the new path.
- LoRA: update `agents.json`'s `saga.checkpoint` to the *pre-trained* base
  checkpoint and set `saga.lora_path` to the new `.lora` file.

Record the comparison numbers here or in `expansion_PLAN.md` (whichever this
session is extending) before moving on, the same way every prior checkpoint
swap in `expansion_PLAN.md` is logged with its evaluation numbers — the
checkpoint-swap history is only useful if every swap is traceable to the
evidence that justified it.

### `general-expansion-v1` ships, replaces production (2026-08-19)

Full fine-tune (`checkpoints/finetune/general-expansion-v1/step_0013213.pt`,
13,213 steps on `combined_v2.jsonl`, 140,945 examples) off the
general-content-expansion pretrain checkpoint
(`checkpoints/pretrain/general_expansion_v1/step_0015259.pt`, 324.2M
tokens, ~65% of the small-25M Chinchilla target — see
`expansion_PLAN.md`'s "General-content expansion" section). `agents.json`'s
`saga.checkpoint` updated to point at it, replacing
`saga-se-qa-weighted-clean-v2` (the long-standing production checkpoint).

**Quiz eval, 5 seeds, production-matching sampling (temperature 0.8/top_k
50/top_p 0.9)**: `general-expansion-v1` pass-rate 16.3%, kw-recall 10.7%,
token-F1 0.175, vs. production's 15.1%/9.3%/0.174 and `combined-v1`'s
17.6%/10.5%/0.161. Every pairwise difference is smaller than each
checkpoint's own seed-to-seed noise (σ≈1-3pp on pass-rate/kw-recall) —
statistically tied, not a clean numeric win. Expected: this quiz measures
narrow D&D factual recall, which this session's strategy deliberately
deprioritized in favor of general conversational capability, with
retrieval (not pretrain memorization) carrying D&D precision instead.

**Qualitative (`compare_checkpoints.py` vs. production, 5 seeds, 12
prompts)**: the severe Stack-Exchange-answer register drift found in the
pretrain-only qualitative check (near-universal — simulated forum
threads, new questions posed mid-completion, personal anecdotes, even on
narrative/definitional prompts that should've resisted it) is
substantially fixed by fine-tuning. Not fully eliminated — a milder
residual tic (self-referential "in this answer"/"I'm going to answer
this by" framing) survives in ~4-5 of 50 completions; see
`known_bugs.md`. Overall factual coherence is roughly on par with
production — neither checkpoint reads as more reliable on this pass.

**Decision**: shipped on this evidence — comparable-or-tied quantitative
result plus a real, if partial, fix to the specific concern flagged after
pretraining (the register drift) was judged sufficient, rather than
chasing a clean numeric win that the quiz metric was never going to show
given what this session optimized for.
