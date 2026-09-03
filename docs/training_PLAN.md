# Training Resume Plan — New Fine-Tune Data Sources

Picks up after [PR #175](https://github.com/BasLinders/grimoire/pull/175)
(merged), which added `scripts/scrape/scrape_stackexchange.py` (any Stack Exchange
site, not just rpg.SE) and `scripts/finetune/generate_open5e_qa.py` (template-based,
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
python scripts/scrape/scrape_stackexchange.py --site history
python scripts/scrape/scrape_stackexchange.py --site travel
python scripts/scrape/scrape_stackexchange.py --site skeptics
python scripts/finetune/generate_open5e_qa.py --output data/finetune/open5e_qa.jsonl
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
python scripts/finetune/build_finetune_data_from_qa.py \
    --corpus-dir data/corpus/general_qa/ \
    --pattern    "*_se_*.txt" \
    --output     data/finetune/general_se_qa.jsonl

# Regenerate the existing D&D Q&A data too if data/finetune/saga_se_qa.jsonl
# isn't already present locally (data/ is gitignored, so a fresh clone or
# worktree won't have it) — see scripts/finetune/build_finetune_data_from_qa.py's own
# docstring for the saga_se_qa_source/ --corpus-dir and --min-score 1 flags
# that produced the currently-shipped checkpoint.

cat data/finetune/general_se_qa.jsonl \
    data/finetune/open5e_qa.jsonl \
    data/finetune/saga_se_qa.jsonl \
    > data/finetune/combined_v1.jsonl

python scripts/finetune/validate_finetune_data.py \
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
(`scripts/train/finetune_saga.py`), not LoRA — despite LoRA being fully
implemented and marked done in `PLAN.md`'s Phase 2 item 5. That item's
stated rationale for LoRA was regularizing against catastrophic forgetting
on a *small* dataset (29–36 hand-authored examples). `combined_v1.jsonl`
from step 2 is several orders of magnitude larger, which weakens that
specific rationale — full fine-tuning on a large, diverse dataset is less
prone to catastrophic forgetting in the first place. This is a real decision
to make before training, not a default to skip past:

- **Full fine-tune** (`scripts/train/finetune_saga.py` or
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
python scripts/train/finetune_saga.py \
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
python scripts/eval/evaluate.py \
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
(`scripts/eval/evaluate.py --quiz-loop-guard-max-repeats 3`, now supported):
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

## Step 7 — RepetitionLoopGuard's templated-loop gap (fixed, 2026-08-21)

`known_bugs.md` had flagged that `RepetitionLoopGuard`'s exact-repeat
check (`max_repeats`/`max_period`) missed a *templated* loop — the
surrounding structure repeats but a substituted value changes each cycle
(e.g. `saga-combined-v1`, seed 4, "armor class of a Goblin": `CR = 10 +
Dex bonus. CR = 14 + Str bonus. CR = 18 + Con bonus + Int bonus...`,
continuing for ~20 cycles even with the exact guard active).

**Fix**: `RepetitionLoopGuard` (`grimoire_ai/llm/inference/constrained_decoding.py`)
gained a `template_match_ratio` parameter (default `1.0`). Instead of
requiring a whole repeating block to match verbatim, it now checks
*position by position* across the trailing cycles whether a slot holds
the same token in every cycle ("invariant"). If the position that would
open the next cycle (the recurring anchor — "CR" in the example above) is
itself invariant, and the fraction of invariant positions overall meets
`template_match_ratio`, that anchor token is banned from opening another
cycle — the genuinely-varying slots (the substituted number/word) are
never touched. At `template_match_ratio = 1.0` this is exactly the
original exact-repeat check (all pre-existing `RepetitionLoopGuard` tests
pass unchanged); lowering it (e.g. `0.6`) additionally catches templated
loops. Threaded through `GenerationConfig`, `cli/chat.py`,
`compare_checkpoints.py`, and the quiz eval harness/`evaluate.py` alongside
the existing `loop_guard_max_repeats`/`loop_guard_max_period` flags.

**Verified against the real bug, same checkpoint/prompt/seed that found
it** (`saga-combined-v1`, `step_0011339.pt`, seed 4, "armor class of a
Goblin", `compare_checkpoints.py`, both runs otherwise identical):

- Old exact-only guard (`--loop-guard-max-repeats 3 --loop-guard-max-period 4`):
  reproduced the original collapse verbatim — `CR = 10 + Dex bonus. CR =
  14 + Str bonus. CR = 18 + Con bonus + Int bonus. CR = 15 + Dex
  bonus...` continuing for 20 cycles to the `max_new_tokens` cutoff.
- New guard (`--loop-guard-max-repeats 3 --loop-guard-max-period 16
  --loop-guard-template-match-ratio 0.6`): stopped after exactly 2 cycles
  — `CR = 10 + Dex bonus. CR = 14 + Str bonus` — then diverged into
  unrelated text (`(if any) +1 (for the armor class description says
  it...)`) instead of starting a 3rd repeat. The rest of the 12-prompt
  set was byte-identical between the two runs, confirming the guard only
  intervenes where it should.

Landed via [PR #204](https://github.com/BasLinders/grimoire/pull/204).
Moved out of `known_bugs.md`.

**Shipped to production (2026-08-21)**: `agents.json`'s `saga.gen_config`
updated to `loop_guard_max_period: 16` /
`loop_guard_template_match_ratio: 0.6` (from `max_period: 4`, no ratio ⇒
`1.0`). Checked for regressions with the same before/after quiz-eval
methodology as the checkpoint swaps above, 5 seeds each, temperature 0.8/
top_k 50/top_p 0.9/repetition_penalty 1.3 (matching `agents.json`):

| Seed | Old (max_period=4) pass / kw-recall / F1 | New (max_period=16, ratio=0.6) pass / kw-recall / F1 |
|---|---|---|
| 0 | 18.4% / 10.54% / 0.2212 | 18.4% / 10.54% / 0.2212 |
| 1 | 22.4% / 14.63% / 0.2163 | 22.4% / 14.63% / 0.2163 |
| 2 | 24.5% / 14.97% / 0.2370 | 24.5% / 14.97% / 0.2370 |
| 3 | 20.4% / 12.24% / 0.2191 | 20.4% / 12.24% / 0.2191 |
| 4 | 28.6% / 16.33% / 0.2488 | 28.6% / 16.33% / 0.2488 |

Every metric is identical to four decimal places, seed for seed — not
just "close," but byte-identical generation output, which only happens if
the loop guard's mask never actually fired in any of these 245
generations (5 seeds × 49 questions) under either config. Expected, not a
test artifact: the quiz eval caps responses at 128 tokens
(`quiz_gen_config`'s `max_new_tokens=128` in `harness.py`) vs.
`compare_checkpoints.py`'s 256-token default used to find and verify the
bug above, and the `saga_quiz.jsonl` Q&A prompts don't match the shape of
the free-form "describe X" prompt that produced the original collapse —
there just wasn't room or occasion for this specific templated pattern to
establish itself here. This confirms **no regression** on the standard
quiz set from widening the period and lowering the ratio (no new false-
positive bans creeping into otherwise-fine generations); it does not
re-demonstrate the fix itself — that's the `compare_checkpoints.py`
before/after against the real bug repro, already recorded above.

## Step 8 — `general_expansion_v2`/`v3`: corpus rebalancing attempt, negative result (2026-08-21 to 2026-08-23)

Follow-up to the round-2 general-content scrape (`expansion_PLAN.md`'s
"General-content expansion" section, now at 9 Stack Exchange sites,
470M tokens, ~94% of Chinchilla-optimal). Goal: retrain on the larger
corpus with a deliberate `--weight-pattern` scheme instead of letting
new content fall into the accidental `*:1.75` catch-all the way round 1
did, and see whether that fixes `general-expansion-v1`'s residual
Stack-Exchange-answer register-drift tic (`known_bugs.md`).

**`general_expansion_v2` (first attempt, superseded, not shipped)**:
pretrained on the round-2 corpus with a 4-tier weight scheme favoring
D&D-adjacent conversational content (`rpg_se_*`/`worldbuilding_se_*`/
`gaming_se_*`/`boardgames_se_*` at `1.75`, general-unrelated Q&A at
`1.25`, D&D reference material at `1.25`, narrative bulk at `0.5`).
Qualitative check on 6 raw pretrain prompts found **two problems**:

1. A literal `## Answer (score: N)` markup leak — the new `general_qa/`
   scrape (via the `scrape_huggingface_stackexchange.py` fallback, since
   archive.org is blocked on this network) had never had
   `clean_stackexchange_markup.py` applied, unlike the original
   `rpg_se_*` data. Confirmed systemic: 2,092/2,092 files in
   `general_qa/` still had the scaffolding.
2. Every one of the 6 samples showed pervasive first-person forum-answer
   voice ("I remember from experience...", "I'm not sure whether your
   intent here is correct...") — worse than `general-expansion-v1`'s
   documented ~4-5/50 residual. The weight scheme put general +
   D&D-adjacent Q&A at ~93% of effective sampling exposure (computed
   from `eval_per_tier.py`'s window counts), higher than round 1's.

**Fix attempt → `general_expansion_v3`**: cleaned `general_qa/` for real
(backed up to `general_qa_source/`), rebuilt `corpus.bin`, and revised
the weight scheme twice to pull Q&A dominance down — first to `~86.3%`
effective exposure (D&D-adjacent `1.75→1.5`, general-unrelated `1.25`
unchanged), then more decisively to `~75.4%` (D&D-adjacent `1.5→1.0`,
general-unrelated `1.25→0.75`, narrative bulk `0.5→0.75`, D&D reference
`1.25→1.5`). Retrained (15,259 steps, same Chinchilla-derived step count
as `v1`/`v2` — see `docs/PARAM_OPT.md`, `total_steps` depends on model
size, not corpus size).

**Pretrain-only qualitative check on `v3`**: a real, clean win — the
`## Answer` leak was gone (markup fix confirmed), and the pervasive
forum-voice tic was **gone from all 6 samples** (0/6, down from 6/6 on
`v2`). New, milder issue observed instead: `mechanics`/`cantrip`
completions started emitting raw stat-block/spell-block markdown syntax
verbatim (`# Monster Revival`, `## Traits`, `Category: spell`, `School:
... | Casting time: ...`) — the same *kind* of bug (literal source
scaffolding reproduced instead of prose), now from `5etools_*`/
`open5e_*` files since their weight went up. Noted but not blocking.

**Fine-tuned `v3`** on the same recipe as `v1` (full fine-tune,
`combined_v3.jsonl`, 140,945 examples — `general_se_qa.jsonl`
regenerated from the cleaned 9-site corpus via
`data/corpus/general_qa_source/` since the cleaned files are no longer
parseable by `qa_pairs.py`, downsampled back to 60,000 to hold the
established general:D&D fine-tune ratio steady; `saga_se_qa.jsonl`/
`open5e_qa.jsonl` unchanged, D&D-specific data untouched by any of this
round's changes).

**Result: negative.** The pretrain-level fix did not survive fine-tuning:

- **Quiz eval** (5 seeds, `agents.json`-matching sampling): `v3`
  pass-rate 16.7% / kw-recall 10.1% / token-F1 0.219, vs. production
  (`general-expansion-v1`) 22.9% / 13.7% / 0.229 — consistently worse
  across all 5 seeds on all 3 metrics, more than the ~3.5-4.7pp
  within-checkpoint seed spread would explain on its own.
- **Qualitative comparison** (5 seeds, 12 prompts, `compare_checkpoints.py`
  vs. production): the forum-answer-voice tic that was cleanly absent at
  the pretrain-only stage **reappeared in both checkpoints at similar
  density** after fine-tuning, including production's own previously-
  documented exact phrasing ("As you can see in this answer, I would
  suggest...", armor class prompt) resurfacing verbatim. No clear
  coherence advantage for `v3` over production either way — both show
  comparable rates of muddled facts and fabricated citations.

**Diagnosis**: `v1` and `v3` share almost the same fine-tune data shape
— `saga_se_qa.jsonl` (77,740), `open5e_qa.jsonl` (3,205), and a 60,000-
example general sample, all 100% context→answer Q&A pairs regardless of
which registers fed the *pretrain* corpus. Per this project's own stated
model ("fine-tuning teaches conversation behaviour"), the fine-tune
stage is where response voice is learned most directly — and if the
entire fine-tune mix is Q&A-shaped, the model adopts that register
regardless of how the pretrain corpus was weighted. The pretrain-level
fix was real and independently confirmed; it just doesn't propagate
through a fine-tune stage that reintroduces the same register from a
different (fine-tune-data) angle.

**Decision: did not ship `v3`.** Reverted to `general-expansion-v1` as
production (`agents.json` unchanged throughout this experiment).
`general_expansion_v2`'s checkpoints and the superseded pretrain
lineage (`baseline`/`weighted`/`weighted_clean`/`weighted_clean_v2`/
`weighted_clean_v3`) and fine-tune lineage (`saga-se-qa-clean-v2`,
`saga-se-qa-weighted-clean`, `saga-se-qa-weighted-clean-v2`,
`saga-combined-v1`, `base-294-9`) were deleted from local disk (~29.3GB
freed) once `v3` was confirmed not worth keeping either.

**Practical implication for next time**: if the register-drift tic is
worth pursuing further, the lever is more likely the *fine-tune data's
format* — e.g. blending in non-Q&A-shaped conversational examples,
closer to `scripts/finetune/data/general_conversations.jsonl`'s original
64-example set (`PLAN.md`'s LoRA item) — rather than further pretrain
corpus reweighting, which this round showed doesn't survive fine-tuning
on an all-Q&A dataset. Not attempted this round; flagged for a future
session.

## Step 9 — Fine-tune-data dehedging: a real, low-cost win (2026-08-23)

Direct follow-up to Step 8's diagnosis. Instead of touching the pretrain
corpus again, this targets the fine-tune data's *format* directly: a
large share of the SE-answer register tic is a handful of recurring
meta-commentary openers ("I would say that...", "As you can see in this
answer...", "You are correct that...") stapled onto the front of
otherwise-fine answers. `scripts/finetune/dehedge_finetune_data.py` (new,
[PR #211](https://github.com/BasLinders/grimoire/pull/211)) strips these
deterministically via regex — no LLM call, no invented content, same
philosophy as `clean_stackexchange_markup.py` and
`generate_open5e_entigraph.py`. Deliberately conservative: only strips
at the very start of a response, never mid-paragraph; leaves genuine
epistemic hedges ("I'm not sure whether...") untouched since they carry
real meaning; skips any example where stripping would leave a dangling
leading comma (a parenthetical/appositive sat between the hedge and its
`that`) or fewer than 20 characters remaining, rather than emit broken
output. Both edge cases were caught via real `--dry-run` output before
trusting the pattern list, not assumed correct from the regex alone.

Both experiments fine-tuned off `general_expansion_v1`'s pretrain
checkpoint (not `v3`) — isolating the fine-tune-data variable cleanly
after Step 8 showed pretrain-level changes don't survive fine-tuning
anyway, so bundling them again would have muddied attribution a second
time.

**`general-expansion-v1-dehedged`** (general-content Q&A dehedged only,
342/60,000 examples changed — 0.6% — `saga_se_qa.jsonl`/`open5e_qa.jsonl`
untouched): quiz eval (5 seeds) landed statistically tied with
production — pass-rate 21.6% vs. 22.9%, kw-recall 13.74% vs. 13.74%
(identical), token-F1 0.2231 vs. 0.2285, all within the ~3.5-4.7pp
seed-to-seed noise band. **Qualitative** (5 seeds, 12 prompts): the
targeted hedge phrases showed up meaningfully less often than production
(~7-8 occurrences across 60 completions vs. ~12+, including production's
own previously-documented exact phrasing — "As you can see in this
answer, I would suggest..." — reappearing verbatim again on the same
prompt) but not eliminated, since only 0.6% of the general-content
subset was touched and the model generalizes the pattern beyond the
specific rewritten examples. **Verdict: a real, if partial, win at
effectively zero cost.**

**`general-expansion-v1-dehedged-v2`** (extended the same pass to
`saga_se_qa.jsonl` too, 469/77,740 examples changed — 0.6%, same rate):
quiz eval dropped further below production — pass-rate 18.0%, kw-recall
11.7%, token-F1 0.2293 — a real regression (5/5 seeds below every one of
production's 5 seeds on pass-rate, not just noise), plausibly because
`saga_se_qa.jsonl` is exactly the data driving D&D factual recall, which
this quiz measures directly, so even a conservative rewrite there risked
disturbing phrasing the model had anchored specific facts to.
**Qualitative**: tic frequency was comparable to (not clearly better
than) the general-only version — several targeted phrasings still slipped
through ("I've been toying the spell out" vs. the pattern list's
narrower "I've been to/in a similar situation"; "I've been in a similar
situation" itself appeared twice across the 60 completions). **Verdict:
net negative — real quiz-score cost, no corresponding qualitative gain
to justify it.**

**Decision: keep `general-expansion-v1-dehedged` as the best result of
this session's register-drift work; discard `-v2`.**

**Full harness confirmation (2026-08-24)**: `scripts/eval/evaluate.py` with
perplexity + retrieval + quiz together (`--corpus-bin
data/processed/corpus.bin`, `--corpus-dir data/corpus/saga/
--corpus-limit 200`, seed 0) against both `general-expansion-v1-dehedged`
and production:

| | `dehedged` | production |
|---|---|---|
| Perplexity / BPC | 88.72 / 6.4711 | 99.69 / 6.6394 |
| Retrieval hit-rate | 0.0% (0/20) | 0.0% (0/20) |
| Quiz pass-rate / kw-recall | 8.2% / 4.76% | 6.1% / 4.42% |

Retrieval hit-rate 0% for *both* is a known `--corpus-limit 200`
sampling artifact already documented earlier in this file (a random
200/1469-file sample rarely contains the right passage for a fixed
20-query set) — tied at zero, no differential signal. The grounded quiz
numbers are correspondingly depressed for both relative to the 5-seed
ungrounded comparison above, for the same reason (irrelevant retrieved
context sometimes actively hurting rather than helping); the ordering
(`dehedged` ahead, 8.2% vs. 6.1%) is at least consistent with, not
contradicting, the earlier tied-to-slightly-favoring result. Perplexity/
BPC is the one genuinely new, uncontaminated signal here, and it favors
`dehedged` — fine-tuning on the dehedged data didn't cause any extra
drift away from the base language-modeling distribution; if anything
slightly less than production's fine-tune did.

**No red flags found.**

### `general-expansion-v1-dehedged` ships, replaces `general-expansion-v1` (2026-08-24)

`agents.json`'s `saga.checkpoint` updated to
`checkpoints/finetune/general-expansion-v1-dehedged/step_0013213.pt`,
replacing `general-expansion-v1`. `gen_config` unchanged — this swap is
purely a fine-tune-data-format change (Step 9's dehedge pass), not a
pretrain or decoding-parameter change, so nothing else needed touching.

Shipped on the evidence above: quiz-eval parity (kw-recall identical,
pass-rate/F1 within seed noise), a real reduction in the SE-answer-voice
register-drift tic across a 5-seed qualitative pass, better perplexity/
BPC on the shared pretrain corpus, and no regression anywhere in the
full harness (perplexity/retrieval/quiz together). The lowest-risk
production swap of this session — no pretrain change, a 0.6%-of-subset
deterministic edit to existing fine-tune data, verified before/after on
every axis this project checks before shipping.

`general-expansion-v1` (fine-tune) and `general-expansion-v1-dehedged-v2`
(the discarded extended-dehedge experiment) are now superseded;
`general_expansion_v1`/`general_expansion_v3` (pretrain) stay, since
`-dehedged`'s lineage traces back to `general_expansion_v1`'s pretrain
checkpoint and `v3` is still the reference point for Step 8's finding.

## Step 10 — Widening the dehedge pattern list: diminishing returns (2026-08-25)

Direct follow-up to Step 9, testing whether the shipped dehedge win could
be extended further with no additional cost. Widened
`dehedge_finetune_data.py`'s pattern list ([PR #213](https://github.com/BasLinders/grimoire/pull/213)):
added coverage for hedge phrasings observed slipping through in Step 9's
qualitative pass ("I've been toying X out", "I'd rule that") plus a
batch of common generic hedge openers in the same pure-meta-commentary
category as what's already covered ("I guess"/"I suppose", "honestly",
"to be honest", "if you ask me", "my take is", "it seems to me").
Verified against real transcript examples and negative controls before
using it. Regenerated the general-content Q&A (388/60,000 examples
changed, up from 342 — a modest 13% increase), re-fine-tuned off the
same `general_expansion_v1` pretrain checkpoint
(`general-expansion-v1-dehedged-v3`), and compared directly against the
currently-shipped `general-expansion-v1-dehedged` (not raw production —
that's the actual bar to beat now).

**Quiz eval** (5 seeds): statistically tied with the shipped checkpoint
— pass-rate 21.2% vs. 21.6%, kw-recall 13.1% vs. 13.7%, token-F1 0.222
vs. 0.223, all within the established seed-to-seed noise band. No
regression from the wider pattern list.

**Qualitative** (5 seeds, 12 prompts): no clear improvement over the
already-shipped checkpoint — hedge-phrase frequency was roughly
comparable between the two (~5-7 occurrences per 60 completions in
both), not a further reduction. Both still show "I would have to say",
"I've been in a similar situation" (twice each), "I'd say", "You are
correct that" at similar rates.

**Diagnosis for why widening the list didn't help further**: this
dehedge approach only strips response-*initial* clauses, by design
(mid-paragraph stripping risks corrupting otherwise-fine content). Most
of what's left in these transcripts is either (a) hedges embedded
mid-sentence ("...at least, all others...", "if this ability says
otherwise...") that this design deliberately never targets, or (b) the
model generalizing the hedge *pattern* itself from the training examples
that remain, rather than reproducing specific removed phrasings verbatim
— so shrinking the literal-match surface has diminishing marginal
effect once the obvious, common phrasings are already covered.

**Decision: this approach has hit its practical ceiling.** Not shipped —
tied quiz score, no qualitative win to justify a swap.
`general-expansion-v1-dehedged` (Step 9's version) remains production.
Further pattern-list expansion isn't worth pursuing further on its own;
if the register-drift tic is worth chasing further, the more promising
untried lever is pairing a cleaned/reweighted pretrain corpus (Step 8's
`general_expansion_v3` scheme, whose pretrain-only qualitative check was
a clean 0/6 before an all-Q&A fine-tune undid it) with this dehedged
fine-tune data — a combination never tested, since Step 8 and Step 9
each isolated one variable at a time. That pretrain checkpoint was
deleted in this session's disk cleanup, so pursuing it means a full
pretrain rerun, not a quick follow-up.
