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

- [x] **Degenerate collapse, checked properly.** The first side-by-side
      (`scripts/compare_checkpoints.py`, 6 prompts, no seed control) looked
      like a clean win for `combined-v1` — production echoed a question
      verbatim and hit a repetition loop on 4 of 6 prompts, `combined-v1`
      showed neither. That result didn't hold up: the script had no RNG
      seed, so every run sampled differently. Fixed (seed now resets before
      every prompt, not just once) and reran across 5 seeds × 12 prompts
      (120 responses total, counting only severe collapses — 5+ consecutive
      repeated tokens, or total breakdown): **7/60 (11.7%) for each
      checkpoint.** Exactly tied, and which checkpoint looked worse flipped
      from seed to seed with no consistent pattern. Conclusion: no
      difference between these two checkpoints on collapse frequency; the
      single-run result was sampling variance, not a real property of
      either one.
- [x] **Token-F1 gap, explained.** Not a corpus/training quality
      difference — a mechanical artifact of the metric interacting with a
      real behavioral difference. `combined-v1` averages 60.4 response
      tokens vs. production's 46.0 (both against ~15-token reference
      answers); within-checkpoint, response length correlates negatively
      with F1 (`combined-v1`: -0.545, production: -0.417) — SQuAD-style
      token-F1's precision term is `correct_tokens / len(entire_response)`,
      so a longer-but-equally-correct answer scores lower purely from
      length. `combined-v1` also hit/neared the 128-token generation cap on
      4/49 quiz questions; production hit it on 0/49 — `combined-v1` has a
      real tendency to keep generating rather than concluding naturally.
      keyword_recall (length-insensitive) favors `combined-v1` (13.61% vs
      12.24%), consistent with it containing at least as much correct
      content, just wrapped in more verbosity that token-F1 penalizes.
- [ ] **Follow-up, not yet investigated: why doesn't `repetition_penalty`
      fully suppress repetition loops?** Both checkpoints were run at
      `repetition_penalty=1.3` throughout this evaluation and both still
      produced multi-token hard loops (up to ~150 consecutive repeats) on
      a real fraction of prompts. `--loop-guard`
      (`RepetitionLoopGuard`, a hard structural ban rather than a soft
      logit discount) exists in `grimoire_ai/llm/inference/constrained_decoding.py`
      and is wired into `grimoire-chat`/the Chat tab, but wasn't used in
      any of this session's evaluation scripts
      (`compare_checkpoints.py`, `qualitative_check.py`, `evaluate.py`'s
      quiz eval). Possibly connects to the same "won't stop generating"
      tendency behind the token-F1 finding above -- worth checking
      together, not as two separate investigations. Flagged here to pick
      up next session, not investigated yet.

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
