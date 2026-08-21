# Known Bugs

Outstanding, unresolved problems — as opposed to `expansion_PLAN.md`/
`training_PLAN.md`, which are narrative logs of decisions and history
(including plenty of *already-fixed* bugs). This is the short, scannable
list of what's still actually broken or misleading, so it doesn't have
to be reconstructed by reading through a session's worth of narrative
each time. Move an entry out once it's fixed — record the fix in
whichever plan doc is tracking that work, not here.

See also `docs/corpus_index_scaling.md` for a scoped architectural item
(not tracked here since it's a known scaling ceiling with a documented
mitigation, not an active bug).

## RepetitionLoopGuard doesn't catch templated (varying-content) loops

**Status:** fix implemented, not yet empirically validated against a real
checkpoint.

`RepetitionLoopGuard` (deployed in `agents.json`'s `saga.gen_config`,
`loop_guard_max_repeats: 3`) hard-bans the token that would extend an
*exact* repeating token sequence past N consecutive copies. It was built
and verified against literal repeats (`"to to to to..."`, `"does does
does..."`) and confirmed effective there.

Found via `scripts/compare_checkpoints.py` (5 seeds, temperature 0.8/
top_k 50/top_p 0.9, `--loop-guard-max-repeats 3 --loop-guard-max-period 4`
— i.e. the guard was active) 2026-08-16: both checkpoints produced a
different, structurally-repeating pattern the guard never caught, because
the *token sequence* never exactly repeats even though the *structure*
clearly does — a different value gets substituted each cycle:

- `saga-combined-v1`, seed 4, "armor class of a Goblin": `CR = 10 + Dex
  bonus. CR = 14 + Str bonus. CR = 18 + Con bonus + Int bonus. CR = 15 +
  Dex bonus...` — repeats the `CR = N + X bonus.` template roughly 15
  times with a different `N`/`X` substituted each time.
- production, seed 4, "difference between mean and median": `A means A -
  B means A - B, C means A - B means A - B and C = A means A - B mean A
  - B means A - B means B...` — same phenomenon, a template repeating
  with a substituted letter each cycle.

Both are clearly degenerate to a human reader, and both happened with
the guard enabled and configured the same way that successfully
eliminated the original exact-repeat collapse. This is a distinct
failure mode from the one already fixed, not a regression of it.

Practical implication: don't treat `loop_guard_max_repeats` as having
solved "repetition loops" as a category — it solved the exact-repeat
case specifically. A real fix needs some notion of structural/template
repetition (e.g. detecting a repeating skeleton with token spans masked
out), not just exact n-gram matching.

**Fix**: `RepetitionLoopGuard` (`grimoire_ai/llm/inference/constrained_decoding.py`)
now takes a `template_match_ratio` (default `1.0`, exact behaviour
unchanged). For each candidate period, it splits the trailing history into
`max_repeats - 1` cycles and checks *position by position* whether a
cycle's slot holds the same token across all cycles ("invariant"), rather
than requiring the whole cycle to match verbatim. If the position that
would open the next cycle (the recurring anchor, e.g. "CR" in "CR = 10 +
Dex bonus. CR = 14 + Str bonus...") is itself invariant, and the fraction
of invariant positions overall meets `template_match_ratio`, that anchor
token is banned from opening another cycle — the varying "slot" positions
(the substituted number/word) are never touched. At `template_match_ratio
= 1.0` every position must match, which is exactly the original
exact-repeat check (verified: all pre-existing `RepetitionLoopGuard` tests
pass unchanged). Lowering it (e.g. `0.6`) additionally catches templated
loops. Threaded through everywhere `loop_guard_max_repeats`/
`loop_guard_max_period` already were: `GenerationConfig`
(`loop_guard_template_match_ratio`), `cli/chat.py`
(`--loop-guard-template-match-ratio`), `compare_checkpoints.py`, and the
quiz eval harness/`evaluate.py` (`--quiz-loop-guard-template-match-ratio`).
Also raise `--loop-guard-max-period`/`--quiz-loop-guard-max-period` past
the `4`-token default when trying this — the observed templated loops
repeat at whole-sentence granularity (well beyond 4 tokens), not the short
phrase length the exact check was originally tuned for.

Covered by new unit tests in `tests/llm/test_constrained_decoding.py`
(anchor-invariant-but-slot-varies detection, non-firing when the anchor
itself is the varying position, and a match-ratio-below-threshold
non-firing case) — but **not yet re-run against a real checkpoint's
qualitative output** the way the original exact-repeat fix was (5-seed
`compare_checkpoints.py` before/after). Validate with:

`compare_checkpoints.py` applies one set of loop-guard flags to *both*
sides of a run (it's built for comparing two different checkpoints, not
two configs of the same one), so validating this means two separate runs
of the *same* checkpoint against itself, with and without the new flag,
and diffing the transcripts by hand:

```bash
python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/finetune/general-expansion-v1/step_0013213.pt --label-a same \
    --checkpoint-b checkpoints/finetune/general-expansion-v1/step_0013213.pt --label-b same \
    --vocab data/tokenizer/bpe.json --seed 0 \
    --loop-guard-max-repeats 3 --loop-guard-max-period 16 --loop-guard-template-match-ratio 0.6 \
    > with_guard.txt

python scripts/compare_checkpoints.py \
    --checkpoint-a checkpoints/finetune/general-expansion-v1/step_0013213.pt --label-a same \
    --checkpoint-b checkpoints/finetune/general-expansion-v1/step_0013213.pt --label-b same \
    --vocab data/tokenizer/bpe.json --seed 0 \
    > without_guard.txt
```

Try a few seeds, and specifically re-check the two transcripts quoted
above (or fresh ones showing the same templated-loop shape) before moving
this out of `known_bugs.md`.

## Occasional malformed-UTF-8 replacement glyph in generated text

**Status:** root cause understood, likely not code-fixable — a model
calibration signal, not a decode bug. Documented for awareness, not
necessarily actionable.

`BPETokenizer.decode()` (`grimoire_ai/llm/tokenizer/bpe.py:499-539`)
accumulates every predicted token's raw bytes into one buffer across the
whole decode call, then decodes once with `bytes.decode("utf-8",
errors="replace")` — the standard Python fallback for malformed UTF-8,
which substitutes the U+FFFD replacement character. Bytes are fully
accumulated before decoding (not decoded incrementally per-token), so
this isn't a buffering bug splitting a multi-byte character across
token boundaries — the fallback firing means the model itself predicted
a byte-level token sequence that doesn't reconstruct valid UTF-8,
almost certainly for multi-byte punctuation (curly quotes/apostrophes
are 3 UTF-8 bytes each and easy to get one byte wrong on).

Observed via `compare_checkpoints.py`'s 5-seed qualitative comparison
(2026-08-16): the replacement glyph appears repeatedly in
`saga-combined-v1`'s output (`aren[glyph]t`, `monster[glyph]s CR`,
`creature[glyph]s turns`, several more) and occasionally in
production's (`else[glyph]s movement`, `monster[glyph]s lair`). More
frequent in `combined-v1`, plausibly because its fine-tune mix includes
the general (non-D&D) StackExchange data, which likely uses curly-quote
typography far more than the D&D-SRD-heavy corpus production trained
on — more exposure to that byte pattern, more chances to get it wrong.

Practical implication: `errors="replace"` is the correct, deliberate
choice here (degrade gracefully instead of crashing on the model's own
output) — this isn't a decode bug to fix. If it's worth mitigating at
all, the lever is training-data normalization (e.g. normalizing curly
quotes to ASCII equivalents during preprocessing so the model never
needs to reproduce the 3-byte sequence exactly), not a tokenizer change.
Low severity — cosmetic, not a crash or a correctness issue — flagged
for awareness rather than as a must-fix.

## Residual Stack-Exchange-answer register drift in `general-expansion-v1`

**Status:** confirmed, partially mitigated by fine-tuning, not fully fixed.

The pretrain-only qualitative check on `general_expansion_v1` (before
fine-tuning) found near-universal "forum Q&A register" drift — completions
simulating a full Stack Exchange thread (posing new questions mid-response,
first-person anecdotes like "I was playing this campaign using my first
homebrew", "No: If you have..." answer-style closers) even on prompts that
should've resisted it (narrative description, rules definitions). Expected,
given the corpus's general content is now 73.6%-of-weighted-windows
Q&A-dominated (see `expansion_PLAN.md`).

Fine-tuning `general-expansion-v1` (`checkpoints/finetune/general-expansion-v1/
step_0013213.pt`) substantially fixed this — the severe, near-universal form
is gone from a 5-seed `compare_checkpoints.py` qualitative pass. A milder
residual tic survives in a real minority of completions (~4-5 of 50),
self-referential SE-answer framing rather than full thread simulation:

- `"As you can see in this answer, I would suggest..."` (armor class of a
  Goblin, seed 4)
- `"What I would like to add to the answer that it depends on..."`
  (Challenge Rating, seed 0)
- `"I've been to a similar situation in one of the previous answers..."`
  (rogue in a crypt, seed 2)
- `"I'm going to answer this by using the general rule..."` (saving throw
  DC, seed 0)

Practical implication: don't expect this register tic to be fully absent
from `general-expansion-v1`'s output — it's rarer and milder than
pretrain-only, not eliminated. Shipped anyway (2026-08-19, see
`training_PLAN.md`'s Step 6 record) since the severe form was the actual
concern and that's fixed; this residual is minor enough not to block on.
No further fix attempted yet — if it's worth pursuing, the lever is
likely the same one flagged for the corpus-composition shift itself
(e.g. further tuning the fine-tune data's general:D&D ratio), not a new
mechanism.
