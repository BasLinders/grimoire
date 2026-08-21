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
