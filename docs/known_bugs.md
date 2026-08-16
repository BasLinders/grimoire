# Known Bugs

Outstanding, unresolved problems — as opposed to `expansion_PLAN.md`/
`training_PLAN.md`, which are narrative logs of decisions and history
(including plenty of *already-fixed* bugs). This is the short, scannable
list of what's still actually broken or misleading, so it doesn't have
to be reconstructed by reading through a session's worth of narrative
each time. Move an entry out once it's fixed — record the fix in
whichever plan doc is tracking that work, not here.

## Token-level F1 quiz metric is length-sensitive, can misrepresent checkpoint quality

**Status:** understood, not fixed — a metric limitation to account for
when reading eval reports, not (yet) a code change.

`grimoire_ai/llm/eval/quiz.py`'s `token_f1` uses the standard SQuAD
formula: `precision = correct_tokens / len(entire_response)`. A
longer-but-equally-correct response scores a *lower* F1 purely from
length, independent of actual answer quality.

Confirmed empirically comparing `saga-combined-v1` (token-F1 0.1748)
against production (token-F1 0.1961), which on its own reads as a
regression: `combined-v1` averages 60.4 response tokens vs.
production's 46.0, against ~15-token reference answers, and within
each checkpoint separately, response length correlates negatively with
F1 (`combined-v1`: -0.545, production: -0.417) — the same pattern shows
up *within* a single checkpoint's own results, not just between the two
checkpoints, confirming length (not content) is driving the score.
`combined-v1` also hit or neared the 128-token generation cap on 4/49
quiz questions vs. production's 0/49. `keyword_recall` (length-insensitive
— substring presence only) favors `combined-v1` (13.61% vs. 12.24%),
consistent with it containing at least as much correct content, just
wrapped in more verbosity that token-F1 penalizes.

Practical implication: don't read a token-F1 delta between two
checkpoints as a quality signal without also checking response length —
compare `keyword_recall` or a length-normalized variant instead when
verbosity might differ between the checkpoints being compared. No code
changed yet; a real fix would need either a length-normalized F1
variant or capping/trimming responses before scoring.

**Tested and ruled out**: not the same issue as the (now-fixed, see
`training_PLAN.md`) repetition-loop bug. Re-ran the quiz eval on both
checkpoints with `loop_guard_max_repeats=3` enabled — token-F1 barely
moved (`combined-v1`: 0.1748→0.1765, production: 0.1961→0.1962).
`loop_guard` only intervenes once a loop is already forming; it has no
effect on ordinary verbose-but-not-looping responses, which is what's
actually driving this gap. Two separate mechanisms, not one.
