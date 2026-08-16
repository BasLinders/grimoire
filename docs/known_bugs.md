# Known Bugs

Outstanding, unresolved problems — as opposed to `expansion_PLAN.md`/
`training_PLAN.md`, which are narrative logs of decisions and history
(including plenty of *already-fixed* bugs). This is the short, scannable
list of what's still actually broken or misleading, so it doesn't have
to be reconstructed by reading through a session's worth of narrative
each time. Move an entry out once it's fixed — record the fix in
whichever plan doc is tracking that work, not here.

## Repetition loops survive `repetition_penalty=1.3`

**Status:** unresolved, not yet investigated (queued next).

Both `checkpoints/finetune/saga-combined-v1/step_0011339.pt` and the
current production checkpoint
(`checkpoints/finetune/saga-se-qa-weighted-clean-v2/step_0007288.pt`)
produce genuine multi-token hard loops at `repetition_penalty=1.3` —
not rare edge cases. Measured via `scripts/compare_checkpoints.py`
across 5 seeds × 12 prompts (120 responses, 60 per checkpoint), counting
only severe collapses (5+ consecutive repeated tokens, or total
breakdown): **7/60 (11.7%) for each checkpoint**, tied exactly, with no
consistent pattern across seeds. Worst observed instances: ~150
consecutive repeats of "at"/"At", a full sentence-and-a-half repeated
verbatim, and a response that degenerated into several hundred
dash characters with zero actual content.

Possibly connected: `combined-v1` also shows a real tendency to keep
generating rather than concluding naturally (see the token-F1 entry
below) — a model that won't stop has nowhere to go but repeat itself.
Worth investigating together rather than as two separate issues.

`--loop-guard` (`RepetitionLoopGuard` in
`grimoire_ai/llm/inference/constrained_decoding.py`) is a *hard*
structural ban on extending an established loop, distinct from
`repetition_penalty`'s soft logit discount — already implemented and
wired into `grimoire-chat`/the Chat tab (`--loop-guard` /
"Prevent repetition loops"), but not used by any of the evaluation
tooling that surfaced this (`scripts/compare_checkpoints.py`,
`scripts/qualitative_check.py`, `grimoire_ai/llm/eval/quiz.py`'s quiz
eval). Whether the fix is "use `--loop-guard` more broadly," "tune
`repetition_penalty` higher," or something in the sampler itself hasn't
been determined yet.

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
