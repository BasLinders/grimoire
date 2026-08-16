# Known Bugs

Outstanding, unresolved problems — as opposed to `expansion_PLAN.md`/
`training_PLAN.md`, which are narrative logs of decisions and history
(including plenty of *already-fixed* bugs). This is the short, scannable
list of what's still actually broken or misleading, so it doesn't have
to be reconstructed by reading through a session's worth of narrative
each time. Move an entry out once it's fixed — record the fix in
whichever plan doc is tracking that work, not here.

## Repetition loops survive `repetition_penalty=1.3`

**Status:** root cause understood, fix verified, **not yet deployed anywhere**
(not in `agents.json`, not in any eval script's default config).

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

**Root cause**: `repetition_penalty` (`sampler.py`'s
`_apply_repetition_penalty`) is the standard CTRL-style penalty — a
*flat*, one-time discount applied per unique already-generated token id.
It doesn't escalate with how many times a token has already repeated,
so once the model is confident enough about a continuation that even
the discounted logit is still the highest-scoring option, nothing stops
it from looping indefinitely. This is an inherent property of this
penalty family, not a bug in the (already-vectorized, PR #194)
implementation.

**Fix, verified**: `RepetitionLoopGuard`
(`grimoire_ai/llm/inference/constrained_decoding.py`) is a *hard*
structural ban — once a period-`p` block has repeated `max_repeats - 1`
times consecutively, the token that would extend it one more time is
masked to `-inf`, literally unsampleable rather than just discounted.
Already implemented and wired into `grimoire-chat`/the Chat tab
(`--loop-guard`, defaults `max_repeats=3`/`max_period=4`), and now into
`scripts/compare_checkpoints.py` (`--loop-guard-max-repeats`). Quantified
before/after on `saga-combined-v1`, same 5-seed × 12-prompt methodology,
same severity threshold: **6/60 (10%) severe collapses without the
guard, 0/60 (0%) with `loop_guard_max_repeats=3`** — complete
elimination, no measurable speed cost (42.3s vs. 48.1s for the 60-generation
run, if anything faster). `keyword_recall/token_f1` weren't re-measured
with the guard on; worth doing before/if this ships, in case forcing a
different continuation changes answer content, not just suppresses the
loop.

**Not yet done**: add `loop_guard_max_repeats`/`loop_guard_max_period`
to `agents.json`'s `saga.gen_config` (the loading path,
`AgentRegistry`'s `GenerationConfig(**cfg.gen_config)`, already accepts
these keys with zero code changes needed) and/or to
`scripts/qualitative_check.py` and `grimoire_ai/llm/eval/quiz.py`'s
default `GenerationConfig`, which still don't set it. This is a live
production-serving config change — flagged rather than made
unilaterally.

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
