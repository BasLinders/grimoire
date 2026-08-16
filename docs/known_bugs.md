# Known Bugs

Outstanding, unresolved problems — as opposed to `expansion_PLAN.md`/
`training_PLAN.md`, which are narrative logs of decisions and history
(including plenty of *already-fixed* bugs). This is the short, scannable
list of what's still actually broken or misleading, so it doesn't have
to be reconstructed by reading through a session's worth of narrative
each time. Move an entry out once it's fixed — record the fix in
whichever plan doc is tracking that work, not here.

## Token-level F1 quiz metric doesn't fit free-form generation

**Status:** root cause understood (three concrete symptoms, not fixed),
not a code change yet.

`grimoire_ai/llm/eval/quiz.py`'s `_token_f1` is a mathematically
correct implementation of the standard SQuAD F1 formula. The problem
isn't the math — SQuAD F1 assumes predictions are short *extracted
spans* (a few words pulled verbatim from a passage), and this quiz
eval applies it to full free-form chat *generations* (multi-sentence
paragraphs) instead. That mismatch produces three separate, concrete
symptoms:

**1. Length-sensitivity.** `precision = correct_tokens / len(entire_response)`
— a longer-but-equally-correct response scores lower purely from
length. Confirmed comparing `saga-combined-v1` (token-F1 0.1748, avg
60.4 response tokens) against production (0.1961, avg 46.0 tokens)
against ~15-token references: within each checkpoint separately,
response length correlates negatively with F1 (`combined-v1`: -0.545,
production: -0.417) — the same pattern shows up *within* one
checkpoint's own results, not just between the two, confirming length
(not content) drives the score. `keyword_recall` (length-insensitive)
favors `combined-v1` instead (13.61% vs. 12.24%), consistent with it
containing at least as much correct content, just penalized for
verbosity. This is the dominant driver of the current `combined-v1`
vs. production F1 gap.

**2. `_tokenize`'s punctuation-stripping loses semantically-critical
characters**, with no domain awareness of which ones matter here:
- **Signs vanish**: `"+3"` and `"-3"` both tokenize to `"3"`. Verified
  directly: `_token_f1("...is +3.", ref)` and `_token_f1("...is -3.",
  ref)` against a `"...is +3."` reference score byte-identically
  (0.7692...) — the metric cannot tell a correct signed answer from a
  wrong-sign one. Hits exactly the content a D&D quiz is full of
  (attack bonuses, ability modifiers, save DCs).
- **Fractions fragment**: `"1/4"` → `["1","4"]`; CR `1/4` vs. `1/8`
  differ by only one of two tokens, so a wrong CR only costs ~13
  points of F1 (0.80→0.667) instead of being clearly penalized.
- **Possessives/contractions fragment**: `"Gundren's"` →
  `["gundren","s"]`, injecting a meaningless `"s"` token into both
  sides.
- **Decimals fragment**: `"15.5"` → `["15","5"]`.

  Currently affects a minority of the 49-question quiz directly (2
  signed-number references, 2 fractions, 8 possessives, 4 decimals),
  and isn't visibly distorting the *current* comparison — both
  checkpoints already score `keyword_recall=0.0` on every affected
  question, so there's nothing for the sign-blindness to falsely
  reward yet. **Latent risk**: this will silently misrepresent
  progress the moment either checkpoint starts actually getting these
  fact types right, which is the whole point of iterating on
  fine-tuning — flagged before it quietly caps how much credit future
  improvements get, not because it's visibly wrong today.

**3. Question-vocabulary echo inflates F1 independent of correctness.**
Restricting to responses that scored `keyword_recall=0.0` (confirmed
completely wrong by the quiz's own separate check), `token_f1` still
averages 0.15–0.18, and correlates positively with how much of the
*question's own wording* appears in the response (r=0.43 for
`combined-v1`, r=0.55 for production) — even within this all-wrong
subset. A model that's better at restating the question gets rewarded
regardless of whether it then answers it correctly. Doesn't explain
the *direction* of the current `combined-v1`-vs-production gap
(`combined-v1` actually echoes more on average, 0.702 vs. 0.595, which
would push its F1 up, not down — length dominates instead), but it's a
real validity problem generally: any future F1 comparison could be
confounded by which checkpoint happens to be more question-echo-prone
in style, independent of who actually knows more.

**Tested and ruled out as a fourth cause**: not the same issue as the
(now-fixed, see `training_PLAN.md`) repetition-loop bug. Re-ran the
quiz eval on both checkpoints with `loop_guard_max_repeats=3` enabled —
token-F1 barely moved (`combined-v1`: 0.1748→0.1765, production:
0.1961→0.1962). `loop_guard` only intervenes once a loop is already
forming; it has no effect on ordinary verbose-but-not-looping
responses.

Practical implication: don't read a token-F1 delta between checkpoints
as a quality signal on its own — compare `keyword_recall` alongside it,
and treat a large F1/kw-recall disagreement as a cue to check response
length and question-echoing before concluding anything. No code
changed yet; a real fix needs several independent pieces: a
domain-aware tokenizer (preserve signs/fractions/decimals), a
length-normalized F1 variant or response-length capping before
scoring, and ideally excluding the question's own vocabulary from the
overlap count.
