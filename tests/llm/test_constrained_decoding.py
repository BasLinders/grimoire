"""Tests for decode-time constrained generation (docs/architecture_optimization.md item #5).

Three layers, each tested independently:
- ``RepetitionLoopGuard``: pure token-id logic, no model or tokenizer needed.
- ``IntegerGrammar`` / ``ChallengeRatingGrammar``: pure string logic.
- ``StatBlockConstraint``: needs a tokenizer's decode(); uses a minimal fake
  tokenizer rather than a trained BPE, so these tests don't depend on BPE
  training details.

The bottom section wires both mechanisms through the real ``generate()``
sampler with scripted stub models (same pattern as test_sampler.py's
``_ConstantModel``) to prove the masks actually change end-to-end output,
not just that the classes compute the right set in isolation.
"""

import re

import pytest
import torch
import torch.nn as nn

from grimoire_ai.llm.inference.constrained_decoding import (
    ChallengeRatingGrammar,
    DEFAULT_STAT_BLOCK_FIELDS,
    FieldSpec,
    IntegerGrammar,
    RepetitionLoopGuard,
    StatBlockConstraint,
)
from grimoire_ai.llm.inference.sampler import GenerationConfig, generate
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID


# ---------------------------------------------------------------------------
# RepetitionLoopGuard
# ---------------------------------------------------------------------------

def test_loop_guard_rejects_max_repeats_below_two() -> None:
    with pytest.raises(ValueError, match="max_repeats"):
        RepetitionLoopGuard(max_repeats=1)


def test_loop_guard_rejects_max_period_below_one() -> None:
    with pytest.raises(ValueError, match="max_period"):
        RepetitionLoopGuard(max_period=0)


def test_loop_guard_no_ban_with_insufficient_history() -> None:
    """A single token isn't enough history for even period=1 (needs 2 prior
    copies to justify banning a 3rd)."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    assert guard.banned_token_ids([7]) == set()


def test_loop_guard_bans_third_consecutive_repeat() -> None:
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    assert guard.banned_token_ids([5, 7, 7]) == {7}


def test_loop_guard_no_false_positive_when_tail_is_not_looping() -> None:
    """A single trailing token following an earlier (unrelated) repeat must
    not be banned — only the *current* trailing block matters."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    assert guard.banned_token_ids([7, 7, 9]) == set()


def test_loop_guard_bans_period_two_phrase_loop() -> None:
    """'is a' repeated twice must ban 'is' as the third-repeat starter."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    # token ids standing in for ["is", "a", "is", "a"]
    banned = guard.banned_token_ids([1, 2, 1, 2])
    assert banned == {1}


def test_loop_guard_period_two_does_not_fire_on_period_one_data() -> None:
    guard = RepetitionLoopGuard(max_repeats=3, max_period=1)
    # This would be a period-2 loop, but max_period=1 only checks period 1.
    assert guard.banned_token_ids([1, 2, 1, 2]) == set()


def test_loop_guard_mask_leaves_unbanned_logits_untouched() -> None:
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0])
    masked = guard.mask(logits, [1, 1])
    assert masked[1] == float("-inf")
    assert masked[0] == 1.0
    assert masked[2] == 3.0
    assert masked[3] == 4.0


def test_loop_guard_mask_no_op_when_nothing_banned() -> None:
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    logits = torch.tensor([1.0, 2.0, 3.0])
    masked = guard.mask(logits, [0])
    assert torch.equal(masked, logits)


# ---------------------------------------------------------------------------
# RepetitionLoopGuard -- templated (varying-content) loops
#
# docs/known_bugs.md: RepetitionLoopGuard's exact-match check misses a
# repeating *structure* with a substituted value each cycle (e.g. "CR = 10 +
# Dex bonus. CR = 14 + Str bonus..."). template_match_ratio < 1.0 relaxes the
# per-position check so a bounded fraction of positions may vary, as long as
# the anchor position (the one that would open the next cycle) stays fixed.
# ---------------------------------------------------------------------------

def test_template_ratio_rejects_bad_value() -> None:
    with pytest.raises(ValueError, match="template_match_ratio"):
        RepetitionLoopGuard(template_match_ratio=0.0)
    with pytest.raises(ValueError, match="template_match_ratio"):
        RepetitionLoopGuard(template_match_ratio=1.5)


def test_exact_repeat_still_bans_at_ratio_below_one() -> None:
    """A ratio below 1.0 must still catch plain exact repeats (a stricter
    match trivially satisfies a looser threshold)."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=1, template_match_ratio=0.5)
    assert guard.banned_token_ids([7, 7]) == {7}


def test_default_ratio_ignores_templated_loop() -> None:
    """Baseline: at the default ratio=1.0, a cycle that differs at even one
    position (the varying "slot") is never flagged -- this is the exact
    behaviour docs/known_bugs.md described as missing."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4)
    # Two cycles of ["CR", "=", <value>, "."] with only the value (index 2)
    # varying: ids 1="CR", 2="=", 3="10", 4="14", 5=".".
    generated = [1, 2, 3, 5, 1, 2, 4, 5]
    assert guard.banned_token_ids(generated) == set()


def test_lowered_ratio_catches_templated_loop_anchor() -> None:
    """With the anchor position ("CR") invariant across both prior cycles
    and 3/4 positions matching, a ratio of 0.6 should ban the anchor token
    from opening a third repeat of the template -- without banning anything
    about the varying value itself."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4, template_match_ratio=0.6)
    generated = [1, 2, 3, 5, 1, 2, 4, 5]
    assert guard.banned_token_ids(generated) == {1}


def test_lowered_ratio_does_not_fire_when_anchor_itself_varies() -> None:
    """If the position that would open the next cycle is the one that
    varies (not just some other slot), the guard must not fire there --
    banning it would be guessing at a value, not blocking a stable anchor."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4, template_match_ratio=0.6)
    # Same shape as above but the *first* position of each cycle varies
    # instead of the third: ["10"/"14", "=", "CR", "."].
    generated = [3, 2, 1, 5, 4, 2, 1, 5]
    assert guard.banned_token_ids(generated) == set()


def test_lowered_ratio_respects_threshold_not_just_anchor() -> None:
    """Anchor being invariant alone isn't sufficient -- overall match_ratio
    must also clear the configured threshold, or the guard would fire on
    almost-entirely-different cycles that merely happen to share one
    coincidentally-repeated token."""
    guard = RepetitionLoopGuard(max_repeats=3, max_period=4, template_match_ratio=0.9)
    # Anchor (index 0, id=1) invariant, but only 2/4 positions match overall
    # -- below the 0.9 threshold.
    generated = [1, 2, 3, 5, 1, 9, 4, 6]
    assert guard.banned_token_ids(generated) == set()


# ---------------------------------------------------------------------------
# IntegerGrammar
# ---------------------------------------------------------------------------

def test_integer_grammar_empty_prefix_always_valid() -> None:
    assert IntegerGrammar().is_valid_prefix("") is True


def test_integer_grammar_rejects_non_digits() -> None:
    assert IntegerGrammar().is_valid_prefix("12a") is False


def test_integer_grammar_unbounded_accepts_any_digit_string() -> None:
    grammar = IntegerGrammar(max_value=None)
    assert grammar.is_valid_prefix("6400000") is True


def test_integer_grammar_bounded_accepts_short_prefix_below_digit_count() -> None:
    grammar = IntegerGrammar(max_value=30)
    assert grammar.is_valid_prefix("9") is True  # could become "9"? no, but len(1) < len("30")=2, deferred


def test_integer_grammar_bounded_rejects_prefix_exceeding_max() -> None:
    grammar = IntegerGrammar(max_value=30)
    assert grammar.is_valid_prefix("31") is False
    assert grammar.is_valid_prefix("99") is False


def test_integer_grammar_bounded_accepts_exact_max() -> None:
    grammar = IntegerGrammar(max_value=30)
    assert grammar.is_valid_prefix("30") is True


def test_integer_grammar_bounded_rejects_too_many_digits() -> None:
    grammar = IntegerGrammar(max_value=30)
    assert grammar.is_valid_prefix("100") is False


def test_integer_grammar_comma_grouping_accepted() -> None:
    grammar = IntegerGrammar(max_value=None, allow_commas=True)
    assert grammar.is_valid_prefix("6,400") is True
    assert grammar.is_valid_prefix("6,") is True  # mid-typing, comma just typed


def test_integer_grammar_comma_rejected_without_allow_commas() -> None:
    grammar = IntegerGrammar(max_value=None, allow_commas=False)
    assert grammar.is_valid_prefix("6,400") is False


def test_integer_grammar_leading_comma_rejected() -> None:
    grammar = IntegerGrammar(allow_commas=True)
    assert grammar.is_valid_prefix(",400") is False


# ---------------------------------------------------------------------------
# ChallengeRatingGrammar
# ---------------------------------------------------------------------------

def test_cr_grammar_empty_prefix_valid() -> None:
    assert ChallengeRatingGrammar().is_valid_prefix("") is True


def test_cr_grammar_fraction_prefix_valid() -> None:
    grammar = ChallengeRatingGrammar()
    assert grammar.is_valid_prefix("1") is True
    assert grammar.is_valid_prefix("1/") is True
    assert grammar.is_valid_prefix("1/8") is True


def test_cr_grammar_invalid_fraction_rejected() -> None:
    """1/3 is not a real D&D CR value."""
    assert ChallengeRatingGrammar().is_valid_prefix("1/3") is False


def test_cr_grammar_two_digit_prefix_valid() -> None:
    grammar = ChallengeRatingGrammar()
    assert grammar.is_valid_prefix("3") is True   # prefix of "3" and "30"
    assert grammar.is_valid_prefix("30") is True


def test_cr_grammar_out_of_range_rejected() -> None:
    assert ChallengeRatingGrammar().is_valid_prefix("31") is False


def test_cr_grammar_zero_valid() -> None:
    assert ChallengeRatingGrammar().is_valid_prefix("0") is True


# ---------------------------------------------------------------------------
# StatBlockConstraint (fake tokenizer — decode-only stub, no real BPE needed)
# ---------------------------------------------------------------------------

class _FakeIncrementalDecoder:
    """Fake counterpart to BytePairEncoder's IncrementalDecoder.

    _FakeTokenizer maps each token id directly to a complete string (no
    byte-level buffering, unlike the real BPE tokenizer), so every push()
    immediately resolves -- nothing is ever held back pending more bytes,
    and preview_pending() is always empty.
    """

    def __init__(self, vocab: dict[int, str]) -> None:
        self._vocab = vocab

    def push(self, token_id: int) -> str:
        return self._vocab.get(token_id, "")

    def preview_pending(self) -> str:
        return ""

    def finish(self) -> str:
        return ""


class _FakeTokenizer:
    """Minimal decode-only tokenizer: token id -> fixed string, via a dict."""

    def __init__(self, vocab: dict[int, str], vocab_size: int) -> None:
        self._vocab = vocab
        self.vocab_size = vocab_size

    def decode(self, ids: list[int]) -> str:
        return "".join(self._vocab.get(i, "") for i in ids)

    def incremental_decoder(self) -> _FakeIncrementalDecoder:
        return _FakeIncrementalDecoder(self._vocab)


# Token ids used across the StatBlockConstraint tests below.
_LABEL_CR = 10   # "Challenge Rating: "
_LABEL_XP = 11   # "XP: "
_DIGIT = {str(d): 20 + d for d in range(10)}   # "0".."9" -> ids 20-29
_SLASH = 30      # "/"
_COMMA = 31      # ","
_SPACE = 32      # " "
_BANANA = 33     # "banana" — not a valid prefix of anything


def _build_fake_tokenizer() -> _FakeTokenizer:
    vocab = {
        EOS_ID: "<EOS>",
        _LABEL_CR: "Challenge Rating: ",
        _LABEL_XP: "XP: ",
        _SLASH: "/",
        _COMMA: ",",
        _SPACE: " ",
        _BANANA: "banana",
    }
    vocab.update({tid: digit for digit, tid in _DIGIT.items()})
    return _FakeTokenizer(vocab, vocab_size=64)


def test_stat_block_constraint_alphabet_scan_finds_digits_and_slash() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    for tid in _DIGIT.values():
        assert tid in constraint._alphabet_token_ids
    assert _SLASH in constraint._alphabet_token_ids
    assert _BANANA not in constraint._alphabet_token_ids


def test_stat_block_constraint_no_active_field_before_any_label() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    assert constraint._active_field("just some regular text") is None


def test_stat_block_constraint_active_right_after_label() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    active = constraint._active_field("Challenge Rating: ")
    assert active is not None
    field, value_so_far = active
    assert isinstance(field.grammar, ChallengeRatingGrammar)
    assert value_so_far == ""


def test_stat_block_constraint_tracks_partial_value() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    field, value_so_far = constraint._active_field("Challenge Rating: 1")
    assert value_so_far == "1"


def test_stat_block_constraint_releases_after_invalid_value() -> None:
    """Once the value stops being a valid CR prefix, the field is no longer active."""
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    assert constraint._active_field("Challenge Rating: 31") is None


def test_stat_block_constraint_releases_after_max_value_len() -> None:
    tokenizer = _build_fake_tokenizer()
    fields = [FieldSpec(re.compile("XP: "), IntegerGrammar(max_value=None), max_value_len=3)]
    constraint = StatBlockConstraint(tokenizer, fields=fields)
    assert constraint._active_field("XP: 123") is not None
    assert constraint._active_field("XP: 1234") is None


def test_stat_block_constraint_most_recent_label_wins() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    field, value_so_far = constraint._active_field(
        "Challenge Rating: 5 XP: 1"
    )
    assert isinstance(field.grammar, IntegerGrammar)
    assert value_so_far == "1"


def test_stat_block_constraint_mask_blocks_invalid_and_allows_valid() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    logits = torch.zeros(64)
    logits[_BANANA] = 100.0   # would win under normal sampling
    logits[_DIGIT["5"]] = 1.0
    masked = constraint.mask(logits, [_LABEL_CR], tokenizer)
    assert masked[_BANANA] == float("-inf")
    assert masked[_DIGIT["5"]] == 1.0
    assert masked[EOS_ID] == 0.0  # EOS always allowed, untouched


def test_stat_block_constraint_mask_no_op_when_no_field_active() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    logits = torch.randn(64)
    masked = constraint.mask(logits, [_BANANA], tokenizer)
    assert torch.equal(masked, logits)


def test_stat_block_constraint_allows_terminator_after_value() -> None:
    """Once a value has been typed, a terminator (e.g. space) must remain
    allowed so generation can naturally move past the field."""
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    logits = torch.zeros(64)
    masked = constraint.mask(logits, [_LABEL_CR, _DIGIT["5"]], tokenizer)
    assert masked[_SPACE] == 0.0


def test_stat_block_constraint_incremental_text_matches_decode_each_step() -> None:
    """_text_so_far's incrementally-cached result must match a fresh
    tokenizer.decode(generated) at every step of a growing generation, not
    just on the first call -- this is the property the cache in
    docs/inference_optimization.md item #6 has to preserve relative to the
    old call-decode()-fresh-every-step implementation."""
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)

    generated: list[int] = []
    steps = [_LABEL_CR, _DIGIT["1"], _DIGIT["2"], _SPACE, _LABEL_XP, _DIGIT["4"]]
    for token_id in steps:
        generated.append(token_id)  # same list object, mutated in place -- matches sampler.py
        incremental = constraint._text_so_far(generated, tokenizer)
        assert incremental == tokenizer.decode(generated)


def test_stat_block_constraint_cache_resets_for_a_new_generation() -> None:
    """A StatBlockConstraint instance is attached to the engine once and
    reused across separate chat turns/generations. A second, unrelated
    `generated` list (as a fresh generate() call would pass) must not be
    contaminated by a previous generation's cached text."""
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)

    first_generation = [_LABEL_CR, _DIGIT["5"]]
    assert constraint._text_so_far(first_generation, tokenizer) == tokenizer.decode(first_generation)

    second_generation = [_LABEL_XP, _DIGIT["9"], _DIGIT["9"]]
    assert constraint._text_so_far(second_generation, tokenizer) == tokenizer.decode(second_generation)


def test_default_stat_block_fields_cover_cr_xp_ac_hp() -> None:
    labels = {spec.label_pattern.pattern for spec in DEFAULT_STAT_BLOCK_FIELDS}
    assert any("Challenge Rating" in p for p in labels)
    assert any("XP" in p for p in labels)
    assert any("Armor Class" in p for p in labels)
    assert any("Hit Points" in p for p in labels)


# ---------------------------------------------------------------------------
# End-to-end: masks actually change generate()'s output
# ---------------------------------------------------------------------------

class _TwoTokenPreferenceModel(nn.Module):
    """Always prefers token_a over token_b over everything else — on its
    own, greedy decoding would loop on token_a forever."""

    def __init__(self, vocab_size: int, token_a: int = 10, token_b: int = 11) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, max_seq_len=32)
        self._dummy = nn.Parameter(torch.zeros(1))
        self.token_a = token_a
        self.token_b = token_b

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False,
                past_kvs=None, **_kwargs):
        batch, seq = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        logits = torch.full((batch, seq, self.config.vocab_size), -1e9)
        logits[:, :, self.token_b] = 0.0
        logits[:, :, self.token_a] = 1.0
        if use_cache:
            full = past_len + seq
            kv = (torch.zeros(batch, 1, full, 1), torch.zeros(batch, 1, full, 1))
            return logits, [kv]
        return logits


def test_loop_guard_disabled_by_default_in_generate() -> None:
    """Baseline: with the guard off (default), greedy decoding loops forever."""
    model = _TwoTokenPreferenceModel(vocab_size=64)
    result = generate(
        model, prompt_ids=[1, 4, 6],
        config=GenerationConfig(max_new_tokens=6, temperature=1e-8, top_k=1, top_p=1.0),
    )
    assert result == [10, 10, 10, 10, 10, 10]


def test_loop_guard_breaks_greedy_loop_in_generate() -> None:
    """Enabling loop_guard_max_repeats forces the third repeat to fail over
    to the next-best token, producing a predictable 10,10,11 cycle instead
    of an unbroken run of 10s."""
    model = _TwoTokenPreferenceModel(vocab_size=64)
    result = generate(
        model, prompt_ids=[1, 4, 6],
        config=GenerationConfig(
            max_new_tokens=9, temperature=1e-8, top_k=1, top_p=1.0,
            loop_guard_max_repeats=3, loop_guard_max_period=1,
        ),
    )
    assert result == [10, 10, 11, 10, 10, 11, 10, 10, 11]


class _ScriptedByPastLenModel(nn.Module):
    """Returns a preprogrammed logits row selected by the current past_len —
    lets a test dictate exactly what the model "wants" to say at each step."""

    def __init__(self, vocab_size: int, script: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.config = TransformerConfig(vocab_size=vocab_size, max_seq_len=32)
        self._dummy = nn.Parameter(torch.zeros(1))
        self._script = script

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False,
                past_kvs=None, **_kwargs):
        batch, seq = input_ids.shape
        past_len = past_kvs[0][0].shape[2] if past_kvs is not None else 0
        row = self._script[past_len]
        logits = row.view(1, 1, -1).expand(batch, seq, -1).clone()
        if use_cache:
            full = past_len + seq
            kv = (torch.zeros(batch, 1, full, 1), torch.zeros(batch, 1, full, 1))
            return logits, [kv]
        return logits


def test_stat_block_constraint_prevents_hallucinated_value_in_generate() -> None:
    """End to end: the model 'wants' to say Challenge Rating: banana, but
    the constraint forces a well-formed value instead."""
    tokenizer = _build_fake_tokenizer()
    vocab_size = tokenizer.vocab_size

    def _prefer(token_id: int) -> torch.Tensor:
        row = torch.full((vocab_size,), -1e9)
        row[token_id] = 1e9
        return row

    script = {
        0: _prefer(_LABEL_CR),                          # prompt pass -> emit label
        1: _prefer(_BANANA).clone(),                     # step 1: "wants" banana
        2: _prefer(EOS_ID),                               # step 2: stop
    }
    # Make the constrained step prefer banana most, "5" second — proves the
    # constraint, not just luck, is what selects "5".
    script[1][_BANANA] = 100.0
    script[1][_DIGIT["5"]] = 50.0

    model = _ScriptedByPastLenModel(vocab_size, script)
    constraint = StatBlockConstraint(tokenizer)

    result = generate(
        model, prompt_ids=[1],
        config=GenerationConfig(max_new_tokens=5, temperature=1e-8, top_k=0, top_p=1.0),
        stat_block_constraint=constraint,
        tokenizer=tokenizer,
    )
    assert result == [_LABEL_CR, _DIGIT["5"]]
    assert tokenizer.decode(result) == "Challenge Rating: 5"


def test_generate_requires_tokenizer_when_stat_block_constraint_given() -> None:
    tokenizer = _build_fake_tokenizer()
    constraint = StatBlockConstraint(tokenizer)
    model = _TwoTokenPreferenceModel(vocab_size=64)
    with pytest.raises(ValueError, match="tokenizer"):
        generate(
            model, prompt_ids=[1],
            config=GenerationConfig(max_new_tokens=1),
            stat_block_constraint=constraint,
        )
