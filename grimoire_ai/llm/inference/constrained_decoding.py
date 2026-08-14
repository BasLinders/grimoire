"""Decode-time constraints for structured generation: stat-block value
grammars and a hard repetition-loop guard.

Item #5 from docs/architecture_optimization.md. Both mechanisms here work
by masking next-token logits to ``-inf`` before sampling — they change
nothing about the model or training, only which continuations the sampler
is even allowed to consider. Two failure modes documented in
docs/training_PLAN.md motivate this:

- **Invented CR/XP values**: nothing stops a fine-tuned model from emitting
  a Challenge Rating or XP number that isn't a real SRD value, or an
  Armor Class that's nonsensically large. ``StatBlockConstraint`` detects
  when generation has just emitted a recognised stat-block field label
  (e.g. "Challenge Rating:") and restricts the tokens that can follow to
  ones that keep the value a well-formed prefix for that field — not by
  making the model *know* the right number, but by making it structurally
  impossible to type a malformed one.
- **Degenerate repetition loops** ("does does does..."-style collapse):
  the existing ``repetition_penalty`` in ``sampler.py`` only *discounts*
  repeated tokens, which a confident-enough model can still override.
  ``RepetitionLoopGuard`` makes the token that would extend an
  already-established short repeating cycle literally unsampleable.

Both are opt-in and additive: ``sampler.py``'s ``generate``/``generate_stream``
behave exactly as before when neither is supplied.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID

if TYPE_CHECKING:
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder


# ---------------------------------------------------------------------------
# Repetition-loop guard
# ---------------------------------------------------------------------------

class RepetitionLoopGuard:
    """Hard-bans the token that would extend an already-established loop.

    For each period ``p`` in ``1..max_period``, checks whether the most
    recently generated ``(max_repeats - 1) * p`` tokens already consist of
    ``max_repeats - 1`` back-to-back copies of the trailing ``p``-token
    block. If so, the single token that would complete one more repeat of
    that block (the first token of the block) is masked to ``-inf`` — it
    becomes impossible to sample, not just less likely.

    ``period=1`` catches single-token collapse ("does does does...");
    higher periods catch short-phrase loops ("is a monster is a
    monster..."). This is deliberately a targeted structural guard, not a
    general n-gram blocking scheme: it only ever bans a token once a loop
    has *already* run for ``max_repeats - 1`` cycles, so it never
    interferes with ordinary repeated words (e.g. "the... the...") that
    aren't actually looping.
    """

    def __init__(self, max_repeats: int = 3, max_period: int = 4) -> None:
        """Configure the guard.

        Args:
            max_repeats: How many consecutive copies of a block are
                tolerated before the next one is banned. Must be >= 2 (a
                guard that fires on the very first repeat would ban
                completely ordinary language).
            max_period: Longest block length (in tokens) checked for
                looping. ``1`` alone catches single-token loops; higher
                values also catch short repeating phrases.

        Raises:
            ValueError: If ``max_repeats < 2`` or ``max_period < 1``.
        """
        if max_repeats < 2:
            raise ValueError(f"max_repeats ({max_repeats}) must be at least 2.")
        if max_period < 1:
            raise ValueError(f"max_period ({max_period}) must be at least 1.")
        self.max_repeats = max_repeats
        self.max_period = max_period

    def banned_token_ids(self, generated: list[int]) -> set[int]:
        """Return the token ids that would extend an existing loop right now.

        Args:
            generated: Token ids produced so far in this response (not
                including the prompt).

        Returns:
            A set of banned token ids — empty when no loop is currently
            established for any checked period.
        """
        banned: set[int] = set()
        for period in range(1, self.max_period + 1):
            block = generated[-period:]
            if len(block) < period:
                continue  # not enough history for this period yet
            span = period * (self.max_repeats - 1)
            if len(generated) < span:
                continue
            if generated[-span:] == block * (self.max_repeats - 1):
                banned.add(block[0])
        return banned

    def mask(self, logits: torch.Tensor, generated: list[int]) -> torch.Tensor:
        """Return ``logits`` with any currently-looping continuation banned.

        Args:
            logits: 1-D logits tensor of shape ``(vocab,)``.
            generated: Token ids produced so far in this response.

        Returns:
            ``logits`` unchanged (same tensor) if nothing is banned, else a
            clone with the banned entries set to ``-inf``.
        """
        banned = self.banned_token_ids(generated)
        if not banned:
            return logits
        masked = logits.clone()
        for token_id in banned:
            masked[token_id] = float("-inf")
        return masked


# ---------------------------------------------------------------------------
# Stat-block field value grammars
# ---------------------------------------------------------------------------

class ValueGrammar(ABC):
    """A small, explicit grammar for one stat-block field's value.

    Not a general context-free grammar engine — each implementation is a
    hand-written validator for one concrete value shape (an integer, a
    Challenge Rating). ``is_valid_prefix`` is the only operation needed for
    constrained decoding: at every step the sampler only needs to know
    whether *appending one more candidate token's text* still leaves a
    string that could complete into a well-formed value.
    """

    @abstractmethod
    def is_valid_prefix(self, text: str) -> bool:
        """Return whether *text* could be the start of some valid value.

        Must return ``True`` for the empty string (nothing typed yet is
        always a valid prefix).
        """
        raise NotImplementedError


class IntegerGrammar(ValueGrammar):
    """Digits only, optionally comma-grouped, optionally capped at a maximum.

    Used for XP (unbounded, comma-grouped, e.g. "6,400"), Armor Class, and
    Hit Points (bounded, no commas needed at D&D's usual value ranges).
    """

    _COMMA_GROUPED = re.compile(r"^\d{1,3}(,\d{0,3})*$")

    def __init__(self, max_value: Optional[int] = None, allow_commas: bool = False) -> None:
        """Configure the grammar.

        Args:
            max_value: If set, the complete value may not exceed this. Only
                enforced once enough digits have been typed to know for
                sure (a bound like 30 can't reject the single digit "9" as
                a prefix, since "9" could still become... well, it can't
                here, but shorter prefixes than max_value's own digit count
                are always accepted — see ``is_valid_prefix``).
            allow_commas: If ``True``, thousands-separator commas are
                accepted (grouped loosely, not strictly every 3 digits,
                since this is a structural guard against garbage rather
                than a typographic validator).
        """
        self.max_value = max_value
        self.allow_commas = allow_commas

    def is_valid_prefix(self, text: str) -> bool:
        if text == "":
            return True
        if self.allow_commas:
            if not self._COMMA_GROUPED.fullmatch(text):
                return False
            digits = text.replace(",", "")
        else:
            if not text.isdigit():
                return False
            digits = text
        if self.max_value is None or digits == "":
            return True
        max_digit_count = len(str(self.max_value))
        if len(digits) > max_digit_count:
            return False
        if len(digits) == max_digit_count and int(digits) > self.max_value:
            return False
        return True


class ChallengeRatingGrammar(ValueGrammar):
    """CR is one of a fixed, known set of SRD values — not an arbitrary number."""

    VALUES: frozenset[str] = frozenset(
        {"0", "1/8", "1/4", "1/2"} | {str(i) for i in range(1, 31)}
    )

    def is_valid_prefix(self, text: str) -> bool:
        if text == "":
            return True
        return any(value.startswith(text) for value in self.VALUES)


# ---------------------------------------------------------------------------
# Field registry and orchestrator
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """One recognised stat-block field: how to spot it, and its value grammar.

    Attributes:
        label_pattern: Compiled regex matched against generated text
            (not anchored) to find where this field's label was just
            emitted, e.g. ``"Challenge Rating:"``. The value grammar
            applies to whatever comes after the *last* (most recent) match
            of this pattern.
        grammar: The value grammar to enforce after the label.
        max_value_len: Hard cap on how many characters the value may grow
            to before the constraint releases regardless of grammar — a
            safety valve against a field never syntactically "finishing"
            (e.g. an unbounded integer grammar) and locking generation into
            digits-only forever.
    """

    label_pattern: re.Pattern
    grammar: ValueGrammar
    max_value_len: int = 12


#: Default field registry covering the failure modes named in
#: docs/training_PLAN.md: CR and XP specifically, plus AC and HP as the
#: same well-defined bounded-integer shape. Deliberately not exhaustive —
#: free-form fields like Speed ("30 ft., fly 60 ft.") aren't given a
#: grammar here; see the module docstring for the intended scope.
DEFAULT_STAT_BLOCK_FIELDS: list[FieldSpec] = [
    FieldSpec(
        re.compile(r"(?:Challenge Rating|CR)\s*:?\s*", re.IGNORECASE),
        ChallengeRatingGrammar(),
    ),
    FieldSpec(
        re.compile(r"\bXP\s*:?\s*", re.IGNORECASE),
        IntegerGrammar(max_value=None, allow_commas=True),
    ),
    FieldSpec(
        re.compile(r"(?:Armor Class|AC)\s*:?\s*", re.IGNORECASE),
        IntegerGrammar(max_value=30),
    ),
    FieldSpec(
        re.compile(r"(?:Hit Points|HP)\s*:?\s*", re.IGNORECASE),
        IntegerGrammar(max_value=999),
    ),
]

#: Characters that legitimately end a field value. Not part of any
#: grammar's own alphabet, so they must be allowed explicitly — otherwise
#: the model could never naturally stop typing digits.
_TERMINATORS: tuple[str, ...] = (" ", "\n", ",", ")", ".", ":", ";")


class StatBlockConstraint:
    """Token-level grammar constraint for D&D stat-block numeric fields.

    Detects when generated text has just emitted a recognised field label
    and, while that field's value is being generated, masks every
    next-token candidate that would make the value stop looking like a
    well-formed value for that field. The constraint self-releases the
    instant the model picks a terminator token (space, newline, comma,
    etc.) to end the value, or once ``max_value_len`` is reached — there is
    no explicit "value complete" check, which would otherwise have to
    guess an arbitrary correct length for an open-ended field like XP.

    Attach to an ``InferenceEngine`` via
    ``engine.stat_block_constraint = StatBlockConstraint(engine.tokenizer)``.
    """

    def __init__(
        self,
        tokenizer: "BytePairEncoder",
        fields: Optional[list[FieldSpec]] = None,
    ) -> None:
        """Precompute the small token subsets this constraint needs.

        Args:
            tokenizer: A trained ``BytePairEncoder`` — used once here to
                find every vocabulary token that decodes to pure
                digits/comma/slash/space (a field value's own alphabet) or
                to a single terminator character. Byte-level BPE guarantees
                every single ASCII character exists as a standalone token,
                so this is never empty.
            fields: Field registry to use. Defaults to
                ``DEFAULT_STAT_BLOCK_FIELDS``.
        """
        self._fields = fields if fields is not None else DEFAULT_STAT_BLOCK_FIELDS
        self._alphabet_token_ids = self._scan_tokens(tokenizer, set("0123456789,/ "))
        self._terminator_token_ids = set(
            self._scan_tokens(tokenizer, None, exact=_TERMINATORS).keys()
        )

    @staticmethod
    def _scan_tokens(
        tokenizer: "BytePairEncoder",
        allowed_chars: Optional[set[str]],
        exact: Optional[tuple[str, ...]] = None,
    ) -> dict[int, str]:
        """Decode every vocab id once and keep the ones matching a filter.

        Args:
            allowed_chars: Keep tokens whose decoded text uses only these
                characters (and is non-empty). Mutually exclusive with
                ``exact``.
            exact: Keep tokens whose decoded text exactly equals one of
                these strings. Mutually exclusive with ``allowed_chars``.
        """
        out: dict[int, str] = {}
        for token_id in range(tokenizer.vocab_size):
            text = tokenizer.decode([token_id])
            if not text:
                continue
            if exact is not None:
                if text in exact:
                    out[token_id] = text
            elif allowed_chars is not None and set(text) <= allowed_chars:
                out[token_id] = text
        return out

    def _active_field(self, text_so_far: str) -> Optional[tuple[FieldSpec, str]]:
        """Return ``(field, value_so_far)`` if a field's value is being
        generated right now, else ``None``.

        Picks whichever registered field's label most recently appeared in
        ``text_so_far`` (largest match end position), so a later field
        label always takes over from an earlier one.
        """
        best: Optional[tuple[FieldSpec, int]] = None
        for spec in self._fields:
            matches = list(spec.label_pattern.finditer(text_so_far))
            if not matches:
                continue
            end = matches[-1].end()
            if best is None or end > best[1]:
                best = (spec, end)
        if best is None:
            return None
        spec, end = best
        value_so_far = text_so_far[end:]
        if len(value_so_far) > spec.max_value_len:
            return None
        if not spec.grammar.is_valid_prefix(value_so_far):
            return None
        return spec, value_so_far

    def mask(self, logits: torch.Tensor, generated: list[int], tokenizer: "BytePairEncoder") -> torch.Tensor:
        """Return ``logits`` restricted to the active field's valid continuations.

        Args:
            logits: 1-D logits tensor of shape ``(vocab,)``.
            generated: Token ids produced so far in this response (not
                including the prompt) — decoded to find the active field.
            tokenizer: The same tokenizer this constraint was built from
                (passed per call rather than stored twice, since
                ``generate``/``generate_stream`` already needs one).

        Returns:
            ``logits`` unchanged if no field is currently active, else a
            new tensor with every invalid continuation set to ``-inf``.
        """
        text_so_far = tokenizer.decode(generated)
        active = self._active_field(text_so_far)
        if active is None:
            return logits
        spec, value_so_far = active

        allowed = {EOS_ID} | self._terminator_token_ids
        for token_id, token_text in self._alphabet_token_ids.items():
            candidate = value_so_far + token_text
            if len(candidate) <= spec.max_value_len and spec.grammar.is_valid_prefix(candidate):
                allowed.add(token_id)

        masked = torch.full_like(logits, float("-inf"))
        idx = torch.tensor(sorted(allowed), device=logits.device, dtype=torch.long)
        masked[idx] = logits[idx]
        return masked
