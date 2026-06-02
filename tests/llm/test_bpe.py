"""Tests for the Phase 1 BPE tokenizer.

Gate criteria (all must pass before moving to Phase 2):
- Round-trip: decode(encode(s)) == s for ASCII and Unicode text.
- Special tokens never produced as merge outputs.
- Vocabulary file round-trips through save/load without change.
- Encoding a domain compound term produces fewer tokens than characters.
- Special token ids are stable across save/load.
"""

import json
import tempfile
from pathlib import Path

import pytest

from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    AST_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    SPECIAL_TOKEN_TO_ID,
    USR_ID,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SMALL_CORPUS = [
    "A grappled creature has its speed reduced to zero.",
    "The grappling condition ends if the grappler is incapacitated.",
    "Multiattack allows a creature to make multiple attacks on its turn.",
    "Spellcasting requires concentration for many spells.",
    "DC 15 Dexterity saving throw or take 2d6 piercing damage.",
    "The overfitting problem occurs when a model learns noise instead of signal.",
    "Gradient descent minimises the loss function by following the negative gradient.",
]


@pytest.fixture(scope="module")
def trained_encoder() -> BytePairEncoder:
    """A BytePairEncoder trained on a small domain corpus.

    ``scope="module"`` means training runs once per test module, not once
    per test, since BPE training is the slow step.
    """
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=512)
    return enc


# ---------------------------------------------------------------------------
# Untrained encoder guards
# ---------------------------------------------------------------------------

def test_encode_without_training_raises() -> None:
    """``encode`` must raise RuntimeError on an untrained encoder."""
    enc = BytePairEncoder()
    with pytest.raises(RuntimeError, match="no vocabulary"):
        enc.encode("hello")


def test_decode_without_training_raises() -> None:
    """``decode`` must raise RuntimeError on an untrained encoder."""
    enc = BytePairEncoder()
    with pytest.raises(RuntimeError, match="no vocabulary"):
        enc.decode([1, 2, 3])


def test_train_rejects_undersized_vocab() -> None:
    """Training with a vocab_size below the minimum must raise ValueError."""
    enc = BytePairEncoder()
    with pytest.raises(ValueError):
        enc.train(["hello world"], vocab_size=10)


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "A grappled creature loses its speed.",
    "DC 15 saving throw",
    "2d6+3 piercing damage",
    "hello world",
    "Multiattack, spellcasting, concentration",
    "Héllo wörld",           # non-ASCII
    "∇f(x) = 0",             # math symbols
    "D&D 5e Player's Handbook",
])
def test_round_trip(trained_encoder: BytePairEncoder, text: str) -> None:
    """decode(encode(text)) must equal text for arbitrary input."""
    ids = trained_encoder.encode(text)
    recovered = trained_encoder.decode(ids)
    assert recovered == text, f"Round-trip failed for {text!r}: got {recovered!r}"


def test_round_trip_empty_string(trained_encoder: BytePairEncoder) -> None:
    """Encoding an empty string must return an empty id list."""
    assert trained_encoder.encode("") == []


# ---------------------------------------------------------------------------
# Special token handling
# ---------------------------------------------------------------------------

def test_special_token_ids_are_correct() -> None:
    """Special token ids must match the constants in special_tokens.py."""
    assert PAD_ID == 0
    assert BOS_ID == 1
    assert EOS_ID == 2
    assert SEP_ID == 3
    assert USR_ID == 4
    assert AST_ID == 5


def test_special_tokens_not_in_merges(trained_encoder: BytePairEncoder) -> None:
    """No BPE merge rule must produce a string equal to a special token."""
    merge_outputs = {a + b for a, b in trained_encoder._merges}
    for special in ALL_SPECIAL_TOKENS:
        assert special not in merge_outputs, (
            f"Special token {special!r} was produced as a merge output."
        )


def test_encode_literal_special_tokens(trained_encoder: BytePairEncoder) -> None:
    """Literal special token strings in input must encode to their reserved ids."""
    ids = trained_encoder.encode("<BOS>")
    assert ids == [BOS_ID]

    ids = trained_encoder.encode("<SEP>")
    assert ids == [SEP_ID]


def test_special_tokens_survive_round_trip(trained_encoder: BytePairEncoder) -> None:
    """A sequence containing special tokens must decode back to the same string."""
    text = "<BOS>hello world<SEP>goodbye<EOS>"
    ids = trained_encoder.encode(text)
    assert trained_encoder.decode(ids) == text


# ---------------------------------------------------------------------------
# Merge effectiveness
# ---------------------------------------------------------------------------

def test_compound_term_uses_fewer_tokens_than_characters(
    trained_encoder: BytePairEncoder,
) -> None:
    """A domain compound word must encode to fewer tokens than its character count.

    This verifies that BPE merges actually fired — if not, each character
    would be its own token and the encode would return len(word) ids.
    """
    word = "multiattack"
    ids = trained_encoder.encode(word)
    assert len(ids) < len(word), (
        f"Expected fewer tokens than characters for {word!r}, "
        f"got {len(ids)} ids for {len(word)} chars."
    )


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def test_save_load_round_trip(trained_encoder: BytePairEncoder) -> None:
    """Saving then loading must produce an encoder with identical vocabulary."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bpe.json")
        trained_encoder.save(path)
        loaded = BytePairEncoder.load(path)

    assert loaded.vocab_size == trained_encoder.vocab_size
    assert loaded._encoder == trained_encoder._encoder
    assert loaded._merges == trained_encoder._merges


def test_save_load_encodes_identically(trained_encoder: BytePairEncoder) -> None:
    """An encoder reloaded from disk must produce identical ids to the original."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bpe.json")
        trained_encoder.save(path)
        loaded = BytePairEncoder.load(path)

    text = "grappled creature spellcasting DC 15"
    assert loaded.encode(text) == trained_encoder.encode(text)


def test_special_token_ids_stable_after_save_load(
    trained_encoder: BytePairEncoder,
) -> None:
    """Special token ids must not change after a save/load cycle."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bpe.json")
        trained_encoder.save(path)
        loaded = BytePairEncoder.load(path)

    for token, expected_id in SPECIAL_TOKEN_TO_ID.items():
        assert loaded._encoder[token] == expected_id, (
            f"Special token {token!r} id changed after save/load: "
            f"expected {expected_id}, got {loaded._encoder[token]}."
        )


def test_load_nonexistent_file_raises() -> None:
    """Loading from a path that does not exist must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        BytePairEncoder.load("/tmp/grimoire_nonexistent_bpe.json")


# ---------------------------------------------------------------------------
# Vocabulary size
# ---------------------------------------------------------------------------

def test_vocab_size_matches_len(trained_encoder: BytePairEncoder) -> None:
    """``len(encoder)`` must equal ``encoder.vocab_size``."""
    assert len(trained_encoder) == trained_encoder.vocab_size


def test_vocab_size_at_most_requested() -> None:
    """The trained vocabulary must not exceed the requested size.

    It may be smaller if the corpus is too small to fill the target.
    """
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    assert len(enc) <= 300
