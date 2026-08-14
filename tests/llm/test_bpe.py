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

from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import (
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
# Incremental (streaming) decoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "A grappled creature loses its speed.",
    "DC 15 saving throw",
    "2d6+3 piercing damage",
    "hello world",
    "Multiattack, spellcasting, concentration",
    "Héllo wörld",           # non-ASCII, exercises multi-byte UTF-8 chars
    "∇f(x) = 0",             # 3-byte UTF-8 math symbols
    "D&D 5e Player's Handbook",
    "🎲 rolling dice 🎲",     # 4-byte UTF-8 (surrogate-pair-free astral char)
])
def test_incremental_decode_matches_full_decode(trained_encoder: BytePairEncoder, text: str) -> None:
    """Feeding ids one at a time through IncrementalDecoder must produce the
    exact same text as decode(ids) called once on the whole list -- the
    property chat_stream's per-token yields depend on. Multi-byte UTF-8
    characters can straddle a BPE token boundary, so this is the case that
    would actually catch a broken incremental implementation."""
    ids = trained_encoder.encode(text)
    reference = trained_encoder.decode(ids)

    decoder = trained_encoder.incremental_decoder()
    pieces = [decoder.push(token_id) for token_id in ids]
    pieces.append(decoder.finish())
    incremental = "".join(pieces)

    assert incremental == reference == text


def test_incremental_decode_partial_yields_are_prefixes(trained_encoder: BytePairEncoder) -> None:
    """Every intermediate accumulated yield must be a prefix of the final
    text -- streaming output must never later 'unsay' something already
    shown to the user."""
    text = "The overfitting problem occurs when Héllo wörld ∇f(x) = 0"
    ids = trained_encoder.encode(text)
    reference = trained_encoder.decode(ids)

    decoder = trained_encoder.incremental_decoder()
    committed = ""
    for token_id in ids:
        committed += decoder.push(token_id)
        assert reference.startswith(committed)
    committed += decoder.finish()
    assert committed == reference


def test_incremental_decode_with_special_tokens(trained_encoder: BytePairEncoder) -> None:
    """A special token id mixed into the stream must flush pending bytes and
    pass the special token's literal string straight through, matching
    decode()'s behaviour exactly."""
    ids = (
        trained_encoder.encode("hello")
        + [SEP_ID]
        + trained_encoder.encode("wörld")
    )
    reference = trained_encoder.decode(ids)

    decoder = trained_encoder.incremental_decoder()
    incremental = "".join(decoder.push(token_id) for token_id in ids) + decoder.finish()

    assert incremental == reference


def test_incremental_decoder_requires_trained_encoder() -> None:
    enc = BytePairEncoder()
    with pytest.raises(RuntimeError, match="no vocabulary"):
        enc.incremental_decoder()


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


# ---------------------------------------------------------------------------
# extend() — growing a vocabulary without disturbing existing ids
# ---------------------------------------------------------------------------

def test_extend_preserves_existing_token_ids(trained_encoder: BytePairEncoder) -> None:
    """Encoding old text must produce byte-identical ids before and after extend()."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=trained_encoder.vocab_size)
    sample_text = SMALL_CORPUS[0]
    ids_before = enc.encode(sample_text)

    new_corpus = SMALL_CORPUS + [
        "Resistance halves damage from a particular type.",
        "A familiar can deliver touch spells on the caster's behalf.",
    ]
    enc.extend(new_corpus, vocab_size=enc.vocab_size + 32)

    assert enc.encode(sample_text) == ids_before


def test_extend_increases_vocab_size() -> None:
    """extend() must grow the vocabulary, not just reorder it."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    old_size = len(enc)
    enc.extend(SMALL_CORPUS, vocab_size=old_size + 20)
    assert len(enc) > old_size


def test_extend_new_text_round_trips() -> None:
    """Newly learned merges must still encode/decode losslessly."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    new_corpus = SMALL_CORPUS + [
        "Resistance halves damage from a particular type. " * 10,
    ]
    enc.extend(new_corpus, vocab_size=len(enc) + 20)
    text = "Resistance halves damage from a particular type."
    assert enc.decode(enc.encode(text)) == text


def test_extend_rejects_smaller_or_equal_vocab_size() -> None:
    """extend() must reject a vocab_size that doesn't actually grow anything."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    with pytest.raises(ValueError):
        enc.extend(SMALL_CORPUS, vocab_size=len(enc))
    with pytest.raises(ValueError):
        enc.extend(SMALL_CORPUS, vocab_size=len(enc) - 10)


def test_extend_without_training_raises() -> None:
    """extend() on an untrained encoder must raise, like train()/encode()."""
    enc = BytePairEncoder()
    with pytest.raises(RuntimeError, match="no vocabulary"):
        enc.extend(SMALL_CORPUS, vocab_size=512)


def test_extend_empty_corpus_is_a_noop() -> None:
    """An empty corpus has no pairs to learn from — extend() must not raise,
    and the vocabulary must simply stay at its pre-extend size."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    old_size = len(enc)
    enc.extend([], vocab_size=old_size + 50)
    assert len(enc) == old_size


def test_extend_merges_list_is_append_only() -> None:
    """All original merge rules must still be present, in their original
    order, after extend() — only new merges may be appended."""
    enc = BytePairEncoder()
    enc.train(SMALL_CORPUS, vocab_size=300)
    original_merges = list(enc._merges)
    enc.extend(SMALL_CORPUS, vocab_size=len(enc) + 20)
    assert enc._merges[: len(original_merges)] == original_merges


# ---------------------------------------------------------------------------
# save() — atomic write
# ---------------------------------------------------------------------------

def test_save_does_not_leave_tmp_file(trained_encoder: BytePairEncoder) -> None:
    """save() must clean up its temp file and leave only the final path."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bpe.json"
        trained_encoder.save(str(path))
        assert path.exists()
        assert list(Path(tmp).iterdir()) == [path]


def test_save_overwrite_round_trips(trained_encoder: BytePairEncoder) -> None:
    """Overwriting an existing vocab file (the extend() workflow) must
    still round-trip correctly — this is the in-place rewrite path that
    motivated making save() atomic."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bpe.json"
        trained_encoder.save(str(path))

        enc2 = BytePairEncoder()
        enc2.train(SMALL_CORPUS, vocab_size=trained_encoder.vocab_size)
        enc2.extend(SMALL_CORPUS, vocab_size=enc2.vocab_size + 10)
        enc2.save(str(path))  # overwrite

        reloaded = BytePairEncoder.load(str(path))
        assert reloaded.vocab_size == enc2.vocab_size
        assert reloaded._merges == enc2._merges
