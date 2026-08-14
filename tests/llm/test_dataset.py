"""Tests for TokenizedDataset and PaddingCollator.

Gate criteria:
- TokenizedDataset produces the correct number of windows for a given stride.
- input_ids and target_ids are offset by exactly one position.
- Dataset raises FileNotFoundError on a missing corpus.
- Dataset raises ValueError when corpus is too short.
- PaddingCollator produces correctly shaped tensors and masks.
- PaddingCollator pads shorter sequences correctly (left-pad).
- PaddingCollator handles a batch of equal-length sequences without padding.
- PaddingCollator's equal-length fast path actually skips pad_sequence,
  not just produces the same shape the slow path would also produce.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_corpus(tokens: list[int], tmp_dir: str) -> str:
    """Write a list of token ids to a binary corpus file and return its path."""
    path = str(Path(tmp_dir) / "corpus.bin")
    arr = np.array(tokens, dtype=np.int32)
    fp = np.memmap(path, dtype=np.int32, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    del fp
    return path


# ---------------------------------------------------------------------------
# TokenizedDataset
# ---------------------------------------------------------------------------

def test_dataset_window_count_non_overlapping() -> None:
    """Non-overlapping windows (stride=seq_len) should give floor((N-1)/L) items."""
    seq_len = 4
    tokens = list(range(20))   # 20 tokens
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds = TokenizedDataset(path, seq_len=seq_len, stride=seq_len)
    # valid starts: 0, 4, 8, 12, 16 — but 16+4+1=21 > 20 so 16 is excluded
    # starts: range(0, 20-4, 4) = 0, 4, 8, 12, 16 — check
    expected = len(list(range(0, 20 - seq_len, seq_len)))
    assert len(ds) == expected


def test_dataset_window_count_overlapping() -> None:
    """Overlapping windows (stride=seq_len//2) should double the window count."""
    seq_len = 4
    tokens = list(range(20))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds_overlap = TokenizedDataset(path, seq_len=seq_len, stride=seq_len // 2)
        ds_no_overlap = TokenizedDataset(path, seq_len=seq_len, stride=seq_len)
    assert len(ds_overlap) > len(ds_no_overlap)


def test_dataset_item_shapes() -> None:
    """Each item must return two tensors of shape (seq_len,)."""
    seq_len = 8
    tokens = list(range(50))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds = TokenizedDataset(path, seq_len=seq_len, stride=seq_len)
    inp, tgt = ds[0]
    assert inp.shape == (seq_len,)
    assert tgt.shape == (seq_len,)


def test_dataset_target_is_shifted_by_one() -> None:
    """target_ids must be input_ids shifted left by one position."""
    seq_len = 8
    tokens = list(range(100))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds = TokenizedDataset(path, seq_len=seq_len, stride=seq_len)
    for i in range(min(5, len(ds))):
        inp, tgt = ds[i]
        assert torch.equal(inp[1:], tgt[:-1]), (
            f"Item {i}: target is not a one-step shift of input."
        )


def test_dataset_dtype_is_long() -> None:
    """Tensors returned by the dataset must have dtype torch.long (int64)."""
    tokens = list(range(50))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds = TokenizedDataset(path, seq_len=8)
    inp, tgt = ds[0]
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long


def test_dataset_missing_file_raises() -> None:
    """Constructing a dataset from a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        TokenizedDataset("/tmp/grimoire_nonexistent_corpus.bin", seq_len=8)


def test_dataset_too_short_raises() -> None:
    """A corpus shorter than seq_len+1 tokens must raise ValueError."""
    tokens = list(range(5))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        with pytest.raises(ValueError, match="too short"):
            TokenizedDataset(path, seq_len=8)


def test_dataset_range_restricts_windows() -> None:
    """start/end must restrict windows to the requested token region."""
    seq_len = 4
    tokens = list(range(40))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        # Region [20, 40) with non-overlapping windows.
        ds = TokenizedDataset(path, seq_len=seq_len, stride=seq_len, start=20, end=40)
        # Offsets are absolute into the file, so the first window starts at 20.
        first_inp, _ = ds[0]
        assert first_inp[0].item() == 20
        # No window may read past end=40.
        for i in range(len(ds)):
            inp, tgt = ds[i]
            assert tgt[-1].item() < 40, "Window read past the region end."


def test_dataset_train_val_split_no_token_overlap() -> None:
    """A start/end split must yield train and val regions that share no tokens."""
    seq_len = 4
    tokens = list(range(40))
    split = 24
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        train = TokenizedDataset(path, seq_len=seq_len, stride=seq_len, end=split)
        val = TokenizedDataset(path, seq_len=seq_len, stride=seq_len, start=split)

    def tokens_in(ds: TokenizedDataset) -> set[int]:
        seen: set[int] = set()
        for i in range(len(ds)):
            inp, tgt = ds[i]
            seen.update(inp.tolist())
            seen.update(tgt.tolist())
        return seen

    train_tokens = tokens_in(train)
    val_tokens = tokens_in(val)
    assert max(train_tokens) < split, "Train region leaked tokens past the split."
    assert min(val_tokens) >= split, "Val region leaked tokens before the split."
    assert train_tokens.isdisjoint(val_tokens), "Train and val regions overlap."


def test_dataset_range_too_short_raises() -> None:
    """A selected region shorter than seq_len+1 must raise ValueError."""
    tokens = list(range(40))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        with pytest.raises(ValueError, match="too short"):
            TokenizedDataset(path, seq_len=8, start=35, end=40)


def test_dataset_regions_scattered_no_overlap() -> None:
    """Multiple scattered regions should each be windowed independently with
    no window straddling a region boundary, matching what a single
    contiguous region would give if regions were merged."""
    seq_len = 4
    tokens = list(range(60))
    regions = [(0, 15), (20, 35), (50, 60)]
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        ds = TokenizedDataset(path, seq_len=seq_len, stride=seq_len, regions=regions)

        seen: set[int] = set()
        for i in range(len(ds)):
            inp, tgt = ds[i]
            seen.update(inp.tolist())
            seen.update(tgt.tolist())

    # Every observed token must fall inside one of the given regions.
    for tok in seen:
        assert any(s <= tok < e for s, e in regions), (
            f"Token {tok} fell outside all regions {regions}."
        )
    # Matches windowing each region independently and concatenating offsets.
    expected = sum(
        len(range(s, e - seq_len, seq_len)) for s, e in regions
    )
    assert len(ds) == expected


def test_dataset_regions_too_short_raises() -> None:
    """Regions that collectively can't form a single window must raise ValueError."""
    tokens = list(range(40))
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_corpus(tokens, tmp)
        with pytest.raises(ValueError, match="too short"):
            TokenizedDataset(path, seq_len=8, regions=[(0, 3), (10, 12)])


# ---------------------------------------------------------------------------
# PaddingCollator
# ---------------------------------------------------------------------------

def _make_batch(lengths: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Create a batch of (input, target) pairs with specified lengths."""
    return [
        (torch.arange(1, L + 1, dtype=torch.long),
         torch.arange(2, L + 2, dtype=torch.long))
        for L in lengths
    ]


def test_collator_equal_length_batch() -> None:
    """Collating equal-length sequences should produce no padding."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([8, 8, 8])
    inp, tgt, mask = collator(batch)
    assert inp.shape  == (3, 8)
    assert tgt.shape  == (3, 8)
    assert mask.shape == (3, 8)
    assert mask.all(), "No padding expected for equal-length sequences."


def test_collator_variable_length_batch() -> None:
    """Shorter sequences must be left-padded to the longest sequence length."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([4, 8, 6])
    inp, tgt, mask = collator(batch)
    max_len = 8
    assert inp.shape  == (3, max_len)
    assert tgt.shape  == (3, max_len)
    assert mask.shape == (3, max_len)


def test_collator_padding_positions_are_zero_in_mask() -> None:
    """Padding positions must have mask value 0; real tokens must have value 1."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([3, 7])
    inp, tgt, mask = collator(batch)
    # First sequence has length 3, padded to 7 → first 4 positions are padding.
    assert mask[0, :4].sum() == 0, "Padding positions should have mask=0."
    assert mask[0, 4:].sum() == 3, "Real-token positions should have mask=1."


def test_collator_padding_token_id() -> None:
    """Padded input_ids positions must contain PAD_ID."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([2, 6])
    inp, _, _ = collator(batch)
    # First sequence padded to length 6 → first 4 tokens should be PAD_ID.
    assert (inp[0, :4] == PAD_ID).all(), "Padded positions must contain PAD_ID."


def test_collator_real_tokens_unchanged() -> None:
    """The real token values must not be altered by the collator."""
    collator = PaddingCollator(pad_id=PAD_ID)
    seq = torch.tensor([10, 20, 30], dtype=torch.long)
    batch = [(seq, seq + 1)]
    inp, tgt, _ = collator(batch)
    assert torch.equal(inp[0], seq)
    assert torch.equal(tgt[0], seq + 1)


def test_collator_equal_length_batch_skips_pad_sequence() -> None:
    """Equal-length batches (the pretraining norm) must take the fast
    torch.stack path, not the flip/pad_sequence/flip machinery -- proven by
    making pad_sequence raise if it's called at all, not just by checking
    the output shape (which the slow path would also get right)."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([8, 8, 8])
    with patch(
        "grimoire_ai.llm.data.collator.pad_sequence",
        side_effect=AssertionError("pad_sequence must not be called on an equal-length batch"),
    ):
        inp, tgt, mask = collator(batch)
    assert inp.shape == (3, 8)
    assert tgt.shape == (3, 8)
    assert mask.all()


def test_collator_variable_length_batch_still_uses_pad_sequence() -> None:
    """Confirms the mock in the test above would actually catch a
    regression -- variable-length batches must still go through
    pad_sequence, not silently take the fast path with wrong output."""
    collator = PaddingCollator(pad_id=PAD_ID)
    batch = _make_batch([4, 8, 6])
    with patch(
        "grimoire_ai.llm.data.collator.pad_sequence",
        side_effect=AssertionError("pad_sequence must not be called on an equal-length batch"),
    ) as mocked:
        with pytest.raises(AssertionError, match="must not be called"):
            collator(batch)
    mocked.assert_called()
