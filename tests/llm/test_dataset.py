"""Tests for TokenizedDataset and PaddingCollator.

Gate criteria:
- TokenizedDataset produces the correct number of windows for a given stride.
- input_ids and target_ids are offset by exactly one position.
- Dataset raises FileNotFoundError on a missing corpus.
- Dataset raises ValueError when corpus is too short.
- PaddingCollator produces correctly shaped tensors and masks.
- PaddingCollator pads shorter sequences correctly (left-pad).
- PaddingCollator handles a batch of equal-length sequences without padding.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from grimoire.llm.data.collator import PaddingCollator
from grimoire.llm.data.dataset import TokenizedDataset
from grimoire.llm.tokenizer.special_tokens import PAD_ID


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
