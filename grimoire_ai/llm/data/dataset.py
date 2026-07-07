"""PyTorch Dataset over a memory-mapped binary corpus file.

The training corpus is stored as a flat array of int32 token ids in a
``.bin`` file produced by ``grimoire.llm.data.preprocessing``.  Loading
the entire file into RAM would be wasteful (and impossible for large
corpora), so this module uses ``numpy.memmap`` to map the file into the
virtual address space and let the OS page in only the slices that are
actually needed.

Sliding-window chunking
-----------------------
A language model is trained to predict the next token at every position.
For a chunk of tokens ``[t_0, t_1, …, t_L]``:

    input_ids  = [t_0, t_1, …, t_{L-1}]   (length L)
    target_ids = [t_1, t_2, …,  t_L   ]   (shifted by one)

The dataset slides a window of width ``seq_len + 1`` across the full
token array with step ``stride``.  A stride of ``seq_len // 2`` makes
adjacent windows overlap by 50 %, so every token appears in roughly two
training examples — this increases data diversity without duplicating
storage.

Using a stride equal to ``seq_len`` gives non-overlapping windows, which
is faster but wastes tokens at the end of each document boundary.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TokenizedDataset(Dataset):
    """Dataset of ``(input_ids, target_ids)`` pairs from a binary corpus.

    Each item is a pair of int64 tensors of length ``seq_len``:

    - ``input_ids``:  tokens at positions ``i`` to ``i + seq_len - 1``
    - ``target_ids``: tokens at positions ``i + 1`` to ``i + seq_len``
      (the next-token targets for the causal LM objective)

    The underlying corpus file is memory-mapped, so only the pages
    corresponding to the requested slice are loaded from disk.

    Attributes:
        seq_len: Length of each training sequence.
        stride: Step size between consecutive windows.  Defaults to
            ``seq_len // 2`` for 50 % overlap.
        _data: ``numpy.memmap`` array of int32 token ids.
        _offsets: List of start indices for each window.
    """

    def __init__(
        self,
        corpus_path: str,
        seq_len: int = 1024,
        stride: Optional[int] = None,
        start: int = 0,
        end: Optional[int] = None,
        regions: Optional[list[tuple[int, int]]] = None,
    ) -> None:
        """Initialise the dataset from a binary corpus file.

        Args:
            corpus_path: Path to the ``.bin`` file written by
                ``grimoire.llm.data.preprocessing``.  Must contain a flat
                sequence of int32 token ids.
            seq_len: Number of tokens per training sequence.  Should match
                ``TransformerConfig.max_seq_len``.
            stride: Step between window start positions.  Defaults to
                ``seq_len // 2`` (50 % overlap).  Use ``stride=seq_len``
                for non-overlapping windows.
            start: First token index (inclusive) of the region to draw
                windows from.  Defaults to 0 (start of file).  Used to carve
                a contiguous train/validation split out of one corpus file
                without any window overlap between the two halves.  Ignored
                if ``regions`` is given.
            end: One-past-the-last token index (exclusive) of the region.
                Defaults to ``None`` (end of file).  All windows lie entirely
                within ``[start, end)``, so a train region ``end=N`` and a
                val region ``start=N`` share no tokens.  Ignored if
                ``regions`` is given.
            regions: Optional list of ``(start, end)`` token-index pairs to
                draw windows from instead of the single ``[start, end)``
                region above.  Each region is windowed independently (no
                window straddles a region boundary, so scattered regions
                still share no tokens with each other), and the resulting
                offsets are the concatenation of all regions' offsets, in
                the order given.  Used to carve a validation split out of
                scattered blocks rather than one contiguous chunk — see
                ``_split_blocks`` in ``training/train.py``.

        Raises:
            FileNotFoundError: If ``corpus_path`` does not exist.
            ValueError: If the selected region(s) are too short to form even
                one window (fewer than ``seq_len + 1`` tokens) in total.
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2

        # Memory-map the file — no data is read into RAM at this point.
        self._data = np.memmap(str(path), dtype=np.int32, mode="r")
        self.n_tokens = len(self._data)

        if regions is None:
            region_start = max(0, start)
            region_end = self.n_tokens if end is None else min(end, self.n_tokens)
            regions = [(region_start, region_end)]

        # Precompute the start index of every valid window within each
        # region. A window starting at offset `i` needs tokens i … i+seq_len
        # (inclusive), so the last valid start in a region is
        # region_end - seq_len - 1. Offsets are absolute indices into the
        # memmap, so ``__getitem__`` is unchanged regardless of how many
        # regions they were drawn from.
        self._offsets: list[int] = []
        for region_start, region_end in regions:
            region_start = max(0, region_start)
            region_end = self.n_tokens if region_end is None else min(region_end, self.n_tokens)
            self._offsets.extend(range(region_start, region_end - seq_len, self.stride))

        if not self._offsets:
            total_len = sum(max(0, min(e, self.n_tokens) - max(0, s)) for s, e in regions)
            raise ValueError(
                f"Selected corpus region(s) {regions} have {total_len} tokens "
                f"total, which is too short to form a single window of "
                f"seq_len={seq_len}. Need at least {seq_len + 1} tokens in "
                f"some region."
            )

    def __len__(self) -> int:
        """Return the number of training windows in the dataset.

        Returns:
            Total count of ``(input_ids, target_ids)`` pairs.
        """
        return len(self._offsets)

    @property
    def offsets(self) -> list[int]:
        """Absolute start offset (into the corpus token array) of every window.

        Exposed so external tooling (e.g. a sample-weight builder) can align
        per-window weights to dataset order without depending on the private
        ``_offsets`` attribute. Index ``i`` here corresponds to ``self[i]``.
        """
        return self._offsets

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``idx``-th ``(input_ids, target_ids)`` pair.

        Reads a slice of ``seq_len + 1`` tokens from the memory-mapped
        array starting at the precomputed offset, converts it to a
        PyTorch int64 tensor, then splits into input and target.

        Args:
            idx: Index into the dataset (0 to ``len(self) - 1``).

        Returns:
            A tuple ``(input_ids, target_ids)`` where both tensors have
            dtype ``torch.long`` and shape ``(seq_len,)``.
        """
        start = self._offsets[idx]
        # Copy the slice out of the mmap into a regular numpy array first
        # to avoid a PyTorch warning about non-writable memory.
        chunk = np.array(self._data[start : start + self.seq_len + 1], dtype=np.int64)
        tokens = torch.from_numpy(chunk)
        return tokens[:-1], tokens[1:]
