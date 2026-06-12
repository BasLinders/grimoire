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
                without any window overlap between the two halves.
            end: One-past-the-last token index (exclusive) of the region.
                Defaults to ``None`` (end of file).  All windows lie entirely
                within ``[start, end)``, so a train region ``end=N`` and a
                val region ``start=N`` share no tokens.

        Raises:
            FileNotFoundError: If ``corpus_path`` does not exist.
            ValueError: If the selected region is too short to form even one
                window (fewer than ``seq_len + 1`` tokens).
        """
        path = Path(corpus_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len // 2

        # Memory-map the file — no data is read into RAM at this point.
        self._data = np.memmap(str(path), dtype=np.int32, mode="r")
        self.n_tokens = len(self._data)

        # Clamp the requested region to the bounds of the file.
        region_start = max(0, start)
        region_end = self.n_tokens if end is None else min(end, self.n_tokens)
        region_len = region_end - region_start
        if region_len < seq_len + 1:
            raise ValueError(
                f"Selected corpus region [{region_start}, {region_end}) has "
                f"only {max(region_len, 0)} tokens, which is too short to "
                f"form a single window of seq_len={seq_len}. "
                f"Need at least {seq_len + 1} tokens."
            )

        # Precompute the start index of every valid window within the region.
        # A window starting at offset `i` needs tokens i … i+seq_len (inclusive),
        # so the last valid start is region_end - seq_len - 1.  Offsets are
        # absolute indices into the memmap, so ``__getitem__`` is unchanged.
        self._offsets: list[int] = list(
            range(region_start, region_end - seq_len, self.stride)
        )

    def __len__(self) -> int:
        """Return the number of training windows in the dataset.

        Returns:
            Total count of ``(input_ids, target_ids)`` pairs.
        """
        return len(self._offsets)

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
