"""Dataset wrapper pairing TokenizedDataset windows with precomputed retrieval neighbors.

Deliberately a separate wrapper rather than a change to ``TokenizedDataset``
itself: that class's ``(input_ids, target_ids)`` 2-tuple contract is relied
on by ``PaddingCollator`` and a large existing test suite, and pretraining
windows are always fixed-length (no padding path exercised in practice —
see ``TokenizedDataset``'s own docstring), so there is nothing to gain from
touching it. This wrapper only exists for the opt-in RETRO training path
(``TransformerConfig.retro_layers`` — see
``docs/architecture_optimization.md`` item #3); every other caller of
``TokenizedDataset`` is completely unaffected by this module's existence.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID


class NeighborAugmentedDataset(Dataset):
    """Wraps a ``TokenizedDataset``, adding precomputed neighbor token ids per window.

    Pairs each ``(input_ids, target_ids)`` window from ``base`` with the
    corresponding row of a precomputed neighbor-ids array (see
    ``scripts/build_retrieval_neighbors.py``), aligned by index via
    ``base.offsets`` — row ``i`` of ``neighbor_ids`` corresponds to
    ``base[i]``.
    """

    def __init__(self, base: TokenizedDataset, neighbor_ids: np.ndarray) -> None:
        """Pair a dataset with its precomputed neighbor-ids array.

        Args:
            base: The underlying ``TokenizedDataset``.
            neighbor_ids: Integer array of shape
                ``(len(base), n_neighbors, neighbor_len)``, aligned to
                ``base``'s window order — see ``base.offsets``.

        Raises:
            ValueError: If ``neighbor_ids.shape[0] != len(base)`` — almost
                always means the array was built with a different corpus,
                seq_len, stride, or val split than ``base`` was constructed
                with, and silently proceeding would pair every window with
                the wrong neighbors.
        """
        if neighbor_ids.shape[0] != len(base):
            raise ValueError(
                f"neighbor_ids has {neighbor_ids.shape[0]} row(s) but the base "
                f"dataset has {len(base)} window(s) — they must be built from "
                f"the exact same corpus_path/seq_len/stride/regions "
                f"(see scripts/build_retrieval_neighbors.py)."
            )
        self._base = base
        self._neighbor_ids = neighbor_ids

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(input_ids, target_ids, neighbor_ids)`` for window ``idx``."""
        input_ids, target_ids = self._base[idx]
        neighbor_ids = torch.from_numpy(np.asarray(self._neighbor_ids[idx], dtype=np.int64))
        return input_ids, target_ids, neighbor_ids


def collate_with_neighbors(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    pad_id: int = PAD_ID,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate ``(input_ids, target_ids, neighbor_ids)`` triples into a batch.

    ``input_ids``/``target_ids`` are padded exactly like
    ``PaddingCollator`` (reused internally, not reimplemented).
    ``neighbor_ids`` is simply stacked — every item already has the fixed
    ``(n_neighbors, neighbor_len)`` shape ``build_retrieval_neighbors.py``
    produces, so no padding is needed on that axis.

    Args:
        batch: A list of ``(input_ids, target_ids, neighbor_ids)`` tuples,
            as returned by ``NeighborAugmentedDataset.__getitem__``.
        pad_id: Forwarded to the underlying ``PaddingCollator``.

    Returns:
        ``(input_ids, target_ids, attention_mask, neighbor_ids)``.
    """
    from grimoire_ai.llm.data.collator import PaddingCollator

    pairs = [(inp, tgt) for inp, tgt, _ in batch]
    input_ids, target_ids, attention_mask = PaddingCollator(pad_id=pad_id)(pairs)
    neighbor_ids = torch.stack([n for _, _, n in batch], dim=0)
    return input_ids, target_ids, attention_mask, neighbor_ids
