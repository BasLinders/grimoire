"""Dataset and data-loading utilities for training the GrimoireTransformer.

Public surface
--------------
TokenizedDataset
    ``torch.utils.data.Dataset`` over a memory-mapped binary of token ids.
PaddingCollator
    ``collate_fn`` for ``DataLoader`` that pads variable-length sequences.
"""

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.dataset import TokenizedDataset

__all__ = ["TokenizedDataset", "PaddingCollator"]
