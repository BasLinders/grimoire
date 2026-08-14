"""Collator that pads variable-length sequences into a batch tensor.

PyTorch's ``DataLoader`` calls the collator on a list of items returned
by the dataset's ``__getitem__``.  Our ``TokenizedDataset`` always returns
fixed-length sequences (``seq_len`` tokens), so in practice all sequences
in a batch are the same length and no padding is needed during normal
training.

The collator is still useful for two cases:
1. Inference or evaluation, where the input may be shorter than ``seq_len``.
2. Any future dataset variant that returns variable-length sequences
   (e.g. document-level batching where each document is its own sequence).

Padding strategy: left-pad with ``PAD_ID = 0``.
Attention mask: ``1`` for real tokens, ``0`` for padding positions.
The model's attention layer adds ``-inf`` to padded positions before the
softmax, so they receive zero attention weight and do not influence any
real token's representation.

Why left-pad (not right-pad)?
------------------------------
In a causal (decoder-only) model, the prediction at each position depends
only on positions to its left.  If we right-padded, the model would see
padding tokens in the middle of a sequence when the autoregressive context
window crosses a document boundary — a confusing signal.  Left-padding
pushes all real tokens to the right end of the sequence so the model
always sees a clean, uninterrupted context window.
"""

import torch
from torch.nn.utils.rnn import pad_sequence

from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID

# Labels at padding positions must be ignored by the loss; PyTorch's canonical
# ignore value is -100.  Using PAD_ID (0) was wrong: it would silently exclude
# any real response token whose id happened to be 0.
_LABEL_IGNORE_IDX: int = -100


class PaddingCollator:
    """Collate a list of ``(input_ids, target_ids)`` pairs into padded batches.

    Intended to be passed as ``collate_fn`` to ``torch.utils.data.DataLoader``.

    Attributes:
        pad_id: Token id used for padding.  Defaults to ``PAD_ID`` (0).
    """

    def __init__(self, pad_id: int = PAD_ID) -> None:
        """Initialise the collator.

        Args:
            pad_id: The token id to use for padding positions.  Must match
                the id that the loss function is configured to ignore
                (``ignore_index`` in ``F.cross_entropy``).
        """
        self.pad_id = pad_id

    def __call__(
        self,
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad and stack a list of sequence pairs into batch tensors.

        For each sequence, an attention mask is created before padding:
        ``1`` for every real token, ``0`` for every padding position.
        Sequences are left-padded to the length of the longest sequence
        in the batch.

        Args:
            batch: A list of ``(input_ids, target_ids)`` tuples as
                returned by ``TokenizedDataset.__getitem__``.  Sequences
                may have different lengths.

        Returns:
            A tuple ``(input_ids, target_ids, attention_mask)`` where all
            three tensors have shape ``(batch_size, max_seq_len)``:

            - ``input_ids``: left-padded token id sequences (dtype long).
            - ``target_ids``: left-padded target id sequences (dtype long).
              Padding positions hold ``pad_id`` and are ignored by the loss.
            - ``attention_mask``: binary mask, ``1`` = real token,
              ``0`` = padding (dtype long).
        """
        inputs, targets = zip(*batch)

        # Fast path: TokenizedDataset always yields fixed-length windows
        # during pretraining, so in the common case there is nothing to pad
        # at all -- skip the flip/pad_sequence/flip machinery below (which
        # runs on the main thread, synchronously, every micro-batch) and
        # just stack directly. Falls back to the general padding path
        # whenever lengths actually differ (inference/eval with variable-
        # length prompts, or any future variable-length dataset -- see the
        # module docstring's two documented use cases for that).
        lengths = [seq.shape[0] for seq in inputs]
        if len(set(lengths)) == 1:
            input_ids = torch.stack(inputs)
            target_ids = torch.stack(targets)
            attention_mask = torch.ones_like(input_ids)
            return input_ids, target_ids, attention_mask

        # Build attention masks before padding — each mask has the same
        # length as its corresponding unpadded sequence.
        masks = [torch.ones(len(seq), dtype=torch.long) for seq in inputs]

        # pad_sequence expects a list of 1-D tensors and pads on the right
        # by default.  We flip each sequence, pad, then flip back to
        # achieve left-padding.
        def left_pad(sequences: list[torch.Tensor], fill: int) -> torch.Tensor:
            flipped = [seq.flip(0) for seq in sequences]
            padded = pad_sequence(flipped, batch_first=True, padding_value=fill)
            return padded.flip(1)

        input_ids = left_pad(list(inputs), self.pad_id)
        # Pad labels with -100, not PAD_ID, so cross_entropy(ignore_index=-100)
        # correctly ignores all padding positions without accidentally masking
        # real tokens whose id is 0.
        target_ids = left_pad(list(targets), _LABEL_IGNORE_IDX)
        attention_mask = left_pad(masks, 0)

        return input_ids, target_ids, attention_mask
