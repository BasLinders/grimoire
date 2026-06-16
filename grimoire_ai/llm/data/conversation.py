"""ConversationDataset: fine-tuning dataset from structured JSONL examples.

Each line of the JSONL file is a JSON object with the following fields:

    {"user": "What happens when a creature is grappled?",
     "assistant": "A grappled creature has its speed reduced to zero.",
     "context": "A grappled creature has its speed reduced to zero."}

``"context"`` is optional.  When present it is injected as the corpus
context block between ``<SEP>`` markers, producing the same prompt format
the model will see at inference:

    <BOS> <SEP> {context} <SEP> <USR> {question} <AST> {answer} <EOS>

Without context the format is:

    <BOS> <USR> {question} <AST> {answer} <EOS>

Response-only loss masking
--------------------------
The loss is computed only over the response tokens — everything after
``<AST>``, including the final ``<EOS>``.  The prompt portion of the target
is replaced with ``PAD_ID`` so ``cross_entropy`` ignores those positions.

This is the standard instruction-tuning technique: teach the model *what
to say*, not *how to read the question*.  It produces faster, more stable
convergence because the model is never penalised for its representation of
the input, only for its output.

The input/target pair follows the standard causal LM shift:

    input:  [BOS, SEP?, *ctx?, SEP?, USR, *q, AST, *a]
    target: [PAD, PAD,  PAD,   PAD,  PAD, PAD, PAD, *a, EOS]

where PAD positions are ignored by the loss.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    EOS_ID,
    SEP_ID,
    USR_ID,
)

# PyTorch canonical value for positions excluded from the loss.
# Using -100 (not PAD_ID=0) so real tokens with id 0 are never silently dropped.
_LABEL_IGNORE_IDX: int = -100


class ConversationDataset(Dataset):
    """PyTorch Dataset over a JSONL file of (user, assistant[, context]) examples.

    Each example is tokenised and formatted into the Grimoire prompt template.
    The target sequence has the prompt portion masked to ``PAD_ID`` so only
    the response tokens contribute to the training loss.

    Attributes:
        examples: List of pre-tokenised ``(input_ids, target_ids)`` pairs as
            ``torch.long`` tensors.  Stored in memory; suitable for the small
            datasets typical of instruction fine-tuning.
        max_seq_len: Maximum token length.  Examples that exceed this after
            formatting are truncated from the right.
    """

    def __init__(
        self,
        path: str,
        tokenizer: BytePairEncoder,
        max_seq_len: int = 1024,
    ) -> None:
        """Load and tokenise all examples from a JSONL file.

        Args:
            path: Path to a ``.jsonl`` file.  Each line must be valid JSON
                with at least ``"user"`` and ``"assistant"`` string fields.
                An optional ``"context"`` field may contain a corpus excerpt.
            tokenizer: A trained ``BytePairEncoder``.  Must have a vocabulary
                loaded before this constructor is called.
            max_seq_len: Hard limit on sequence length.  Sequences longer than
                this are truncated; the final ``EOS`` token is preserved.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the file contains no valid examples.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Fine-tuning data file not found: {path}")

        self.max_seq_len = max_seq_len
        self.examples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            pair = self._encode(obj, tokenizer)
            if pair is not None:
                self.examples.append(pair)

        if not self.examples:
            raise ValueError(f"No valid examples found in {path}.")

    def _encode(
        self,
        obj: dict,
        tokenizer: BytePairEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Encode one JSONL object into (input_ids, target_ids).

        Args:
            obj: Dict with ``"user"`` and ``"assistant"`` keys; optional
                ``"context"`` key.
            tokenizer: Trained BPE tokenizer.

        Returns:
            A ``(input_ids, target_ids)`` tuple of ``torch.long`` tensors,
            or ``None`` if the example is empty after encoding.
        """
        user_ids = tokenizer.encode(obj.get("user", ""))
        asst_ids = tokenizer.encode(obj.get("assistant", ""))
        if not user_ids or not asst_ids:
            return None

        context = obj.get("context")
        if context:
            ctx_ids = tokenizer.encode(context)
            prompt_ids = [BOS_ID, SEP_ID] + ctx_ids + [SEP_ID, USR_ID] + user_ids + [AST_ID]
        else:
            prompt_ids = [BOS_ID, USR_ID] + user_ids + [AST_ID]

        response_ids = asst_ids + [EOS_ID]
        full_ids = prompt_ids + response_ids

        # Truncate to max_seq_len, always keeping the final EOS.
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len - 1] + [EOS_ID]

        # Causal LM shift: input is all but the last token; target is all but
        # the first token.
        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Recompute response_start against the (possibly truncated) full_ids so
        # that truncation cutting into the response doesn't corrupt the mask offset.
        truncated_prompt_len = min(len(prompt_ids), len(full_ids))
        response_start = truncated_prompt_len - 1

        # Skip examples where truncation eliminated all response tokens.
        if response_start >= len(target_ids):
            return None

        # Mask the prompt portion of the target with _LABEL_IGNORE_IDX (-100)
        # so cross_entropy ignores those positions.
        target_ids = (
            [_LABEL_IGNORE_IDX] * min(response_start, len(target_ids))
            + target_ids[response_start:]
        )

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``(input_ids, target_ids)`` pair at ``idx``.

        Args:
            idx: Integer index into the dataset.

        Returns:
            A tuple of two ``torch.long`` tensors.
        """
        return self.examples[idx]
