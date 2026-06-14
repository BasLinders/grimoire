"""Perplexity and bits-per-character evaluation on a corpus binary.

Computes the mean cross-entropy over a held-out slice of a tokenised
corpus binary, then converts to perplexity and bits-per-character (BPC).

    BPC        = CE / ln(2)          (nats → bits)
    Perplexity = exp(CE)             (standard LM metric)

Both metrics use the *token*-level cross-entropy, which is what the
training loop minimises.  BPC is more corpus-agnostic because it
normalises by token count rather than vocabulary size.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from grimoire_ai.llm.model.transformer import GrimoireTransformer

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID


def eval_perplexity(
    model: "GrimoireTransformer",
    corpus_path: str,
    seq_len: int = 1024,
    batch_size: int = 4,
    max_batches: int = 50,
    val_split: float = 0.1,
    device: str = "cpu",
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Compute perplexity and BPC on a held-out slice of a corpus binary.

    Args:
        model: A loaded ``GrimoireTransformer`` (training mode has no effect
            — the model is put into eval mode internally).
        corpus_path: Path to the ``.bin`` tokenised corpus file.
        seq_len: Sequence length used when constructing windows.
        batch_size: Micro-batch size for the eval loop.
        max_batches: Cap on the number of batches evaluated.  Set to 0 for
            the full split.  Default of 50 keeps evaluation fast.
        val_split: Fraction of the corpus to use as the held-out slice.
            Default 0.1 takes the *last* 10 % of tokens (same convention as
            the training pipeline's train/val split).
        device: PyTorch device string.
        on_progress: Optional callback for log lines.

    Returns:
        Dict with keys ``perplexity``, ``bpc``, ``mean_loss``, ``n_batches``.
    """
    import numpy as np

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    _log("Perplexity eval: loading corpus …")

    data = np.memmap(corpus_path, dtype=np.int32, mode="r")
    n_tokens = len(data)
    split_idx = max(seq_len + 1, int(n_tokens * (1.0 - val_split)))

    try:
        dataset = TokenizedDataset(
            corpus_path=corpus_path,
            seq_len=seq_len,
            stride=seq_len,
            start=split_idx,
        )
    except ValueError:
        _log("  ⚠  Corpus too small for a held-out slice — skipping perplexity.")
        return {"perplexity": float("nan"), "bpc": float("nan"), "mean_loss": float("nan"), "n_batches": 0}
    if len(dataset) == 0:
        _log("  ⚠  Corpus too small for a held-out slice — skipping perplexity.")
        return {"perplexity": float("nan"), "bpc": float("nan"), "mean_loss": float("nan"), "n_batches": 0}

    collator = PaddingCollator(pad_id=PAD_ID)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    was_training = model.training
    model.eval()

    total_loss = 0.0
    n_batches = 0
    non_blocking = device == "cuda"

    with torch.no_grad():
        for input_ids, target_ids, attention_mask in loader:
            if max_batches and n_batches >= max_batches:
                break
            input_ids      = input_ids.to(device, non_blocking=non_blocking)
            target_ids     = target_ids.to(device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(device, non_blocking=non_blocking)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=PAD_ID,
            )
            total_loss += loss.item()
            n_batches += 1

            if n_batches % 10 == 0:
                _log(f"  batch {n_batches}/{max_batches or len(loader)}  loss {total_loss/n_batches:.4f}")

    if was_training:
        model.train()

    if n_batches == 0:
        return {"perplexity": float("nan"), "bpc": float("nan"), "mean_loss": float("nan"), "n_batches": 0}

    mean_loss = total_loss / n_batches
    perplexity = math.exp(mean_loss)
    bpc = mean_loss / math.log(2)

    _log(f"  mean loss {mean_loss:.4f}  perplexity {perplexity:.2f}  BPC {bpc:.4f}")
    return {
        "perplexity": round(perplexity, 4),
        "bpc": round(bpc, 4),
        "mean_loss": round(mean_loss, 4),
        "n_batches": n_batches,
    }
