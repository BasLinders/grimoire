"""Per-weight-tier validation loss for a pretrained checkpoint.

A single aggregate validation loss can't distinguish "this run is worse"
from "this run's validation set is a harder, more representative sample
than the last one's" -- exactly the trap docs/expansion_PLAN.md already
documented once (2026-07-07): comparing a stratified-validation run's
loss directly against an older scattered-block or contiguous-tail run's
loss is not apples to apples, since the held-out set's composition
differs. The fix used then was a one-off, hand-computed per-tier
breakdown; this script makes that breakdown a reusable, repeatable tool
instead.

Reuses train.py's own --val-stratified split logic (_split_by_tier) so
the val windows evaluated here are byte-identical to the ones a
--val-stratified training run held out -- this only regroups them by
tier instead of flattening them into one combined validation set.

Usage
-----
    python scripts/eval_per_tier.py \\
        --checkpoint checkpoints/pretrain/weighted_clean_v2/step_0015259.pt \\
        --corpus data/processed/corpus.bin \\
        --val-split 0.01

Must be given the same --val-split (and implicitly --seq-len) the
training run used, or this evaluates a different split than the one
that run actually held out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.data.sample_weights import load_doc_weight_sidecars
from grimoire_ai.llm.device import select_device
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.train import _VAL_SPLIT_SEED, _split_by_tier
from grimoire_ai.llm.training.trainer import _select_amp_dtype


def _group_val_regions_by_tier(
    doc_end_offsets: np.ndarray,
    doc_weights: np.ndarray,
    val_split: float,
    seq_len: int,
    seed: int = _VAL_SPLIT_SEED,
) -> dict[float, list[tuple[int, int]]]:
    """Reproduce train.py's --val-stratified split, grouped by tier.

    _split_by_tier returns one flattened val_regions list (that's all
    Trainer needs to build a single val_loader); this calls the exact
    same function for identical regions, then looks each region's
    originating document weight back up from the sidecars to bucket
    them by tier instead of merging them.
    """
    doc_starts = np.concatenate(([0], doc_end_offsets[:-1]))
    doc_regions = list(zip(doc_starts.tolist(), doc_end_offsets.tolist()))
    region_to_weight = dict(zip(doc_regions, doc_weights.tolist()))

    _, val_regions = _split_by_tier(doc_end_offsets, doc_weights, val_split, seq_len, seed=seed)

    by_tier: dict[float, list[tuple[int, int]]] = {}
    for region in val_regions:
        weight = region_to_weight[region]
        by_tier.setdefault(weight, []).append(region)
    return by_tier


@torch.no_grad()
def _eval_loss(
    model: GrimoireTransformer,
    dataset: TokenizedDataset,
    device: str,
    batch_size: int,
) -> tuple[float, int]:
    """Mean cross-entropy loss over *dataset*, matching Trainer.evaluate()."""
    use_amp = device == "cuda"
    amp_dtype, _ = _select_amp_dtype(use_amp)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=PaddingCollator(pad_id=PAD_ID),
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    for input_ids, target_ids, attention_mask in loader:
        input_ids      = input_ids.to(device)
        target_ids     = target_ids.to(device)
        attention_mask = attention_mask.to(device)

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, model.config.vocab_size),
                target_ids.view(-1),
                ignore_index=-100,
            )
        total_loss += loss.item()
        n_batches += 1

    return (total_loss / max(n_batches, 1)), len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report validation loss separately for each --weight-pattern tier.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt file.")
    parser.add_argument("--corpus", default="data/processed/corpus.bin",
                         help="Corpus .bin the checkpoint was trained on.")
    parser.add_argument("--val-split", type=float, required=True,
                         help="Must match the val_split the training run used.")
    parser.add_argument("--seq-len", type=int, default=1024,
                         help="Must match the training run's max_seq_len (default: 1024).")
    parser.add_argument("--batch-size", type=int, default=8,
                         help="Eval batch size (default: 8; does not need to match training).")
    args = parser.parse_args()

    doc_end_offsets, doc_weights = load_doc_weight_sidecars(args.corpus)
    by_tier = _group_val_regions_by_tier(
        doc_end_offsets, doc_weights, args.val_split, args.seq_len,
    )
    if not by_tier:
        print("No validation regions found -- check --val-split matches the training run.",
              file=sys.stderr)
        sys.exit(1)

    device = select_device()
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})\n")

    print(f"{'Tier':>6} | {'Val windows':>11} | {'Val loss':>9}")
    print("-" * 34)
    rows: list[tuple[float, int, float]] = []
    for tier in sorted(by_tier):
        regions = by_tier[tier]
        dataset = TokenizedDataset(corpus_path=args.corpus, seq_len=args.seq_len, regions=regions)
        loss, n_windows = _eval_loss(model, dataset, device, args.batch_size)
        rows.append((tier, n_windows, loss))
        print(f"{tier:>6.2f} | {n_windows:>11,} | {loss:>9.4f}")

    print()
    total_windows = sum(n for _, n, _ in rows)
    weighted_mean = sum(loss * n for _, n, loss in rows) / max(total_windows, 1)
    print(f"Windows-weighted mean across all tiers: {weighted_mean:.4f} "
          f"({total_windows:,} windows total)")


if __name__ == "__main__":
    main()
