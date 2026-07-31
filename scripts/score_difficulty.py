"""Score per-window training difficulty for importance-weighted sampling.

Runs a (partly) trained checkpoint over every window of the tokenised corpus
and records each window's mean cross-entropy loss.  Windows the model already
predicts well (low loss) carry little new signal; windows it predicts poorly
(high loss) are where the remaining learning is.  The resulting weights can be
fed back into training via ``WeightedRandomSampler`` so that hard windows are
sampled more often — a practical form of "stochastic thinning" that spends
compute where it matters instead of uniformly.

Workflow
--------
1. Pre-train for a short warm-up (e.g. 1-2k steps) to get a checkpoint.
2. Run this script to produce ``data/processed/weights.npy``.
3. Point training at it (``"sample_weights_path"`` in the train config) and
   continue / restart training with difficulty-weighted sampling.

The weights are written in dataset order, so they must be scored with the same
``--seq-len`` and ``--stride`` (and corpus) that training uses.

Usage
-----
    python scripts/score_difficulty.py \\
        --checkpoint checkpoints/step_0002000.pt \\
        --corpus     data/processed/corpus.bin \\
        --output     data/processed/weights.npy

Requirements
------------
    torch, numpy  (already core dependencies)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.device import select_device
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
from grimoire_ai.llm.training.checkpoint import load_checkpoint


@torch.no_grad()
def score(
    checkpoint: str,
    corpus: str,
    output: str,
    seq_len: int,
    stride: int,
    batch_size: int,
    alpha: float,
    weight_floor: float,
    device: str,
) -> None:
    """Compute and save per-window importance weights."""
    print(f"Loading checkpoint: {checkpoint}")
    ckpt = load_checkpoint(checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    effective_seq = seq_len or config.max_seq_len
    effective_stride = stride or effective_seq
    dataset = TokenizedDataset(corpus, seq_len=effective_seq, stride=effective_stride)
    print(f"Scoring {len(dataset):,} windows (seq_len={effective_seq}, stride={effective_stride})")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,            # preserve dataset order for alignment
        collate_fn=PaddingCollator(pad_id=PAD_ID),
        drop_last=False,
    )

    losses = np.empty(len(dataset), dtype=np.float64)
    pos = 0
    non_blocking = device == "cuda"
    for input_ids, target_ids, attention_mask in loader:
        input_ids      = input_ids.to(device, non_blocking=non_blocking)
        target_ids     = target_ids.to(device, non_blocking=non_blocking)
        attention_mask = attention_mask.to(device, non_blocking=non_blocking)

        logits = model(input_ids, attention_mask=attention_mask)
        # Per-token loss, then mean over the non-pad tokens of each row.
        tok_loss = F.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1),
            ignore_index=PAD_ID,
            reduction="none",
        ).view(target_ids.shape)                       # (batch, seq)
        valid = (target_ids != PAD_ID).float()
        row_loss = (tok_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

        n = row_loss.shape[0]
        losses[pos: pos + n] = row_loss.double().cpu().numpy()
        pos += n
        if pos % (batch_size * 50) == 0:
            print(f"  {pos:,}/{len(dataset):,} windows scored")

    # Convert losses to sampling weights: weight ∝ loss**alpha, with a floor so
    # no window is ever fully starved.  Normalised to mean 1 for readability.
    weights = np.power(np.maximum(losses, 0.0), alpha)
    weights = np.maximum(weights, weight_floor)
    weights /= weights.mean()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), weights.astype(np.float32))

    print(
        f"\nDone. Wrote {len(weights):,} weights to {out_path}\n"
        f"  loss   min/mean/max: {losses.min():.3f} / {losses.mean():.3f} / {losses.max():.3f}\n"
        f"  weight min/mean/max: {weights.min():.3f} / {weights.mean():.3f} / {weights.max():.3f}"
    )
    print(
        "Set \"sample_weights_path\" in your training config to this file. "
        "It must be re-scored if you change the corpus, seq_len, or stride."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score per-window training difficulty for weighted sampling."
    )
    parser.add_argument("--checkpoint", required=True, help="Warm-up checkpoint (.pt).")
    parser.add_argument("--corpus", default="data/processed/corpus.bin",
                        help="Tokenised corpus (.bin).")
    parser.add_argument("--output", default="data/processed/weights.npy",
                        help="Destination .npy weights file.")
    parser.add_argument("--seq-len", type=int, default=0,
                        help="Window length (0 = model max_seq_len).")
    parser.add_argument("--stride", type=int, default=0,
                        help="Window stride (0 = non-overlapping = seq_len).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Exponent on loss: >1 sharpens toward hard windows, "
                             "<1 flattens. 1.0 = weight directly proportional to loss.")
    parser.add_argument("--weight-floor", type=float, default=0.05,
                        help="Minimum weight so easy windows are still occasionally seen.")
    parser.add_argument("--device", default=None,
                         help="cpu / cuda / mps (auto if omitted: CUDA, then "
                              "MPS on Apple Silicon, then CPU).")
    args = parser.parse_args()

    device = select_device(args.device)
    try:
        score(
            checkpoint=args.checkpoint,
            corpus=args.corpus,
            output=args.output,
            seq_len=args.seq_len,
            stride=args.stride,
            batch_size=args.batch_size,
            alpha=args.alpha,
            weight_floor=args.weight_floor,
            device=device,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
