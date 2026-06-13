"""Learning-rate range test (Smith, 2015) for the GrimoireTransformer.

The trainer hard-codes ``peak_lr`` with no empirical basis.  This script finds
a good peak learning rate for *this* model and *this* corpus by sweeping the LR
exponentially from very small to very large over a few hundred steps and
watching the loss.  Early on the LR is too small and loss barely moves; in a
"sweet spot" the loss falls fastest; past it the loss explodes.  The LR at the
point of steepest loss decrease (divided by a safety factor) is a strong
estimate for the peak LR used in the warmup+cosine schedule.

Usage
-----
    python scripts/lr_range_test.py \\
        --corpus data/processed/corpus.bin \\
        --output lr_range_test.csv

    python scripts/lr_range_test.py --config train_config.json --num-iters 300

The model is built fresh (random init) for the test and thrown away; this does
not modify any checkpoint.  Reads a CSV of (step, lr, loss, smoothed_loss) and
prints a suggested peak LR.  If matplotlib is installed a PNG is also written.

Requirements
------------
    torch, numpy  (already core dependencies)
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from grimoire_ai.llm.data.collator import PaddingCollator
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID

DEFAULT_MODEL_CONFIG = {
    "vocab_size":  16384,
    "d_model":     512,
    "n_layers":    6,
    "n_heads":     8,
    "n_kv_heads":  2,
    "d_ff":        1408,
    "max_seq_len": 1024,
    "dropout":     0.1,
    "rope_theta":  10000.0,
}


def _param_groups(model: GrimoireTransformer) -> list[dict]:
    """Replicate the trainer's decay / no-decay AdamW parameter split."""
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return [
        {"params": decay,    "weight_decay": 0.1},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def run_lr_range_test(
    corpus_path: str,
    model_config: dict,
    min_lr: float = 1e-7,
    max_lr: float = 1.0,
    num_iters: int = 200,
    batch_size: int = 4,
    accumulate_steps: int = 1,
    diverge_factor: float = 4.0,
    smooth_beta: float = 0.98,
    device: Optional[str] = None,
) -> dict:
    """Run the LR sweep and return the recorded curve and a suggested LR.

    Args:
        corpus_path: Path to the tokenised ``.bin`` corpus.
        model_config: ``TransformerConfig`` kwargs for the throwaway model.
        min_lr: Starting (smallest) learning rate.
        max_lr: Ending (largest) learning rate.
        num_iters: Number of optimizer steps across the sweep.
        batch_size: Micro-batch size.
        accumulate_steps: Micro-batches per optimizer step.
        diverge_factor: Stop early once smoothed loss exceeds this multiple of
            the best smoothed loss (the loss has clearly blown up).
        smooth_beta: EMA coefficient for smoothing the noisy step loss.
        device: ``"cpu"`` / ``"cuda"`` (auto-detected when ``None``).

    Returns:
        A dict with ``lrs``, ``losses``, ``smoothed`` (lists, equal length),
        ``suggested_lr`` (float or ``None`` if not determinable), and
        ``steepest_lr`` (the LR at the steepest loss decrease, or ``None``).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = TransformerConfig(**model_config)
    model = GrimoireTransformer(config).to(device)
    model.train()

    dataset = TokenizedDataset(corpus_path, seq_len=config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PaddingCollator(pad_id=PAD_ID),
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(_param_groups(model), lr=min_lr, betas=(0.9, 0.95), eps=1e-8)

    # Geometric LR schedule from min_lr to max_lr across num_iters steps.
    mult = (max_lr / min_lr) ** (1.0 / max(num_iters - 1, 1))

    lrs: list[float] = []
    losses: list[float] = []
    smoothed: list[float] = []

    avg_loss = 0.0
    best_smooth = float("inf")
    data_iter = iter(loader)
    lr = min_lr
    non_blocking = device == "cuda"

    for i in range(num_iters):
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(accumulate_steps):
            try:
                input_ids, target_ids, attention_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                input_ids, target_ids, attention_mask = next(data_iter)
            input_ids      = input_ids.to(device, non_blocking=non_blocking)
            target_ids     = target_ids.to(device, non_blocking=non_blocking)
            attention_mask = attention_mask.to(device, non_blocking=non_blocking)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                target_ids.view(-1),
                ignore_index=PAD_ID,
            ) / accumulate_steps
            loss.backward()
            step_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # EMA smoothing with bias correction.
        avg_loss = smooth_beta * avg_loss + (1.0 - smooth_beta) * step_loss
        smooth = avg_loss / (1.0 - smooth_beta ** (i + 1))

        lrs.append(lr)
        losses.append(step_loss)
        smoothed.append(smooth)

        if math.isfinite(smooth):
            best_smooth = min(best_smooth, smooth)
        # Stop if the loss has exploded or gone non-finite.
        if not math.isfinite(smooth) or (i > 0 and smooth > diverge_factor * best_smooth):
            break

        lr *= mult

    steepest_lr, suggested_lr = _suggest_lr(lrs, smoothed)
    return {
        "lrs": lrs,
        "losses": losses,
        "smoothed": smoothed,
        "steepest_lr": steepest_lr,
        "suggested_lr": suggested_lr,
    }


def _suggest_lr(lrs: list[float], smoothed: list[float]) -> tuple[Optional[float], Optional[float]]:
    """Pick the LR at the steepest loss decrease and a safety-scaled suggestion.

    Returns ``(steepest_lr, suggested_lr)`` where ``suggested_lr`` is the
    steepest-point LR divided by 10 (the common fast.ai heuristic), or
    ``(None, None)`` when there are too few points to estimate a gradient.
    """
    if len(lrs) < 3:
        return None, None
    log_lrs = np.log10(np.asarray(lrs))
    loss = np.asarray(smoothed)
    # Numerical gradient of loss w.r.t. log10(lr); steepest decrease = min.
    grad = np.gradient(loss, log_lrs)
    idx = int(np.argmin(grad))
    steepest = float(lrs[idx])
    return steepest, steepest / 10.0


def _load_model_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return dict(DEFAULT_MODEL_CONFIG)
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return {**DEFAULT_MODEL_CONFIG, **cfg.get("model", {})}


def _write_csv(result: dict, output: str) -> None:
    rows = ["step,lr,loss,smoothed_loss"]
    for i, (lr, loss, sm) in enumerate(zip(result["lrs"], result["losses"], result["smoothed"])):
        rows.append(f"{i},{lr:.8g},{loss:.6f},{sm:.6f}")
    Path(output).write_text("\n".join(rows) + "\n", encoding="utf-8")


def _maybe_plot(result: dict, output_csv: str) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    png = str(Path(output_csv).with_suffix(".png"))
    plt.figure(figsize=(7, 4))
    plt.plot(result["lrs"], result["smoothed"])
    plt.xscale("log")
    plt.xlabel("learning rate")
    plt.ylabel("smoothed loss")
    if result["suggested_lr"]:
        plt.axvline(result["suggested_lr"], color="red", linestyle="--",
                    label=f"suggested {result['suggested_lr']:.2e}")
        plt.legend()
    plt.title("LR range test")
    plt.tight_layout()
    plt.savefig(png, dpi=120)
    plt.close()
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description="Learning-rate range test.")
    parser.add_argument("--corpus", default="data/processed/corpus.bin",
                        help="Tokenised corpus (.bin).")
    parser.add_argument("--config", default=None,
                        help="Optional train config JSON (its 'model' block is used).")
    parser.add_argument("--output", default="lr_range_test.csv",
                        help="Destination CSV (a .png is added if matplotlib is present).")
    parser.add_argument("--min-lr", type=float, default=1e-7)
    parser.add_argument("--max-lr", type=float, default=1.0)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=1)
    parser.add_argument("--device", default=None, help="cpu / cuda (auto if omitted).")
    args = parser.parse_args()

    try:
        result = run_lr_range_test(
            corpus_path=args.corpus,
            model_config=_load_model_config(args.config),
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            num_iters=args.num_iters,
            batch_size=args.batch_size,
            accumulate_steps=args.accumulate,
            device=args.device,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _write_csv(result, args.output)
    png = _maybe_plot(result, args.output)

    print(f"Swept {len(result['lrs'])} iterations.")
    if result["steepest_lr"]:
        print(f"  Steepest loss decrease at LR ≈ {result['steepest_lr']:.2e}")
        print(f"  Suggested peak LR (steepest / 10) ≈ {result['suggested_lr']:.2e}")
    else:
        print("  Not enough points to suggest an LR; try a wider range or more iters.")
    print(f"  Curve written to {args.output}" + (f" and {png}" if png else ""))


if __name__ == "__main__":
    main()
