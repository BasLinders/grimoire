"""Entry point for training the GrimoireTransformer.

Run from the repository root:

    python -m grimoire.llm.training.train

Or with a custom config file:

    python -m grimoire.llm.training.train --config path/to/train_config.json

Or resuming from a checkpoint:

    python -m grimoire.llm.training.train --resume checkpoints/step_0001000.pt

Config file format (JSON)
--------------------------
All keys are optional; missing keys fall back to the defaults shown here.

    {
        "corpus_path":     "data/processed/corpus.bin",
        "val_corpus_path": null,
        "checkpoint_dir":  "checkpoints",
        "resume_from":     null,

        "model": {
            "vocab_size":  16384,
            "d_model":     512,
            "n_layers":    6,
            "n_heads":     8,
            "n_kv_heads":  2,
            "d_ff":        1408,
            "max_seq_len": 1024,
            "dropout":     0.1,
            "rope_theta":  10000.0
        },

        "training": {
            "peak_lr":         3e-4,
            "warmup_steps":    500,
            "total_steps":     10000,
            "batch_size":      4,
            "accumulate_steps": 8,
            "log_every":       50,
            "save_every":      1000,
            "num_workers":     0,
            "val_split":       0.0,
            "eval_every":      1000,
            "eval_batches":    50
        }
    }

Validation loss
---------------
Set ``"val_split"`` (e.g. 0.01) under ``"training"`` to hold out the final
fraction of the corpus as a validation set; a validation loss is then logged
every ``"eval_every"`` steps.  Alternatively point ``"val_corpus_path"`` at a
separate ``.bin`` file.  Watching train-vs-val divergence is the simplest way
to tell a high-LR plateau (both still falling) from overfitting (val rising).

Device selection
----------------
The device is chosen automatically: CUDA if available, otherwise CPU.
On Windows with an RTX card, PyTorch must be installed with CUDA support:

    pip install torch --index-url https://download.pytorch.org/whl/cu124
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CORPUS_PATH    = "data/processed/corpus.bin"
DEFAULT_CHECKPOINT_DIR = "checkpoints"

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

DEFAULT_TRAINING_CONFIG = {
    "peak_lr":          3e-4,
    "warmup_steps":     500,
    "total_steps":      10_000,
    "batch_size":       4,
    "accumulate_steps": 8,
    "log_every":        50,
    "save_every":       1000,
    "num_workers":      0,
    "val_split":        0.0,
    "eval_every":       1000,
    "eval_batches":     50,
}


def _load_config(path: str) -> dict:
    """Read a JSON training config file and return it as a dict.

    Args:
        path: Path to the JSON config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    """Parse arguments, build all objects, and start training."""
    parser = argparse.ArgumentParser(
        description="Train the GrimoireTransformer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to a JSON training config file.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Path to a checkpoint .pt file to resume from.",
    )
    parser.add_argument(
        "--corpus", default=None,
        help="Path to the corpus .bin file (overrides config).",
    )
    args = parser.parse_args()

    # --- Load config --------------------------------------------------------
    cfg: dict = {}
    if args.config:
        try:
            cfg = _load_config(args.config)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

    model_cfg_dict  = {**DEFAULT_MODEL_CONFIG,  **cfg.get("model", {})}
    train_cfg_dict  = {**DEFAULT_TRAINING_CONFIG, **cfg.get("training", {})}
    corpus_path     = args.corpus or cfg.get("corpus_path", DEFAULT_CORPUS_PATH)
    val_corpus_path = cfg.get("val_corpus_path", None)
    checkpoint_dir  = cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    resume_from     = args.resume or cfg.get("resume_from", None)

    # ``val_split`` is a data-splitting concern handled here, not a Trainer
    # constructor argument — pop it out before forwarding the rest.
    val_split = train_cfg_dict.pop("val_split", 0.0)

    # --- Build model --------------------------------------------------------
    model_config = TransformerConfig(**model_cfg_dict)
    print(f"Model config: {model_config}")
    model = GrimoireTransformer(model_config)
    print(f"Parameters: {model.num_parameters():,}")

    # --- Build dataset(s) ---------------------------------------------------
    seq_len = model_config.max_seq_len
    try:
        train_dataset, val_dataset = _build_datasets(
            corpus_path=corpus_path,
            val_corpus_path=val_corpus_path,
            val_split=val_split,
            seq_len=seq_len,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError loading corpus: {exc}", file=sys.stderr)
        print(
            "Run preprocessing first:\n"
            "  python -m grimoire.llm.data.preprocessing "
            "--input data/raw/ --output data/processed/corpus.bin "
            "--vocab data/tokenizer/bpe.json",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Train dataset: {len(train_dataset):,} windows from {corpus_path}")
    if val_dataset is not None:
        print(f"Val dataset:   {len(val_dataset):,} windows")

    # --- Build trainer and run --------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        **train_cfg_dict,
    )
    trainer.train(resume_from=resume_from)


def _build_datasets(
    corpus_path: str,
    val_corpus_path: Optional[str],
    val_split: float,
    seq_len: int,
) -> tuple[TokenizedDataset, Optional[TokenizedDataset]]:
    """Build the train dataset and, optionally, a validation dataset.

    Validation data comes from one of two sources, in priority order:

    1. ``val_corpus_path`` — a separate ``.bin`` file used in full.  The
       training corpus is used in full as well.
    2. ``val_split`` — a fraction (0 < f < 1) of the *tail* of the training
       corpus is held out.  The split is by token index, so the train and
       validation regions share no tokens (no window overlap / leakage).

    If neither is set, the validation dataset is ``None`` and no evaluation
    runs.

    Args:
        corpus_path: Path to the training corpus ``.bin`` file.
        val_corpus_path: Optional path to a separate validation corpus.
        val_split: Fraction of the training corpus tail to hold out when no
            separate validation corpus is given.  Ignored if ``<= 0``.
        seq_len: Sequence length for each window.

    Returns:
        ``(train_dataset, val_dataset)`` where ``val_dataset`` may be ``None``.
    """
    if val_corpus_path:
        train_dataset = TokenizedDataset(corpus_path=corpus_path, seq_len=seq_len)
        val_dataset = TokenizedDataset(corpus_path=val_corpus_path, seq_len=seq_len)
        return train_dataset, val_dataset

    if val_split and val_split > 0.0:
        # Peek the token count to choose a clean split boundary, then build
        # train over [0, split) and validation over [split, end).
        n_tokens = len(np.memmap(corpus_path, dtype=np.int32, mode="r"))
        split = int(n_tokens * (1.0 - val_split))
        train_dataset = TokenizedDataset(
            corpus_path=corpus_path, seq_len=seq_len, end=split
        )
        val_dataset = TokenizedDataset(
            corpus_path=corpus_path, seq_len=seq_len, start=split
        )
        return train_dataset, val_dataset

    return TokenizedDataset(corpus_path=corpus_path, seq_len=seq_len), None


if __name__ == "__main__":
    main()
