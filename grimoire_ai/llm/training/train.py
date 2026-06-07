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
            "num_workers":     0
        }
    }

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
    checkpoint_dir  = cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    resume_from     = args.resume or cfg.get("resume_from", None)

    # --- Build model --------------------------------------------------------
    model_config = TransformerConfig(**model_cfg_dict)
    print(f"Model config: {model_config}")
    model = GrimoireTransformer(model_config)
    print(f"Parameters: {model.num_parameters():,}")

    # --- Build dataset ------------------------------------------------------
    try:
        dataset = TokenizedDataset(
            corpus_path=corpus_path,
            seq_len=model_config.max_seq_len,
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
    print(f"Dataset: {len(dataset):,} windows from {corpus_path}")

    # --- Build trainer and run --------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        checkpoint_dir=checkpoint_dir,
        **train_cfg_dict,
    )
    trainer.train(resume_from=resume_from)


if __name__ == "__main__":
    main()
