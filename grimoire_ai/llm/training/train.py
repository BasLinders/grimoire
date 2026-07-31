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
            "eval_batches":    50,
            "early_stop_enabled":    false,
            "early_stop_patience":   3,
            "early_stop_bootstraps": 1000,
            "early_stop_alpha":      0.05,
            "swa_enabled":           false,
            "swa_start_frac":        0.75
        }
    }

Stochastic Weight Averaging (SWA)
---------------------------------
Set ``"swa_enabled"`` to true to average the model weights from the tail of
training (the final ``1 - swa_start_frac`` of steps).  The averaged weights are
written to ``{checkpoint_dir}/swa.pt`` at the end of the run and usually
generalise slightly better than the final iterate, at the cost of one extra
in-memory copy of the model.

Early stopping
--------------
When ``"early_stop_enabled"`` is true (and a validation set is configured via
``"val_split"`` or ``"val_corpus_path"``), training halts once the validation
loss stops improving beyond its bootstrap confidence band for
``"early_stop_patience"`` consecutive evaluations.  This is a noise-aware
alternative to a fixed step count: it avoids both stopping on a lucky dip and
wasting compute once genuine progress has plateaued.

Validation loss
---------------
Set ``"val_split"`` (e.g. 0.01) under ``"training"`` to hold out that
fraction of the corpus -- scattered across many small blocks, not one
contiguous chunk (see ``_split_blocks``) -- as a validation set; a
validation loss is then logged every ``"eval_every"`` steps.  Alternatively
point ``"val_corpus_path"`` at a separate ``.bin`` file.  Watching
train-vs-val divergence is the simplest way to tell a high-LR plateau (both
still falling) from overfitting (val rising).

Device selection
----------------
The device is chosen automatically: CUDA if available, otherwise MPS on
Apple Silicon Macs, otherwise CPU.
On Windows with an RTX card, PyTorch must be installed with CUDA support:

    pip install torch --index-url https://download.pytorch.org/whl/cu124

MPS ships in the standard PyPI ``torch`` wheel, so no separate install step
is needed on macOS.
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
    "early_stop_enabled":    False,
    "early_stop_patience":   3,
    "early_stop_bootstraps": 1000,
    "early_stop_alpha":      0.05,
    "swa_enabled":           False,
    "swa_start_frac":        0.75,
    "gradient_checkpointing": False,
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
    # Trainer prints Unicode symbols (e.g. the checkpoint-saved arrow) that
    # crash with UnicodeEncodeError on Windows' default cp1252 console
    # encoding -- reconfigure unconditionally so a multi-hour run never dies
    # on a print statement, of all things.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
    parser.add_argument(
        "--gradient-checkpointing", action="store_true", default=False,
        help="Enable gradient checkpointing to reduce VRAM at ~20%% training speed cost.",
    )
    parser.add_argument(
        "--val-stratified", action="store_true", default=False,
        help="Stratify the validation split by --weight-pattern document "
             "tags instead of scattering blocks corpus-wide, so every "
             "weight tier is represented in validation proportional to its "
             "own size. Requires the corpus to have been tagged with "
             "--weight-pattern during preprocessing. Ignored if val_split "
             "is 0.",
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
    if args.gradient_checkpointing:
        train_cfg_dict["gradient_checkpointing"] = True
    corpus_path     = args.corpus or cfg.get("corpus_path", DEFAULT_CORPUS_PATH)
    val_corpus_path = cfg.get("val_corpus_path", None)
    checkpoint_dir  = cfg.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    resume_from     = args.resume or cfg.get("resume_from", None)
    sample_weights_path = cfg.get("sample_weights_path", None)

    # ``val_split`` is a data-splitting concern handled here, not a Trainer
    # constructor argument — pop it out before forwarding the rest.
    val_split = train_cfg_dict.pop("val_split", 0.0)
    val_stratified = train_cfg_dict.pop("val_stratified", False) or args.val_stratified

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
            val_stratified=val_stratified,
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

    # --- Optional difficulty weights --------------------------------------
    sample_weights = None
    if sample_weights_path:
        try:
            sample_weights = np.load(sample_weights_path)
        except OSError as exc:
            print(f"\nError loading sample weights: {exc}", file=sys.stderr)
            print(
                "Generate them first:\n"
                "  python scripts/score_difficulty.py --checkpoint <ckpt> "
                "--corpus <corpus.bin> --output <weights.npy>",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Sample weights: {len(sample_weights):,} from {sample_weights_path}")

    # --- Build trainer and run --------------------------------------------
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=checkpoint_dir,
        sample_weights=sample_weights,
        **train_cfg_dict,
    )
    trainer.train(resume_from=resume_from)


def _build_datasets(
    corpus_path: str,
    val_corpus_path: Optional[str],
    val_split: float,
    seq_len: int,
    val_stratified: bool = False,
) -> tuple[TokenizedDataset, Optional[TokenizedDataset]]:
    """Build the train dataset and, optionally, a validation dataset.

    Validation data comes from one of three sources, in priority order:

    1. ``val_corpus_path`` — a separate ``.bin`` file used in full.  The
       training corpus is used in full as well.
    2. ``val_stratified`` (with ``val_split`` and ``--weight-pattern`` tag
       sidecars present) — holds out ``val_split`` fraction *within each
       weight tier separately* (see ``_split_by_tier``), so every tier is
       represented in validation proportional to its own size instead of
       leaving thin categories to chance.
    3. ``val_split`` alone — a fraction (0 < f < 1) of the training corpus,
       held out as a scatter of many small blocks across the whole token
       range (see ``_split_blocks``) rather than one contiguous tail. Each
       block is windowed independently, so train and validation regions
       still share no tokens (no window overlap / leakage).

    If none is set, the validation dataset is ``None`` and no evaluation
    runs.

    Args:
        corpus_path: Path to the training corpus ``.bin`` file.
        val_corpus_path: Optional path to a separate validation corpus.
        val_split: Fraction of the training corpus to hold out when no
            separate validation corpus is given. Ignored if ``<= 0``.
        seq_len: Sequence length for each window.
        val_stratified: If ``True`` (and ``val_split > 0``), stratify the
            split by the corpus's ``--weight-pattern`` document tags instead
            of scattering blocks corpus-wide. Requires the
            ``<corpus_path>.doc_end_offsets.npy`` / ``.doc_weights.npy``
            sidecar files (written by ``grimoire-preprocess
            --weight-pattern``); raises ``FileNotFoundError`` if they're
            missing, since a stratified split is meaningless without them.

    Returns:
        ``(train_dataset, val_dataset)`` where ``val_dataset`` may be ``None``.
    """
    if val_corpus_path:
        train_dataset = TokenizedDataset(corpus_path=corpus_path, seq_len=seq_len)
        val_dataset = TokenizedDataset(corpus_path=val_corpus_path, seq_len=seq_len)
        return train_dataset, val_dataset

    if val_split and val_split > 0.0:
        if val_stratified:
            from grimoire_ai.llm.data.sample_weights import load_doc_weight_sidecars

            doc_end_offsets, doc_weights = load_doc_weight_sidecars(corpus_path)
            train_regions, val_regions = _split_by_tier(
                doc_end_offsets, doc_weights, val_split, seq_len
            )
        else:
            n_tokens = len(np.memmap(corpus_path, dtype=np.int32, mode="r"))
            train_regions, val_regions = _split_blocks(n_tokens, val_split, seq_len)
        train_dataset = TokenizedDataset(
            corpus_path=corpus_path, seq_len=seq_len, regions=train_regions
        )
        val_dataset = TokenizedDataset(
            corpus_path=corpus_path, seq_len=seq_len, regions=val_regions
        )
        return train_dataset, val_dataset

    return TokenizedDataset(corpus_path=corpus_path, seq_len=seq_len), None


# Fixed (not time-based) seed so _build_datasets is a pure function of its
# arguments -- a training run and a separate scripts/build_source_weights.py
# invocation must independently derive the identical split for the resulting
# sample_weights.npy to line up with the training windows.
_VAL_SPLIT_SEED = 0
_VAL_SPLIT_N_BLOCKS = 500


def _split_blocks(
    n_tokens: int, val_split: float, seq_len: int,
    n_blocks: int = _VAL_SPLIT_N_BLOCKS, seed: int = _VAL_SPLIT_SEED,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition the corpus into many contiguous blocks and randomly assign
    ~val_split fraction of them to validation.

    Corpus files are concatenated in alphabetically-sorted order
    (``preprocessing.py``'s ``_collect_text_files``), so a single contiguous
    "hold out the last N%" split -- the previous behavior -- ends up
    validating on whatever happens to sort last, e.g. entirely on a run of
    Wikipedia/Wikibooks stub articles rather than a representative sample of
    the corpus. Scattering many small blocks across the whole token range
    instead makes the held-out set a much more representative sample, at the
    cost of losing a small number of windows at each of the extra block
    boundaries (no window straddles a boundary, same no-leakage guarantee as
    the single-split version, just paid at ``n_blocks`` boundaries instead
    of 1 -- roughly ``n_blocks * seq_len`` tokens' worth of windows, a small
    fraction of a real corpus).

    Args:
        n_tokens: Total tokens in the corpus.
        val_split: Fraction of blocks (by count, not token weight) to assign
            to validation.
        seq_len: Used only to keep blocks from being pointlessly smaller
            than a single window; does not otherwise affect the split.
        n_blocks: Number of contiguous blocks to partition the corpus into.
        seed: Fixed RNG seed for reproducibility across separate processes.

    Returns:
        ``(train_regions, val_regions)``, each a list of ``(start, end)``
        token-index pairs suitable for ``TokenizedDataset(regions=...)``.
    """
    n_blocks = max(2, min(n_blocks, n_tokens // max(seq_len + 1, 1)))
    boundaries = np.linspace(0, n_tokens, n_blocks + 1).astype(np.int64)
    block_ranges = list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))

    n_val_blocks = min(n_blocks - 1, max(1, round(n_blocks * val_split)))
    rng = np.random.RandomState(seed)
    val_idx = set(rng.choice(n_blocks, size=n_val_blocks, replace=False).tolist())

    train_regions = [r for i, r in enumerate(block_ranges) if i not in val_idx]
    val_regions = [r for i, r in enumerate(block_ranges) if i in val_idx]
    return train_regions, val_regions


def _split_by_tier(
    doc_end_offsets: np.ndarray,
    doc_weights: np.ndarray,
    val_split: float,
    seq_len: int,
    seed: int = _VAL_SPLIT_SEED,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split the corpus into train/val, holding out ``val_split`` fraction
    *within each --weight-pattern tier separately*.

    ``_split_blocks`` scatters val blocks uniformly across raw token
    position, which is representative of the corpus as a whole but gives no
    guarantee that a thin category (e.g. a tier that's only ~10% of the
    corpus) ends up with any windows in val at all -- a small number of
    random blocks can miss it entirely by chance. Stratifying by tier
    guarantees every tier present in the corpus gets its own proportional
    share of validation windows, so per-tier evaluation (e.g. checking
    whether up-weighted content actually improved) doesn't depend on luck.

    Splits at the document level, not sub-document blocks: within each
    tier, documents are shuffled (fixed seed) and greedily assigned to val
    until val's accumulated token count reaches ``val_split`` fraction of
    that tier's total, the rest go to train. Held-out documents may be
    scattered anywhere in the corpus -- since each document's own region is
    contiguous and windowed independently (same as any other region list),
    train and validation still share no tokens.

    Args:
        doc_end_offsets: Cumulative end token-index of each document,
            ascending (as written by ``preprocessing.py``'s
            ``--weight-pattern``, and loaded via
            ``sample_weights.load_doc_weight_sidecars``).
        doc_weights: Weight tag assigned to each document, same length/order
            as ``doc_end_offsets``.
        val_split: Target fraction of each tier's tokens to hold out.
        seq_len: Used only to skip tiers too small to hold out a single
            window's worth of validation data.
        seed: Fixed RNG seed for reproducibility across separate processes
            (training and ``scripts/build_source_weights.py`` must
            independently derive the identical split).

    Returns:
        ``(train_regions, val_regions)``, each a list of ``(start, end)``
        token-index pairs suitable for ``TokenizedDataset(regions=...)``.
    """
    doc_starts = np.concatenate(([0], doc_end_offsets[:-1]))
    doc_regions = list(zip(doc_starts.tolist(), doc_end_offsets.tolist()))

    tiers: dict[float, list[tuple[int, int]]] = {}
    for region, weight in zip(doc_regions, doc_weights.tolist()):
        tiers.setdefault(weight, []).append(region)

    rng = np.random.RandomState(seed)
    train_regions: list[tuple[int, int]] = []
    val_regions: list[tuple[int, int]] = []

    for tier in sorted(tiers):
        tier_docs = tiers[tier]
        tier_tokens = sum(e - s for s, e in tier_docs)
        # A tier too small to hold out anything meaningful, or reduced to a
        # single document, can't be split without either losing all val
        # coverage or all train coverage for it -- keep it entirely in
        # train, since training coverage matters more than validation
        # coverage for a tier this thin.
        if tier_tokens < seq_len + 1 or len(tier_docs) <= 1:
            train_regions.extend(tier_docs)
            continue

        order = rng.permutation(len(tier_docs))
        target_val_tokens = tier_tokens * val_split
        val_tokens_so_far = 0
        tier_train: list[tuple[int, int]] = []
        tier_val: list[tuple[int, int]] = []
        for idx in order:
            region = tier_docs[idx]
            region_tokens = region[1] - region[0]
            if val_tokens_so_far < target_val_tokens:
                tier_val.append(region)
                val_tokens_so_far += region_tokens
            else:
                tier_train.append(region)
        # Guarantee at least one document survives on each side if the tier
        # has more than one -- an all-or-nothing split defeats the purpose
        # of stratifying by tier in the first place.
        if len(tier_docs) > 1 and not tier_train:
            tier_train.append(tier_val.pop())
        train_regions.extend(tier_train)
        val_regions.extend(tier_val)

    return train_regions, val_regions


if __name__ == "__main__":
    main()
