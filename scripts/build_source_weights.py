"""Build a per-window sample_weights.npy from source-based document weights.

Companion to the ``--weight-pattern`` flag on
``grimoire_ai.llm.data.preprocessing`` (the ``grimoire-preprocess`` CLI).
That flag tags each source document with a weight (e.g. D&D-specific files
weighted higher than bulk Gutenberg literature) and writes two small sidecar
files next to the tokenized corpus:

    <corpus>.bin.doc_end_offsets.npy   cumulative end token-index of each document
    <corpus>.bin.doc_weights.npy       weight assigned to each document

This script turns those document-level weights into a per-window weights
array aligned to ``TokenizedDataset``'s window order (the same array shape
and convention already used by ``scripts/score_difficulty.py`` for
difficulty-based weighting), using
``grimoire_ai.llm.data.sample_weights.compute_window_weights`` -- the same
function the Gradio UI's Pre-train tab uses, so both stay in agreement.

Why this exists
----------------
Nothing about *this* mechanism is new: ``trainer.py`` already builds a
``WeightedRandomSampler`` whenever a ``sample_weights_path`` is configured,
and ``scripts/score_difficulty.py`` already produces that array for
difficulty-based weighting. This script produces the same kind of array from
a different (source/subject-based, not difficulty-based) signal, so it can
be pointed at the exact same ``sample_weights_path`` config key with no
changes to the training loop at all. You can even average or otherwise
combine this with a difficulty-weights array from score_difficulty.py if you
want both effects at once (not done automatically here -- do that
multiplication/combination yourself before training if wanted).

Usage
-----
    python scripts/build_source_weights.py \\
        --corpus data/processed/corpus.bin \\
        --seq-len 1024 --stride 512 \\
        --output data/processed/source_weights.npy

The seq_len/stride here MUST match what you pass to TokenizedDataset at
training time (i.e. whatever grimoire-train actually uses) -- the weights
array is only valid for windows in that exact configuration.

If you train with a validation split, pass --val-split with the same value
your training config uses (e.g. --val-split 0.01) -- this excludes the same
scatter of validation blocks (see train.py's _split_blocks) that
``grimoire-train`` (via ``_build_datasets``) actually trains on, using the
same function the Pre-train tab's "Build sample weights from tags" button
uses. Omitting --val-split (or passing 0) scores the full corpus, which only
matches a training run with no validation split.

If you also pass --val-stratified on grimoire-train (so validation holds out
a proportional slice of every --weight-pattern tier rather than scattering
blocks corpus-wide -- see train.py's _split_by_tier), pass --val-stratified
here too. Mismatching this flag between the two commands silently changes
which windows land in train vs val and causes the same window-count
mismatch --val-split alone would.

Requirements
------------
    numpy  (already a core dependency)
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.data.sample_weights import (
    compute_window_weights,
    load_doc_weight_sidecars,
)


def build(
    corpus_path: str,
    seq_len: int,
    stride: int,
    output_path: str,
    doc_end_offsets_path: str | None = None,
    doc_weights_path: str | None = None,
    val_split: float = 0.0,
    val_stratified: bool = False,
) -> None:
    doc_end_offsets, doc_weights = load_doc_weight_sidecars(
        corpus_path, doc_end_offsets_path, doc_weights_path
    )
    print(f"Loaded {len(doc_weights)} document weight(s)")

    if val_split and val_split > 0.0:
        # Match _build_datasets exactly (same function grimoire-train and the
        # UI's "Build sample weights from tags" button use) so the train
        # region -- and therefore the window count/order -- lines up bit for
        # bit with what training will actually see. Note _build_datasets has
        # no stride parameter of its own; it always uses TokenizedDataset's
        # default (seq_len // 2), so --stride is ignored in this branch.
        from grimoire_ai.llm.training.train import _build_datasets

        dataset, _ = _build_datasets(
            corpus_path=corpus_path, val_corpus_path=None,
            val_split=val_split, seq_len=seq_len, val_stratified=val_stratified,
        )
        print(f"Train region has {len(dataset):,} windows "
              f"(seq_len={seq_len}, val_split={val_split}, "
              f"val_stratified={val_stratified})")
    else:
        dataset = TokenizedDataset(corpus_path, seq_len=seq_len, stride=stride)
        print(f"Corpus has {len(dataset):,} windows (seq_len={seq_len}, stride={stride})")

    window_weights = compute_window_weights(dataset.offsets, doc_end_offsets, doc_weights)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), window_weights)

    unique, counts = np.unique(window_weights, return_counts=True)
    print(f"\nDone. Wrote {len(window_weights):,} window weights to {out_path}")
    print("  weight -> window count:")
    for w, c in sorted(zip(unique.tolist(), counts.tolist())):
        print(f"    {w:>6.2f} -> {c:,} ({100 * c / len(window_weights):.1f}%)")
    print(
        '\nSet "sample_weights_path" in your training config to this file. '
        "It must be rebuilt if you change the corpus, seq_len, or stride, "
        "or re-run grimoire-preprocess with different --weight-pattern rules."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-window sample_weights.npy from source-based document weights."
    )
    parser.add_argument("--corpus", default="data/processed/corpus.bin",
                        help="Tokenized corpus (.bin) that grimoire-preprocess wrote.")
    parser.add_argument("--seq-len", type=int, required=True,
                        help="Window length -- must match what training uses.")
    parser.add_argument("--stride", type=int, required=True,
                        help="Window stride -- must match what training uses.")
    parser.add_argument("--output", default="data/processed/source_weights.npy",
                        help="Destination .npy weights file.")
    parser.add_argument("--doc-end-offsets", default=None,
                        help="Override path to the doc_end_offsets.npy sidecar "
                             "(default: <corpus>.doc_end_offsets.npy).")
    parser.add_argument("--doc-weights", default=None,
                        help="Override path to the doc_weights.npy sidecar "
                             "(default: <corpus>.doc_weights.npy).")
    parser.add_argument("--val-split", type=float, default=0.0,
                        help="Fraction of the corpus held out for validation "
                             "in your training config (scattered across many "
                             "blocks, not one contiguous chunk -- see "
                             "train.py's _split_blocks). Must match exactly, or "
                             "the resulting window count won't align with "
                             "training. 0 (default) scores the full corpus.")
    parser.add_argument("--val-stratified", action="store_true", default=False,
                        help="Must match --val-stratified on grimoire-train "
                             "exactly. Stratifies the val split by weight "
                             "tier (train.py's _split_by_tier) instead of "
                             "scattering blocks corpus-wide.")
    args = parser.parse_args()

    try:
        build(
            corpus_path=args.corpus,
            seq_len=args.seq_len,
            stride=args.stride,
            output_path=args.output,
            doc_end_offsets_path=args.doc_end_offsets,
            doc_weights_path=args.doc_weights,
            val_split=args.val_split,
            val_stratified=args.val_stratified,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
