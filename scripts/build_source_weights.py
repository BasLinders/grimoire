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
array is only valid for windows in that exact configuration. If you train
with a validation split, build the weights against the same train-only
region (see the Pre-train tab, which handles this automatically).

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
) -> None:
    doc_end_offsets, doc_weights = load_doc_weight_sidecars(
        corpus_path, doc_end_offsets_path, doc_weights_path
    )
    print(f"Loaded {len(doc_weights)} document weight(s)")

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
    args = parser.parse_args()

    try:
        build(
            corpus_path=args.corpus,
            seq_len=args.seq_len,
            stride=args.stride,
            output_path=args.output,
            doc_end_offsets_path=args.doc_end_offsets,
            doc_weights_path=args.doc_weights,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
