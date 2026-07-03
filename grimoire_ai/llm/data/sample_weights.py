"""Turn per-document source weights into per-window training sample weights.

Companion to the ``--weight-pattern`` flag on
``grimoire_ai.llm.data.preprocessing``. That flag tags each source document
with a weight (e.g. D&D-specific files weighted higher than bulk Gutenberg
literature) and writes two sidecar files next to the tokenized corpus:

    <corpus>.bin.doc_end_offsets.npy   cumulative end token-index of each document
    <corpus>.bin.doc_weights.npy       weight assigned to each document

This module converts those document-level weights into a per-window weights
array aligned to a ``TokenizedDataset``'s window order — the same array
shape and convention ``trainer.py``'s ``sample_weights`` /
``WeightedRandomSampler`` already expects, and that
``scripts/score_difficulty.py`` already produces for difficulty-based
weighting. Used by both ``scripts/build_source_weights.py`` (CLI) and the
Gradio UI's Pre-train tab, so the two stay in exact agreement.

Each window is assigned the weight of whichever document contains its start
offset. Windows are typically far shorter than documents, so the vast
majority fall entirely within one document; a window that straddles a
document boundary is labeled by whichever document it starts in.
"""

from pathlib import Path
from typing import Sequence

import numpy as np


def compute_window_weights(
    offsets: Sequence[int],
    doc_end_offsets: np.ndarray,
    doc_weights: np.ndarray,
) -> np.ndarray:
    """Assign each window offset the weight of its containing document.

    Args:
        offsets: Absolute start offset of every window, in dataset order —
            typically ``TokenizedDataset.offsets``.
        doc_end_offsets: Cumulative end token-index of each document,
            ascending (as written by ``preprocessing.py``'s
            ``--weight-pattern``).
        doc_weights: Weight assigned to each document, same length/order as
            ``doc_end_offsets``.

    Returns:
        A ``float32`` array the same length as ``offsets``.

    Raises:
        ValueError: If ``doc_end_offsets`` and ``doc_weights`` differ in length.
    """
    if len(doc_end_offsets) != len(doc_weights):
        raise ValueError(
            f"doc_end_offsets has {len(doc_end_offsets)} entries but "
            f"doc_weights has {len(doc_weights)} — did the corpus change "
            "since these sidecar files were written?"
        )
    window_starts = np.asarray(offsets, dtype=np.int64)
    # searchsorted(..., side="right") on cumulative end-offsets gives the
    # index of the document each window start falls into.
    doc_idx = np.searchsorted(doc_end_offsets, window_starts, side="right")
    doc_idx = np.clip(doc_idx, 0, len(doc_weights) - 1)
    return doc_weights[doc_idx].astype(np.float32)


def default_sidecar_paths(corpus_path: str) -> tuple[Path, Path]:
    """Return the default (doc_end_offsets, doc_weights) sidecar paths for a corpus."""
    corpus_p = Path(corpus_path)
    return (
        Path(str(corpus_p) + ".doc_end_offsets.npy"),
        Path(str(corpus_p) + ".doc_weights.npy"),
    )


def load_doc_weight_sidecars(
    corpus_path: str,
    doc_end_offsets_path: str | None = None,
    doc_weights_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the per-document weight sidecar files for a preprocessed corpus.

    Args:
        corpus_path: Path to the tokenized ``.bin`` corpus.
        doc_end_offsets_path: Override path (default: ``<corpus>.doc_end_offsets.npy``).
        doc_weights_path: Override path (default: ``<corpus>.doc_weights.npy``).

    Returns:
        ``(doc_end_offsets, doc_weights)`` as loaded numpy arrays.

    Raises:
        FileNotFoundError: If either sidecar file is missing — most likely
            because ``grimoire-preprocess`` was run without
            ``--weight-pattern``.
    """
    default_offsets, default_weights = default_sidecar_paths(corpus_path)
    offsets_p = Path(doc_end_offsets_path) if doc_end_offsets_path else default_offsets
    weights_p = Path(doc_weights_path) if doc_weights_path else default_weights

    for p in (offsets_p, weights_p):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Re-run grimoire-preprocess with at least "
                "one --weight-pattern flag to generate it."
            )

    return np.load(str(offsets_p)), np.load(str(weights_p))
