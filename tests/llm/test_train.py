"""Tests for grimoire_ai.llm.training.train's dataset-building helpers.

Gate criteria:
- _split_blocks produces disjoint train/val regions covering the whole corpus.
- The val fraction (by token count) is close to the requested val_split.
- The split is deterministic across repeated calls (required so a training
  run and a separate scripts/build_source_weights.py invocation agree).
- Val blocks are scattered rather than concentrated at one end of the corpus.
- _split_by_tier holds out val_split fraction within *every* weight tier
  separately, so a thin tier can't end up with zero validation windows.
"""

import numpy as np

from grimoire_ai.llm.training.train import _split_blocks, _split_by_tier


def test_split_blocks_no_overlap_and_full_coverage() -> None:
    """Train and val regions must be disjoint and together cover [0, n_tokens)."""
    n_tokens = 1_000_000
    train_regions, val_regions = _split_blocks(n_tokens, val_split=0.1, seq_len=1024)

    all_regions = sorted(train_regions + val_regions)
    assert all_regions[0][0] == 0
    assert all_regions[-1][1] == n_tokens
    for (_, end_a), (start_b, _) in zip(all_regions, all_regions[1:]):
        assert end_a == start_b, "Blocks must be contiguous with no gaps or overlap."


def test_split_blocks_val_fraction_close_to_requested() -> None:
    """The val region's token share should approximate val_split."""
    n_tokens = 10_000_000
    val_split = 0.05
    _, val_regions = _split_blocks(n_tokens, val_split=val_split, seq_len=1024)
    val_tokens = sum(e - s for s, e in val_regions)
    assert abs(val_tokens / n_tokens - val_split) < 0.02


def test_split_blocks_deterministic() -> None:
    """Repeated calls with the same arguments must produce an identical split."""
    n_tokens = 5_000_000
    t1, v1 = _split_blocks(n_tokens, val_split=0.01, seq_len=1024)
    t2, v2 = _split_blocks(n_tokens, val_split=0.01, seq_len=1024)
    assert t1 == t2
    assert v1 == v2


def test_split_blocks_val_is_scattered_not_one_contiguous_tail() -> None:
    """Val blocks should be spread across the corpus, not one chunk at the end.

    This is the actual bug being fixed: holding out a single contiguous tail
    over-represents whatever content happens to sort last. With many blocks,
    val regions should appear at multiple, non-adjacent locations.
    """
    n_tokens = 100_000_000
    _, val_regions = _split_blocks(n_tokens, val_split=0.01, seq_len=1024)
    assert len(val_regions) > 1, "Expected val blocks scattered across multiple locations."
    starts = sorted(s for s, _ in val_regions)
    # Not all val blocks clustered at the very end of the corpus.
    assert starts[0] < n_tokens * 0.5


def test_split_blocks_at_least_one_train_block_remains() -> None:
    """Even a large val_split must leave at least one train block."""
    n_tokens = 1_000_000
    train_regions, val_regions = _split_blocks(n_tokens, val_split=0.99, seq_len=1024)
    assert len(train_regions) >= 1
    assert len(val_regions) >= 1


# ---------------------------------------------------------------------------
# _split_by_tier
# ---------------------------------------------------------------------------

def _fake_corpus(tier_doc_sizes: dict[float, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic doc_end_offsets/doc_weights arrays for a set of tiers.

    ``tier_doc_sizes`` maps a weight tier to a list of document token counts.
    Documents are laid out sequentially in the order tiers are given, mimicking
    how real corpus documents are interleaved by filename rather than grouped
    by tier.
    """
    end_offsets: list[int] = []
    weights: list[float] = []
    cursor = 0
    # Interleave one doc from each tier at a time so tiers aren't contiguous
    # in the corpus -- matches how real --weight-pattern categories are
    # scattered across an alphabetically-sorted file list.
    max_docs = max(len(sizes) for sizes in tier_doc_sizes.values())
    for i in range(max_docs):
        for tier, sizes in tier_doc_sizes.items():
            if i < len(sizes):
                cursor += sizes[i]
                end_offsets.append(cursor)
                weights.append(tier)
    return np.array(end_offsets, dtype=np.int64), np.array(weights, dtype=np.float32)


def test_split_by_tier_no_overlap() -> None:
    """Train and val regions must be disjoint regardless of tier."""
    doc_end_offsets, doc_weights = _fake_corpus({
        0.5: [5000] * 20,
        1.0: [5000] * 20,
        1.75: [5000] * 20,
    })
    train_regions, val_regions = _split_by_tier(doc_end_offsets, doc_weights, val_split=0.2, seq_len=64)

    all_regions = sorted(train_regions + val_regions)
    for (start_a, end_a), (start_b, end_b) in zip(all_regions, all_regions[1:]):
        assert end_a <= start_b, "Regions must not overlap."


def test_split_by_tier_every_tier_represented_in_val() -> None:
    """The actual bug being fixed: a thin tier must not end up with zero val docs.

    One tier here has only 1/10th the documents of the others -- with the
    corpus-wide scattering approach this could easily be missed by chance;
    stratifying per tier must not let that happen.
    """
    doc_end_offsets, doc_weights = _fake_corpus({
        0.5: [5000] * 40,
        1.0: [5000] * 40,
        1.75: [5000] * 4,   # thin tier
    })
    _, val_regions = _split_by_tier(doc_end_offsets, doc_weights, val_split=0.1, seq_len=64)

    val_weights = set()
    doc_starts = np.concatenate(([0], doc_end_offsets[:-1]))
    region_to_weight = {
        (int(s), int(e)): float(w)
        for s, e, w in zip(doc_starts, doc_end_offsets, doc_weights)
    }
    for region in val_regions:
        val_weights.add(region_to_weight[region])

    assert val_weights == {0.5, 1.0, 1.75}, (
        f"Expected every tier represented in val, got {val_weights}"
    )


def test_split_by_tier_val_fraction_per_tier_close_to_requested() -> None:
    """Each tier's val token share should independently approximate val_split."""
    doc_end_offsets, doc_weights = _fake_corpus({
        0.5: [10_000] * 50,
        1.75: [10_000] * 50,
    })
    val_split = 0.2
    train_regions, val_regions = _split_by_tier(
        doc_end_offsets, doc_weights, val_split=val_split, seq_len=64
    )

    doc_starts = np.concatenate(([0], doc_end_offsets[:-1]))
    region_to_weight = {
        (int(s), int(e)): float(w)
        for s, e, w in zip(doc_starts, doc_end_offsets, doc_weights)
    }
    for tier in (0.5, 1.75):
        tier_total = sum(e - s for s, e in region_to_weight if region_to_weight[(s, e)] == tier)
        tier_val = sum(
            e - s for s, e in val_regions if region_to_weight[(s, e)] == tier
        )
        assert abs(tier_val / tier_total - val_split) < 0.05


def test_split_by_tier_deterministic() -> None:
    """Repeated calls with the same arguments must produce an identical split."""
    doc_end_offsets, doc_weights = _fake_corpus({0.5: [5000] * 20, 1.0: [5000] * 20})
    t1, v1 = _split_by_tier(doc_end_offsets, doc_weights, val_split=0.15, seq_len=64)
    t2, v2 = _split_by_tier(doc_end_offsets, doc_weights, val_split=0.15, seq_len=64)
    assert t1 == t2
    assert v1 == v2


def test_split_by_tier_single_document_tier_stays_in_train() -> None:
    """A tier reduced to one document can't be split -- keep it in train."""
    doc_end_offsets, doc_weights = _fake_corpus({
        0.5: [5000] * 20,
        1.75: [5000],  # single document
    })
    train_regions, val_regions = _split_by_tier(doc_end_offsets, doc_weights, val_split=0.2, seq_len=64)

    doc_starts = np.concatenate(([0], doc_end_offsets[:-1]))
    region_to_weight = {
        (int(s), int(e)): float(w)
        for s, e, w in zip(doc_starts, doc_end_offsets, doc_weights)
    }
    lone_doc = next(r for r, w in region_to_weight.items() if w == 1.75)
    assert lone_doc in train_regions
    assert lone_doc not in val_regions
