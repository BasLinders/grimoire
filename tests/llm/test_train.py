"""Tests for grimoire_ai.llm.training.train's dataset-building helpers.

Gate criteria:
- _split_blocks produces disjoint train/val regions covering the whole corpus.
- The val fraction (by token count) is close to the requested val_split.
- The split is deterministic across repeated calls (required so a training
  run and a separate scripts/build_source_weights.py invocation agree).
- Val blocks are scattered rather than concentrated at one end of the corpus.
"""

from grimoire_ai.llm.training.train import _split_blocks


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
