"""Tests for MinHash + LSH near-duplicate detection.

Gate criteria:
- Identical and near-identical documents are clustered together.
- Clearly distinct documents are not clustered.
- Deduplication keeps the longest representative of each cluster.
- Degenerate inputs (empty / single doc) are handled.
"""

from grimoire_ai.llm.data.dedup import (
    deduplicate_indices,
    find_duplicate_clusters,
)

# A few distinct base documents with enough words to shingle meaningfully.
_DOC_A = (
    "The wizard raised his staff and channelled arcane energy into the "
    "crackling sphere of light that hovered above the ancient stone altar."
)
_DOC_B = (
    "Statistical learning theory studies the problem of inferring a function "
    "from labelled training data by minimising empirical risk with regularisation."
)
_DOC_C = (
    "Dragons of the northern peaks hoard gold and gemstones in their volcanic "
    "lairs, guarding the treasure jealously against any wandering adventurer."
)


def test_identical_documents_cluster() -> None:
    texts = [_DOC_A, _DOC_A, _DOC_B]
    clusters = find_duplicate_clusters(texts, threshold=0.8)
    assert any(set(c) == {0, 1} for c in clusters), clusters


def test_near_identical_documents_cluster() -> None:
    # `near` appends a couple of words to A: it shares the vast majority of A's
    # word-5-grams, so the true Jaccard is ~0.9 — robustly above threshold.
    near = _DOC_A + " indeed it did."
    texts = [_DOC_A, near, _DOC_C]
    clusters = find_duplicate_clusters(texts, threshold=0.7)
    assert any({0, 1} <= set(c) for c in clusters), clusters


def test_distinct_documents_do_not_cluster() -> None:
    texts = [_DOC_A, _DOC_B, _DOC_C]
    clusters = find_duplicate_clusters(texts, threshold=0.8)
    assert clusters == [], f"Distinct docs should not cluster, got {clusters}."


def test_deduplicate_keeps_longest_representative() -> None:
    # Two extra words add 2 new word-5-grams → true Jaccard = 18/20 = 0.90.
    # P(LSH miss at bands=32, rows=4) ≈ 10⁻¹⁵; the original _DOC_A + _DOC_A
    # gave Jaccard 0.818 (P ≈ 7e-5), which is rare but observable over many runs.
    longer = _DOC_A + " Blazing still."
    texts = [_DOC_A, longer, _DOC_B]
    kept, clusters = deduplicate_indices(texts, threshold=0.6)
    assert 1 in kept, "The longer duplicate should be kept."
    assert 0 not in kept, "The shorter duplicate should be dropped."
    assert 2 in kept, "The distinct document should be kept."


def test_empty_and_single_inputs() -> None:
    assert find_duplicate_clusters([]) == []
    assert find_duplicate_clusters([_DOC_A]) == []
    kept, clusters = deduplicate_indices([_DOC_A])
    assert kept == [0] and clusters == []
