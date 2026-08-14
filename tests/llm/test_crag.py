"""Tests for CRAG-style per-passage corrective retrieval
(docs/architecture_optimization.md item #8, CRAG half only)."""

from grimoire_ai.corpus.corpus import QueryResult
from grimoire_ai.llm.inference.crag import CragFilter


def _make_result(score: float, excerpt: str = "text") -> QueryResult:
    return QueryResult(multi_token=(), next_token=None, score=score, source=None, excerpt=excerpt)


def test_filter_drops_below_threshold() -> None:
    results = [
        _make_result(0.9, "high"),
        _make_result(0.05, "low"),
        _make_result(0.5, "mid"),
    ]
    filtered = CragFilter(passage_threshold=0.3).filter(results)

    assert [r.excerpt for r in filtered] == ["high", "mid"]


def test_filter_keeps_order_of_survivors() -> None:
    results = [_make_result(0.5, "a"), _make_result(0.9, "b"), _make_result(0.6, "c")]
    filtered = CragFilter(passage_threshold=0.4).filter(results)

    assert [r.excerpt for r in filtered] == ["a", "b", "c"]


def test_filter_all_below_threshold_returns_empty() -> None:
    results = [_make_result(0.1), _make_result(0.2)]
    assert CragFilter(passage_threshold=0.5).filter(results) == []


def test_filter_all_above_threshold_returns_all() -> None:
    results = [_make_result(0.9), _make_result(0.8)]
    filtered = CragFilter(passage_threshold=0.5).filter(results)
    assert filtered == results


def test_filter_empty_input_is_noop() -> None:
    assert CragFilter(passage_threshold=0.5).filter([]) == []


def test_filter_boundary_score_is_kept() -> None:
    """A score exactly equal to the threshold must be kept (>=, not >)."""
    result = _make_result(0.5)
    assert CragFilter(passage_threshold=0.5).filter([result]) == [result]
