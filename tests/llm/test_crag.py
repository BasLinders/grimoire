"""Tests for CRAG-style per-passage corrective retrieval
(docs/architecture_optimization.md item #8, CRAG half only).

Gate criteria:
- "Correct" (score >= upper) and "Ambiguous" (lower <= score < upper)
  passages both survive; "Incorrect" (score < lower) passages are dropped.
- Correct passages are ordered ahead of Ambiguous ones (demotion), with
  each tier's internal relative order preserved.
- Degenerate inputs (empty list, all-Incorrect) are handled.
- lower_threshold > upper_threshold is rejected at construction time.
"""

import pytest

from grimoire_ai.corpus.corpus import QueryResult
from grimoire_ai.llm.inference.crag import CragFilter


def _make_result(score: float, excerpt: str = "text") -> QueryResult:
    return QueryResult(multi_token=(), next_token=None, score=score, source=None, excerpt=excerpt)


def test_filter_drops_incorrect_keeps_correct_and_ambiguous() -> None:
    results = [
        _make_result(0.9, "correct"),      # >= upper (0.7)
        _make_result(0.05, "incorrect"),   # < lower (0.3)
        _make_result(0.5, "ambiguous"),    # between
    ]
    filtered = CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter(results)

    assert [r.excerpt for r in filtered] == ["correct", "ambiguous"]


def test_filter_demotes_ambiguous_after_correct() -> None:
    """Ambiguous passages must be reordered after every Correct one, even
    when they appeared first in the input."""
    results = [
        _make_result(0.5, "ambiguous_first"),
        _make_result(0.9, "correct_second"),
    ]
    filtered = CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter(results)

    assert [r.excerpt for r in filtered] == ["correct_second", "ambiguous_first"]


def test_filter_preserves_relative_order_within_each_tier() -> None:
    results = [
        _make_result(0.8, "correct_a"),
        _make_result(0.4, "ambiguous_a"),
        _make_result(0.9, "correct_b"),
        _make_result(0.5, "ambiguous_b"),
    ]
    filtered = CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter(results)

    assert [r.excerpt for r in filtered] == ["correct_a", "correct_b", "ambiguous_a", "ambiguous_b"]


def test_filter_all_incorrect_returns_empty() -> None:
    results = [_make_result(0.1), _make_result(0.2)]
    assert CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter(results) == []


def test_filter_all_correct_returns_all_unreordered() -> None:
    results = [_make_result(0.9), _make_result(0.8)]
    filtered = CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter(results)
    assert filtered == results


def test_filter_empty_input_is_noop() -> None:
    assert CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter([]) == []


def test_filter_lower_boundary_score_is_kept_as_ambiguous() -> None:
    """A score exactly equal to lower_threshold must be kept (>=, not >)."""
    result = _make_result(0.3)
    assert CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter([result]) == [result]


def test_filter_upper_boundary_score_is_correct() -> None:
    """A score exactly equal to upper_threshold must classify as Correct
    (>=, not >), ordered ahead of Ambiguous passages."""
    correct = _make_result(0.7, "correct")
    ambiguous = _make_result(0.5, "ambiguous")
    filtered = CragFilter(lower_threshold=0.3, upper_threshold=0.7).filter([ambiguous, correct])
    assert [r.excerpt for r in filtered] == ["correct", "ambiguous"]


def test_equal_thresholds_collapse_ambiguous_zone() -> None:
    """lower_threshold == upper_threshold is legal -- no Ambiguous zone,
    equivalent to the old single-cutoff behaviour."""
    results = [_make_result(0.5, "above"), _make_result(0.4, "below")]
    filtered = CragFilter(lower_threshold=0.5, upper_threshold=0.5).filter(results)
    assert [r.excerpt for r in filtered] == ["above"]


def test_lower_greater_than_upper_raises() -> None:
    with pytest.raises(ValueError, match="lower_threshold"):
        CragFilter(lower_threshold=0.8, upper_threshold=0.3)
