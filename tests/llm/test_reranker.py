"""Tests for cross-encoder reranking (docs/architecture_optimization.md item #7).

A deterministic fake score_fn is used for Reranker's own unit tests, same
approach as test_semantic.py's _keyword_embed -- ranking assertions must not
depend on downloading a real cross-encoder model. make_cross_encoder_score_fn's
ImportError path is tested by forcing the import to fail via sys.modules,
since sentence-transformers is actually installed in this dev environment
(there's no "skip if unavailable" marker to hang this test off of the way
tests/llm/test_rag_index.py does for faiss).
"""

import sys
from unittest.mock import patch

import pytest

from grimoire_ai.corpus.corpus import QueryResult
from grimoire_ai.llm.inference.reranker import Reranker, make_cross_encoder_score_fn


def _length_score_fn(query: str, passages: list[str]) -> list[float]:
    """Deterministic fake: score = negative distance from query's length.

    Lets tests assert a specific, predictable reordering without a real
    cross-encoder model.
    """
    target = len(query)
    return [-abs(len(p) - target) for p in passages]


def _make_result(excerpt: str, score: float = 0.0, source: str = "src") -> QueryResult:
    return QueryResult(multi_token=(), next_token=None, score=score, source=source, excerpt=excerpt)


# ---------------------------------------------------------------------------
# Reranker.rerank
# ---------------------------------------------------------------------------

def test_rerank_reorders_by_score() -> None:
    query = "12345"  # length 5
    results = [
        _make_result("a"),           # len 1, distance 4
        _make_result("abcde"),       # len 5, distance 0 -- best match
        _make_result("abc"),         # len 3, distance 2
    ]
    reranker = Reranker(_length_score_fn)
    reranked = reranker.rerank(query, results)

    assert [r.excerpt for r in reranked] == ["abcde", "abc", "a"]
    assert reranked[0].score == 0
    assert reranked[1].score == -2
    assert reranked[2].score == -4


def test_rerank_empty_results_is_noop() -> None:
    reranker = Reranker(_length_score_fn)
    assert reranker.rerank("anything", []) == []


def test_rerank_preserves_source_and_multi_token() -> None:
    """Only .score should change -- everything else about a QueryResult
    (source, excerpt, multi_token, next_token) must survive rerank untouched."""
    original = QueryResult(
        multi_token=("grappl", "creatur"), next_token="speed", score=0.5,
        source="dnd_srd", excerpt="A grappled creature.",
    )
    reranker = Reranker(_length_score_fn)
    [reranked] = reranker.rerank("query", [original])

    assert reranked.multi_token == original.multi_token
    assert reranked.next_token == original.next_token
    assert reranked.source == original.source
    assert reranked.excerpt == original.excerpt


def test_rerank_does_not_mutate_input_list_or_objects() -> None:
    original = _make_result("abcde", score=0.9)
    results = [original]
    reranker = Reranker(_length_score_fn)
    reranker.rerank("12345", results)

    assert results[0] is original
    assert original.score == 0.9  # unchanged -- rerank() returned new objects


def test_rerank_falls_back_to_next_token_when_no_excerpt() -> None:
    """Text scored per result must be `excerpt or next_token or ""`, same
    fallback PromptBuilder uses -- so the reranker sees what PromptBuilder
    will actually inject."""
    captured_texts: list[str] = []

    def _capturing_score_fn(query: str, passages: list[str]) -> list[float]:
        captured_texts.extend(passages)
        return [0.0] * len(passages)

    result = QueryResult(multi_token=(), next_token="fallback_word", score=0.0, source=None, excerpt=None)
    Reranker(_capturing_score_fn).rerank("query", [result])

    assert captured_texts == ["fallback_word"]


# ---------------------------------------------------------------------------
# make_cross_encoder_score_fn
# ---------------------------------------------------------------------------

def test_make_cross_encoder_score_fn_raises_importerror_when_missing() -> None:
    with patch.dict(sys.modules, {"sentence_transformers": None}):
        with pytest.raises(ImportError, match=r'pip install -e ".\[encoder\]"'):
            make_cross_encoder_score_fn("cross-encoder/ms-marco-TinyBERT-L-2-v2")
