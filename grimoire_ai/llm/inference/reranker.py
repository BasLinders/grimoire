"""Cross-encoder reranking of retrieved passages.

``InferenceEngine._retrieve()`` gets its candidates from a first-stage
retriever (``GrimoireCorpus`` Jaccard scoring or ``SemanticRetriever``
cosine similarity) — both fast, but neither actually reads the query and
the passage together the way a query-passage-pair model can. A
cross-encoder is much slower per pair (it scores one (query, passage) pair
at a time, no precomputed index), so it's only practical as a *second*
stage: rescore a modest first-stage candidate pool, not the whole corpus.

Item #7 from docs/architecture_optimization.md.

Usage
-----
    from grimoire_ai.llm.inference.reranker import (
        CROSS_ENCODER_MODELS, Reranker, make_cross_encoder_score_fn,
    )

    score_fn = make_cross_encoder_score_fn(CROSS_ENCODER_MODELS["TinyBERT (ms-marco-TinyBERT-L-2-v2)"])
    engine.reranker = Reranker(score_fn)
    engine.rerank_candidates = 20  # widen the first-stage pool before rescoring
"""

from typing import Callable

from grimoire_ai.corpus.corpus import QueryResult

# ---------------------------------------------------------------------------
# Cross-encoder factory
# ---------------------------------------------------------------------------

#: Maps the user-facing reranker name to its sentence-transformers
#: CrossEncoder model id. Both ship in the same `sentence-transformers`
#: package EXTERNAL_ENCODERS (semantic.py) already depends on via the
#: `[encoder]` optional-dependency group — no separate group needed.
CROSS_ENCODER_MODELS: dict[str, str] = {
    "TinyBERT (ms-marco-TinyBERT-L-2-v2)": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "MiniLM-12 (ms-marco-MiniLM-L-12-v2)": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}


def make_cross_encoder_score_fn(model_name: str) -> Callable[[str, list[str]], list[float]]:
    """Return a ``score_fn`` backed by a sentence-transformers ``CrossEncoder``.

    The sentence-transformers model is downloaded on first call and cached
    by the ``sentence_transformers`` library, same as ``make_external_embed_fn``
    in ``semantic.py``.

    Args:
        model_name: A sentence-transformers cross-encoder model identifier,
            e.g. ``"cross-encoder/ms-marco-TinyBERT-L-2-v2"``.

    Returns:
        A callable ``(query, passages) -> list[float]`` returning one
        relevance score per passage, same order as ``passages``. Higher is
        more relevant; scores are raw model logits, not normalised to any
        fixed range.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
            Install with ``pip install -e ".[encoder]"``.
    """
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for cross-encoder reranking. "
            "Install it with:  pip install -e \".[encoder]\""
        ) from exc

    _model = CrossEncoder(model_name)

    def _score(query: str, passages: list[str]) -> list[float]:
        pairs = [[query, p] for p in passages]
        return [float(s) for s in _model.predict(pairs)]

    return _score


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class Reranker:
    """Rescore and reorder ``QueryResult`` candidates with a cross-encoder.

    Deliberately does not truncate to a final count — ``InferenceEngine._retrieve()``
    already owns exactly one place where the final ``top_k`` slice happens
    (identical for the reranked and non-reranked paths), so this only
    reorders the full candidate list it's given.
    """

    def __init__(self, score_fn: Callable[[str, list[str]], list[float]]) -> None:
        """Configure the reranker.

        Args:
            score_fn: Callable ``(query, passages) -> list[float]``,
                typically from ``make_cross_encoder_score_fn``.
        """
        self._score_fn = score_fn

    def rerank(self, query: str, results: list[QueryResult]) -> list[QueryResult]:
        """Rescore ``results`` against ``query``, sorted descending by the new score.

        Text scored per result is ``r.excerpt or r.next_token or ""`` — the
        same fallback ``PromptBuilder`` uses when consuming these results,
        so the reranker sees exactly what would end up in the prompt.
        ``.score`` is overwritten with the cross-encoder's score on the
        returned objects (new ``QueryResult`` instances — the input list and
        its objects are never mutated, since a retriever backed by a
        persistent index may return objects backed by shared state).
        Nothing downstream reads the pre-rerank score once ``_retrieve()``
        returns it, so this is a safe, one-way overwrite.

        Args:
            query: The plain-text query the results were retrieved for.
            results: Candidates to rescore, typically a wider pool than the
                caller's final desired count.

        Returns:
            A new list of ``QueryResult`` objects, same length as
            ``results``, sorted descending by the cross-encoder score.
            Returns an empty list unchanged (no-op) when ``results`` is empty.
        """
        if not results:
            return results
        texts = [r.excerpt or r.next_token or "" for r in results]
        scores = self._score_fn(query, texts)
        rescored = [
            QueryResult(
                multi_token=r.multi_token,
                next_token=r.next_token,
                score=score,
                source=r.source,
                excerpt=r.excerpt,
            )
            for r, score in zip(results, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
