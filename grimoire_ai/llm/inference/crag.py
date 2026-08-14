"""CRAG-style per-passage corrective retrieval filter.

Layers on top of (does not replace) ``InferenceEngine.retrieval_threshold``,
which gates on the single best result's score: if the top result is below
threshold, everything is dropped. This module gates each retrieved passage
independently, so an individually weak passage can be dropped even when a
stronger one in the same batch would keep the top-1 gate open, and a batch
of uniformly-mediocre passages that alone wouldn't trip the top-1 gate can
still be filtered down to just the genuinely relevant subset instead of
injecting all of them.

Item #8 from docs/architecture_optimization.md (CRAG half only -- Self-RAG,
the other half of that item, needs dedicated fine-tune data with
reflection-token annotations and is out of scope here).

Usage
-----
    from grimoire_ai.llm.inference.crag import CragFilter

    engine.crag_filter = CragFilter(passage_threshold=0.1)

No new scoring model: ``CragFilter`` reads whatever ``.score`` is already
on each ``QueryResult`` by the time it runs. In ``InferenceEngine._retrieve()``
this runs after an optional ``Reranker`` (see ``reranker.py``), so when a
reranker is attached, CragFilter reads its rescored (materially more
reliable) score for free -- CragFilter itself has zero awareness of
whether reranking happened.
"""

from grimoire_ai.corpus.corpus import QueryResult


class CragFilter:
    """Drop individually low-confidence passages from a retrieved set.

    Attributes:
        passage_threshold: Minimum ``QueryResult.score`` to keep a passage.
            Distinct from ``InferenceEngine.retrieval_threshold`` (that gate
            looks only at the top-1 score and is binary all-or-nothing;
            this one is applied independently to every passage).
    """

    def __init__(self, passage_threshold: float) -> None:
        """Configure the filter.

        Args:
            passage_threshold: Minimum score a passage must reach to survive.
        """
        self.passage_threshold = passage_threshold

    def filter(self, results: list[QueryResult]) -> list[QueryResult]:
        """Return only the passages scoring >= ``passage_threshold``, order preserved.

        Args:
            results: Candidates to filter, typically already reranked/scored.

        Returns:
            A new list containing only the surviving passages, in their
            original relative order. Returns ``[]`` when every passage
            scores below the threshold -- the caller
            (``InferenceEngine._retrieve()``) treats an empty list as the
            existing pure-chat fallback, same as ``corpus.query()``
            returning nothing today. Returns ``results`` unchanged (a
            no-op) when ``results`` is already empty.
        """
        if not results:
            return results
        return [r for r in results if r.score >= self.passage_threshold]
