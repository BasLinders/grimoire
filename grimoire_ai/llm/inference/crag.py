"""CRAG-style per-passage corrective retrieval filter.

Layers on top of (does not replace) ``InferenceEngine.retrieval_threshold``,
which gates on the single best result's score: if the top result is below
threshold, everything is dropped. This module classifies and acts on each
retrieved passage independently, so an individually weak passage can be
dropped even when a stronger one in the same batch would keep the top-1
gate open, and a batch of uniformly-mediocre passages that alone wouldn't
trip the top-1 gate can still be filtered down to just the genuinely
relevant subset instead of injecting all of them.

Item #8 from docs/architecture_optimization.md (CRAG half only -- Self-RAG,
the other half of that item, needs dedicated fine-tune data with
reflection-token annotations and is out of scope here).

Two-threshold classification (matches the CRAG paper's own design, rather
than a single arbitrary cutoff): each passage is classified "Correct"
(score >= upper_threshold), "Ambiguous" (between the two thresholds), or
"Incorrect" (score < lower_threshold). Correct and Ambiguous passages both
survive; Incorrect ones are dropped. See ``CragFilter.filter`` for how
"Ambiguous" is handled given this project has no web-search fallback to
combine it with (unlike the original paper).

Score semantics are NOT uniform across backends -- the defaults
(lower=0.3, upper=0.7) are only meaningful for a similarity score already
on a roughly [0, 1] or [-1, 1] scale:
  - ``GrimoireCorpus`` (lexical): Jaccard similarity, [0, 1].
  - ``SemanticRetriever``: cosine similarity, [-1, 1].
  - A ``Reranker`` (see ``reranker.py``), if attached: raw cross-encoder
    logits, NOT normalised to any fixed range -- these defaults are not
    meaningful there, and there is currently no automatic rescaling.
    Pick thresholds appropriate to the specific cross-encoder model in use
    when combining CragFilter with a reranker, rather than trusting the
    defaults.

Usage
-----
    from grimoire_ai.llm.inference.crag import CragFilter

    engine.crag_filter = CragFilter(lower_threshold=0.3, upper_threshold=0.7)

No new scoring model: ``CragFilter`` reads whatever ``.score`` is already
on each ``QueryResult`` by the time it runs. In ``InferenceEngine._retrieve()``
this runs after an optional ``Reranker`` (see ``reranker.py``), so when a
reranker is attached, CragFilter reads its rescored score for free --
CragFilter itself has zero awareness of whether reranking happened (see
the score-semantics caveat above for why that matters here).
"""

from grimoire_ai.corpus.corpus import QueryResult


class CragFilter:
    """Classify and act on each retrieved passage's confidence, CRAG-style.

    Attributes:
        lower_threshold: Below this, a passage is "Incorrect" and dropped
            entirely.
        upper_threshold: At or above this, a passage is "Correct" and
            ordered ahead of any "Ambiguous" one.
    """

    def __init__(self, lower_threshold: float = 0.3, upper_threshold: float = 0.7) -> None:
        """Configure the classifier.

        Args:
            lower_threshold: Passages scoring below this are dropped.
                Defaults to 0.3, matching typical published thresholds for
                similarity scores on a roughly [0, 1]/[-1, 1] scale -- see
                the module docstring for when this default does not apply
                (a reranker's raw, unnormalised logits).
            upper_threshold: Passages scoring at or above this are treated
                as confidently correct. Defaults to 0.7, same caveat as
                ``lower_threshold``.

        Raises:
            ValueError: If ``lower_threshold`` is greater than
                ``upper_threshold``.
        """
        if lower_threshold > upper_threshold:
            raise ValueError(
                f"lower_threshold ({lower_threshold}) must not exceed "
                f"upper_threshold ({upper_threshold})."
            )
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def filter(self, results: list[QueryResult]) -> list[QueryResult]:
        """Classify every passage; return the survivors, Correct passages first.

        "Ambiguous" passages (between the two thresholds) are demoted
        rather than dropped -- kept, but reordered after every "Correct"
        one. This is this project's translation of the CRAG paper's own
        handling of the ambiguous case (there, an ambiguous passage is
        combined with a corrective web search; Grimoire has no web-search
        fallback). Demoting relies on ``PromptBuilder``'s existing
        behaviour instead: it joins context in list order and trims excess
        from the right, so a demoted passage is the first to be dropped
        if the token budget is tight, but still included when there's room
        — the same "included, but lower priority" outcome without needing
        new prompt-assembly machinery.

        Args:
            results: Candidates to classify, typically already
                reranked/scored.

        Returns:
            A new list: every "Correct" passage (score >= upper_threshold)
            in original relative order, followed by every "Ambiguous"
            passage (lower_threshold <= score < upper_threshold), also in
            original relative order. "Incorrect" passages (score below
            lower_threshold) are omitted entirely. Returns ``[]`` when
            every passage is "Incorrect" -- the caller
            (``InferenceEngine._retrieve()``) treats an empty list as the
            existing pure-chat fallback, same as ``corpus.query()``
            returning nothing today. Returns ``results`` unchanged (a
            no-op) when ``results`` is already empty.
        """
        if not results:
            return results
        correct = [r for r in results if r.score >= self.upper_threshold]
        ambiguous = [r for r in results if self.lower_threshold <= r.score < self.upper_threshold]
        return correct + ambiguous


def has_confident_result(results: list[QueryResult], crag_filter: CragFilter) -> bool:
    """Whether *results* contains at least one "Correct"-tier passage.

    A cheap, pre-generation proxy for "is the retrieved context strong
    enough to trust" -- in a RAG system, weak retrieval is the dominant
    cause of a wrong answer regardless of what specific words end up
    generated, so this is checked *before* generation rather than trying
    to judge the generated text after the fact (see
    ``InferenceEngine``'s ``corrective_retry`` option in ``engine.py``,
    which uses this to decide whether a widened re-query is worth
    attempting).

    Args:
        results: Candidates to check -- typically already classified by
            ``CragFilter.filter()``, though this only reads ``.score``
            directly so an unfiltered list works too.
        crag_filter: Supplies ``upper_threshold``, the same cutoff
            ``filter()`` itself uses for "Correct" -- reusing it here
            keeps this check consistent with whatever the caller
            configured instead of hardcoding a second threshold that
            could drift out of sync with it.

    Returns:
        ``True`` if any passage scores at or above
        ``crag_filter.upper_threshold``; ``False`` for an empty list.
    """
    return any(r.score >= crag_filter.upper_threshold for r in results)
