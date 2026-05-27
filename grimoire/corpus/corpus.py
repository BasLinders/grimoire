"""Public interface for building and querying a Grimoire corpus.

``GrimoireCorpus`` is the entry point for Phase 1. It wires together the
stemmer, tokenizer, and index into a single object that accepts raw text and
returns ranked retrieval results.

Typical usage:

    corpus = GrimoireCorpus()
    corpus.add_text("A grappled creature has its speed reduced to zero.", source="dnd_srd")
    results = corpus.query("grapple speed movement", top_k=3)
    for r in results:
        print(r.multi_token, r.score, r.next_token)

Phase 2 note:
    The ``query`` method currently scores results with Jaccard similarity
    weighted by frequency. This is a structurally correct placeholder.
    In Phase 2 it will be replaced by Granville's RBF kernel formula:

        f_pred(x) = Σ ω_k(x) · f(β_k) · exp[−τ · K(x, β_k)]

    The ``QueryResult`` dataclass and the ``query`` method signature will
    not change — only the internal scoring logic.
"""

from dataclasses import dataclass
from typing import Optional

from grimoire.corpus.index import CorpusIndex
from grimoire.corpus.stemmer import GrimoireStemmer
from grimoire.corpus.tokenizer import GrimoireTokenizer


@dataclass
class QueryResult:
    """A single ranked result returned by ``GrimoireCorpus.query``.

    Attributes:
        multi_token: The matching n-gram tuple from the corpus index,
            e.g. ``("grappl", "creatur", "speed", "reduc")``.
        next_token: The stemmed word that followed this multi-token in
            the source text. Forms the basis of next-token prediction
            once the RBF kernel is in place. May be ``None`` if the
            multi-token ends a document.
        score: Relevance score relative to the query. Higher is more
            relevant. Currently Jaccard-based; will become the RBF
            interpolation value ``f_pred(x)`` in Phase 2.
        source: Label of the document this multi-token was ingested from,
            or ``None`` if no source was provided at ingestion time.
    """

    multi_token: tuple[str, ...]
    next_token: Optional[str]
    score: float
    source: Optional[str]


class GrimoireCorpus:
    """Corpus manager: ingests text, indexes multi-tokens, and retrieves results.

    Internally this class owns a ``GrimoireStemmer``, a ``GrimoireTokenizer``,
    and a ``CorpusIndex``. Calling ``add_text`` runs the full ingestion
    pipeline (stem → tokenize → index). Calling ``query`` stems the query,
    builds query multi-tokens, and scores every entry in the index.

    Attributes:
        n: The n-gram window size used by the tokenizer. Shared with the
            index so that next-token look-ahead offsets are computed
            consistently.
    """

    def __init__(self, n: int = 4) -> None:
        """Initialise an empty corpus with a given n-gram window size.

        Args:
            n: Number of stemmed tokens per multi-token. Defaults to 4,
                matching Granville's case study. Must be a positive integer.
        """
        self.n = n
        self._stemmer = GrimoireStemmer()
        self._tokenizer = GrimoireTokenizer(n=n)
        self._index = CorpusIndex()

    def add_text(self, text: str, source: Optional[str] = None) -> int:
        """Ingest raw text into the corpus index.

        Runs the full pipeline:
        1. Tokenise and stem the text with ``GrimoireStemmer``.
        2. Build overlapping n-gram tuples with ``GrimoireTokenizer``.
        3. Store each multi-token and its following token in ``CorpusIndex``.

        Args:
            text: Raw input text. May be a sentence, paragraph, or full
                document. Punctuation and casing are handled automatically.
            source: Optional label for the originating document, e.g.
                ``"dnd_srd"`` or ``"wikipedia_ml"``. Stored with each entry
                and returned in query results for provenance tracking.

        Returns:
            The number of multi-tokens added to the index. Returns 0 when
            the text is too short to form even one n-gram.
        """
        tokens = self._stemmer.tokenize_and_stem(text)
        multi_tokens = self._tokenizer.build(tokens)
        for i, mt in enumerate(multi_tokens):
            next_tok = tokens[i + self.n] if i + self.n < len(tokens) else None
            self._index.add(mt, next_token=next_tok, source=source)
        return len(multi_tokens)

    def query(self, text: str, top_k: int = 5) -> list[QueryResult]:
        """Retrieve the top-k corpus entries most relevant to a query string.

        Pipeline:
        1. Stem the query text.
        2. Derive a query token set (multi-token members, or individual
           stems if the query is shorter than ``n``).
        3. Score every index entry by Jaccard similarity × frequency.
        4. Return the ``top_k`` highest-scoring entries as ``QueryResult``
           objects, sorted descending by score.

        Scoring note:
            Jaccard similarity measures the overlap between two token sets
            as ``|A ∩ B| / |A ∪ B|``. It is multiplied by the entry's
            frequency so that common corpus phrases rank higher than rare
            ones. This formula is a placeholder for the RBF kernel in Phase 2.

        Args:
            text: The query string in plain language, e.g.
                ``"grapple speed movement"``. Does not need to be in the
                corpus verbatim — stemming normalises it first.
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            A list of ``QueryResult`` objects sorted by descending score.
            Returns an empty list when no indexed multi-tokens share any
            stemmed tokens with the query.
        """
        query_tokens = self._stemmer.tokenize_and_stem(text)

        if not query_tokens:
            return []

        query_mts = self._tokenizer.build(query_tokens)
        # If the query is shorter than n, fall back to individual stemmed tokens
        query_set = (
            {token for mt in query_mts for token in mt}
            if query_mts
            else set(query_tokens)
        )

        scores: dict[tuple[str, ...], float] = {}
        for mt, entry in self._index.all_entries().items():
            overlap = len(set(mt) & query_set)
            if overlap > 0:
                # Jaccard similarity weighted by frequency — replaced by RBF kernel in Phase 2
                union = len(set(mt) | query_set)
                scores[mt] = (overlap / union) * entry.frequency

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            QueryResult(
                multi_token=mt,
                next_token=self._index.get(mt).next_token,
                score=score,
                source=self._index.get(mt).source,
            )
            for mt, score in ranked
        ]

    @property
    def size(self) -> int:
        """The number of unique multi-tokens currently stored in the index.

        Returns:
            An integer count of distinct index entries.
        """
        return len(self._index)
