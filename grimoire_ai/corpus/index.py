"""Hash-map index that stores the corpus multi-tokens.

Granville replaces vector databases with nested hashes for three reasons:
1. Multi-tokens are variable-length strings — hashes handle this natively.
2. Lookup is O(1) per key rather than O(n) for a linear scan.
3. No embedding step is needed: the stemmed word itself is the key.

Each entry in the index maps a multi-token (an n-gram tuple) to metadata:
the token that followed it in the corpus, how many times it appeared, and
which source document it came from. This metadata is used by the RBF
interpolator in Phase 2 to compute f(β_k) and weight the predictions.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class IndexEntry:
    """Metadata stored for a single multi-token in the corpus index.

    Attributes:
        next_token: The stemmed word that immediately followed this
            multi-token in the source text. Used by the RBF interpolator
            for next-token prediction. ``None`` when the multi-token
            ends at the last word of a document.
        frequency: Number of times this exact multi-token was observed
            across all ingested texts. Higher frequency signals a more
            common phrase, which weights its contribution in retrieval.
        source: An optional label identifying the source document
            (e.g. ``"dnd_srd"``, ``"wikipedia_ml"``). Useful for
            filtering results by domain.
        excerpt: A short window of the original (unstemmed) text
            surrounding this multi-token. Used by ``PromptBuilder`` as
            richer prompt context than the stemmed ``next_token`` alone.
            ``None`` when the index was populated without excerpt support.
    """

    next_token: Optional[str]
    frequency: int = 1
    source: Optional[str] = None
    excerpt: Optional[str] = None

    def increment(self) -> None:
        """Increase the frequency count by one.

        Called each time the same multi-token is encountered again during
        corpus ingestion, so the index reflects how common a phrase is.
        """
        self.frequency += 1


class CorpusIndex:
    """Hash map from multi-token tuples to their ``IndexEntry`` metadata.

    Internally backed by a plain Python ``dict``. Keys are tuples of stemmed
    strings (e.g. ``("grappl", "creatur", "speed", "reduc")``); values are
    ``IndexEntry`` instances. Duplicate insertions increment the frequency
    counter rather than overwriting the entry.

    Attributes:
        _store: The underlying dictionary mapping multi-tokens to entries.
    """

    def __init__(self) -> None:
        """Initialise an empty corpus index."""
        self._store: dict[tuple[str, ...], IndexEntry] = {}

    def add(
        self,
        multi_token: tuple[str, ...],
        next_token: Optional[str] = None,
        source: Optional[str] = None,
        excerpt: Optional[str] = None,
    ) -> None:
        """Insert a multi-token into the index or increment its frequency.

        If ``multi_token`` already exists in the index its frequency is
        incremented by one and the existing entry is otherwise unchanged.
        If it is new, a fresh ``IndexEntry`` is created with ``frequency=1``.

        Args:
            multi_token: An n-gram tuple of stemmed strings, e.g.
                ``("grappl", "creatur", "speed", "reduc")``.
            next_token: The stemmed word that followed this multi-token
                in the source text. Defaults to ``None``.
            source: Label of the originating document. Defaults to ``None``.
            excerpt: A short window of the original unstemmed text
                surrounding this multi-token. Defaults to ``None``.
        """
        if multi_token in self._store:
            entry = self._store[multi_token]
            entry.increment()
            # Update metadata to the most recent occurrence so query results
            # reflect a representative context rather than always the first one.
            entry.next_token = next_token
            entry.source = source
            entry.excerpt = excerpt
        else:
            self._store[multi_token] = IndexEntry(
                next_token=next_token, source=source, excerpt=excerpt
            )

    def get(self, multi_token: tuple[str, ...]) -> Optional[IndexEntry]:
        """Retrieve the entry for a given multi-token.

        Args:
            multi_token: An n-gram tuple of stemmed strings to look up.

        Returns:
            The ``IndexEntry`` for that multi-token, or ``None`` if it has
            not been indexed.
        """
        return self._store.get(multi_token)

    def all_entries(self) -> dict[tuple[str, ...], IndexEntry]:
        """Return the full index as a dictionary.

        Used by the retrieval layer to iterate over all stored multi-tokens
        when computing similarity scores against a query.

        Returns:
            A reference to the internal store — not a copy. Do not mutate.
        """
        return self._store

    def __len__(self) -> int:
        """Return the number of unique multi-tokens in the index.

        Returns:
            The count of distinct multi-token keys stored.
        """
        return len(self._store)
