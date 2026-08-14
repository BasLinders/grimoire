"""Hash-map index that stores the corpus multi-tokens.

A nested hash map serves as the lexical retrieval index for three reasons:
1. Multi-tokens are variable-length strings — hashes handle this natively.
2. Lookup is O(1) per key rather than O(n) for a linear scan.
3. No embedding step is needed: the stemmed word itself is the key.

Each entry in the index maps a multi-token (an n-gram tuple) to metadata:
the token that followed it in the corpus, how many times it appeared, and
which source document it came from. ``GrimoireCorpus.query`` uses this
metadata for Jaccard-weighted lexical scoring; semantic retrieval (see
``grimoire_ai.llm.inference.semantic.SemanticRetriever``) does not use
this index at all — it embeds passages directly with the trained model.

Point lookups by exact multi-token key (``get``) were always O(1), matching
the module docstring above. ``GrimoireCorpus.query`` doesn't do point
lookups, though — it scores every candidate against a bag of query words,
which the hash map alone doesn't help with. ``candidates_for_words`` adds a
second, inverted index (stemmed word -> multi-tokens containing it) so a
query only scores multi-tokens sharing at least one word with it, instead
of every multi-token ever indexed (docs/inference_optimization.md item #8).
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class IndexEntry:
    """Metadata stored for a single multi-token in the corpus index.

    Attributes:
        next_token: The stemmed word that immediately followed this
            multi-token in the source text. ``None`` when the multi-token
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
        _word_postings: Inverted index from a single stemmed word to every
            multi-token (already in ``_store``) containing that word —
            populated alongside ``_store`` in ``add``, never separately.
    """

    def __init__(self) -> None:
        """Initialise an empty corpus index."""
        self._store: dict[tuple[str, ...], IndexEntry] = {}
        self._word_postings: dict[str, set[tuple[str, ...]]] = defaultdict(set)

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
            # Keep the first-seen context. Overwriting on every duplicate
            # meant a high-frequency multi-token's excerpt/source ended up
            # being whatever document happened to be ingested *last* in a
            # large corpus -- effectively arbitrary, and decoupled from
            # which occurrence is actually relevant to a given query.
            self._store[multi_token].increment()
        else:
            self._store[multi_token] = IndexEntry(
                next_token=next_token, source=source, excerpt=excerpt
            )
            for word in multi_token:
                self._word_postings[word].add(multi_token)

    def get(self, multi_token: tuple[str, ...]) -> Optional[IndexEntry]:
        """Retrieve the entry for a given multi-token.

        Args:
            multi_token: An n-gram tuple of stemmed strings to look up.

        Returns:
            The ``IndexEntry`` for that multi-token, or ``None`` if it has
            not been indexed.
        """
        return self._store.get(multi_token)

    def candidates_for_words(self, words: "set[str]") -> "set[tuple[str, ...]]":
        """Return every indexed multi-token sharing at least one word with *words*.

        Inverted-index lookup: unions the postings list for each word in
        *words* instead of scanning every multi-token ever indexed. A
        multi-token with zero overlap with the query can never score above
        0 Jaccard similarity anyway (see ``GrimoireCorpus.query``), so this
        candidate set is exactly the set worth scoring at all — narrowing
        to it first is a pure speedup, not an approximation.

        Args:
            words: Stemmed query words to look up.

        Returns:
            The union of postings lists for every word in *words*. Empty
            when none of the words appear in any indexed multi-token.
        """
        candidates: set[tuple[str, ...]] = set()
        for word in words:
            candidates |= self._word_postings.get(word, set())
        return candidates

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
