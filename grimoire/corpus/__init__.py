"""The Grimoire corpus package.

Provides the tools to ingest raw text, index it as stemmed multi-tokens,
and retrieve relevant entries by query. This is the foundation layer that
all other Grimoire components build on.

Public API:
    GrimoireCorpus: The main class — add text, query results.

Internal modules (not imported directly in normal usage):
    stemmer:   GrimoireStemmer — tokenises and stems raw text.
    tokenizer: GrimoireTokenizer — builds n-gram tuples from stemmed tokens.
    index:     CorpusIndex / IndexEntry — the hash-map store.
"""

from grimoire.corpus.corpus import GrimoireCorpus

__all__ = ["GrimoireCorpus"]
