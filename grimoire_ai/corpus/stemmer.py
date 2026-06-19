"""Stemming and tokenization utilities for the Grimoire corpus pipeline.

Stemming reduces inflected words to their base form (their "stem") so that
"grapple", "grappling", and "grappled" all index under the same key.
This module uses the Porter stemmer algorithm — a well-established rule-based
approach that is fast, deterministic, and requires no downloaded data.

Acronyms (e.g. D&D, HP, AC) are detected and preserved as lowercase strings
rather than stemmed, because stemming would destroy their meaning.
"""

import re

from nltk.stem import PorterStemmer


class GrimoireStemmer:
    """Tokenizes raw text and reduces each token to its stemmed form.

    Uses the NLTK Porter stemmer for regular words and preserves acronyms
    (short uppercase sequences such as D&D, HP, AC, NPC) by lowercasing
    them without stemming.

    Attributes:
        _ACRONYM_RE: Compiled regex that matches uppercase acronym tokens
            up to 6 characters, optionally containing digits, & or -.
        _TOKEN_RE: Compiled regex that extracts word-like tokens from raw
            text, including hyphenated and apostrophe-containing words.
        _stemmer: The underlying NLTK PorterStemmer instance.
    """

    # Uppercase sequences up to 6 chars are treated as acronyms and
    # lowercased without stemming to preserve their meaning (e.g. D&D → d&d).
    _ACRONYM_RE = re.compile(r'^[A-Z][A-Z0-9&\-]{0,5}$')
    _TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9&\-']*")

    def __init__(self) -> None:
        """Initialises the GrimoireStemmer with a Porter stemmer instance."""
        self._stemmer = PorterStemmer()
        # NLTK's PorterStemmer is pure Python and re-derives the same suffix
        # rules on every call. Natural-language text is highly Zipfian — a
        # small vocabulary accounts for most token occurrences — so caching
        # per-instance cuts repeated work dramatically on large corpora.
        self._cache: dict[str, str] = {}

    def stem(self, word: str) -> str:
        """Reduce a single word to its stem, preserving acronyms.

        If the word matches the acronym pattern (all uppercase, ≤ 6 chars),
        it is returned as-is in lowercase. Otherwise the Porter stemming
        algorithm is applied after lowercasing.

        Args:
            word: A single token extracted from raw text.

        Returns:
            The stemmed (or lowercased acronym) form of the word.

        Examples:
            >>> s = GrimoireStemmer()
            >>> s.stem("grappling")
            'grappl'
            >>> s.stem("D&D")
            'd&d'
            >>> s.stem("AC")
            'ac'
        """
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        result = word.lower() if self._ACRONYM_RE.match(word) else self._stemmer.stem(word.lower())
        self._cache[word] = result
        return result

    def tokenize_and_stem(self, text: str) -> list[str]:
        """Extract tokens from raw text and stem each one.

        Tokens shorter than 2 characters are discarded because single-letter
        words (articles, prepositions) add noise without contributing meaning
        to the index.

        Args:
            text: Raw input text — a sentence, paragraph, or full document.

        Returns:
            An ordered list of stemmed tokens. Preserves the original word
            order so that downstream n-gram builders can create positional
            multi-tokens.

        Examples:
            >>> s = GrimoireStemmer()
            >>> s.tokenize_and_stem("A grappled creature loses its speed.")
            ['grappl', 'creatur', 'lose', 'it', 'speed']
        """
        tokens = self._TOKEN_RE.findall(text)
        return [self.stem(t) for t in tokens if len(t) > 1]
