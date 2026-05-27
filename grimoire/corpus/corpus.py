from dataclasses import dataclass
from typing import Optional

from grimoire.corpus.index import CorpusIndex
from grimoire.corpus.stemmer import GrimoireStemmer
from grimoire.corpus.tokenizer import GrimoireTokenizer


@dataclass
class QueryResult:
    multi_token: tuple[str, ...]
    next_token: Optional[str]
    score: float
    source: Optional[str]


class GrimoireCorpus:
    def __init__(self, n: int = 4):
        self.n = n
        self._stemmer = GrimoireStemmer()
        self._tokenizer = GrimoireTokenizer(n=n)
        self._index = CorpusIndex()

    def add_text(self, text: str, source: Optional[str] = None) -> int:
        tokens = self._stemmer.tokenize_and_stem(text)
        multi_tokens = self._tokenizer.build(tokens)
        for i, mt in enumerate(multi_tokens):
            next_tok = tokens[i + self.n] if i + self.n < len(tokens) else None
            self._index.add(mt, next_token=next_tok, source=source)
        return len(multi_tokens)

    def query(self, text: str, top_k: int = 5) -> list[QueryResult]:
        query_tokens = self._stemmer.tokenize_and_stem(text)

        if not query_tokens:
            return []

        query_mts = self._tokenizer.build(query_tokens)
        # If query is shorter than n, fall back to individual stemmed tokens
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
        return len(self._index)
