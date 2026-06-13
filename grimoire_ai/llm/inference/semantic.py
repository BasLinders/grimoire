"""Semantic retrieval over a corpus using the transformer's own embeddings.

This module replaces the lexical Jaccard scorer (``GrimoireCorpus.query``)
with *semantic* retrieval. Instead of matching on shared word stems, it
embeds every corpus passage and the query into the same vector space — the
space the trained ``GrimoireTransformer`` learned during pre-training — and
ranks passages by cosine similarity.

Why this is the hybrid step forward
------------------------------------
The original plan paired efficient retrieval with a conversational neural
layer. The neural layer (the transformer) is already trained on the D&D
corpus, so it has learned that "frightened" relates to "fear", that a
"grapple" concerns "speed" and "movement", and so on. Reusing those learned
representations for retrieval means the *same* model powers both halves of
the hybrid — no separate embedding server, no external vector database, and
no generic off-the-shelf encoder that has never seen a stat block.

Drop-in compatibility
----------------------
``SemanticRetriever.query`` returns the exact same ``QueryResult`` objects
that ``GrimoireCorpus.query`` returns, so it can be passed to
``InferenceEngine`` as the ``corpus`` argument with no other changes:
``PromptBuilder`` consumes the ``excerpt`` field identically either way.

Pipeline
--------
1. ``add_text`` chunks each document into passages and queues them.
2. ``index`` embeds every queued passage in batches and L2-normalises the
   vectors so cosine similarity reduces to a dot product.
3. ``query`` embeds the query, dots it against the passage matrix, and
   returns the ``top_k`` highest-scoring passages as ``QueryResult`` objects.
"""

import re
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from grimoire_ai.corpus.corpus import QueryResult

# ---------------------------------------------------------------------------
# External encoder factory
# ---------------------------------------------------------------------------

#: Maps the user-facing encoder name to its sentence-transformers model id.
EXTERNAL_ENCODERS: dict[str, str] = {
    "MiniLM (all-MiniLM-L6-v2)":   "all-MiniLM-L6-v2",
    "MPNet (all-mpnet-base-v2)":    "all-mpnet-base-v2",
}


def make_external_embed_fn(model_name: str) -> Callable[[list[str]], torch.Tensor]:
    """Return an ``embed_fn`` backed by a sentence-transformers model.

    The returned callable has the same signature as ``InferenceEngine.embed``:
    it accepts a list of strings and returns an L2-normalised
    ``(n, d_model)`` float tensor on CPU.  It can be passed directly to
    ``SemanticRetriever`` as ``embed_fn``.

    The sentence-transformers model is downloaded on first call (~90 MB for
    MiniLM) and cached by the ``sentence_transformers`` library in
    ``~/.cache/torch/sentence_transformers``.

    Args:
        model_name: A sentence-transformers model identifier, e.g.
            ``"all-MiniLM-L6-v2"``.

    Returns:
        A callable ``(list[str]) -> Tensor`` that embeds texts as L2-normalised
        CPU float tensors of shape ``(n, hidden_size)``.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
            Install with ``pip install -e ".[encoder]"``.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for external encoders. "
            "Install it with:  pip install -e \".[encoder]\""
        ) from exc

    _model = SentenceTransformer(model_name)

    def _embed(texts: list[str]) -> torch.Tensor:
        vecs = _model.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vecs.float().cpu()

    return _embed

# Target size of a passage chunk, in characters. Short enough that several
# passages fit comfortably inside the prompt context budget, long enough to
# carry a complete rule or description. Roughly one short paragraph.
_CHUNK_CHARS = 400

# Hard ceiling on a single chunk so one runaway paragraph (e.g. a long table
# flattened to text) cannot dominate the prompt budget.
_MAX_CHUNK_CHARS = 600

# Sentence boundary: end punctuation followed by whitespace. Deliberately
# simple — corpus text is already cleaned, and over-engineering sentence
# splitting buys nothing for retrieval quality.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_chars: int = _CHUNK_CHARS) -> list[str]:
    """Split a document into passage-sized chunks for embedding.

    Splits first on blank lines (paragraph boundaries), then packs sentences
    into chunks of up to ``chunk_chars`` characters. A paragraph shorter than
    the target is emitted whole; a longer one is broken on sentence
    boundaries so no chunk greatly exceeds ``_MAX_CHUNK_CHARS``.

    Args:
        text: The full document text.
        chunk_chars: Target chunk size in characters.

    Returns:
        A list of non-empty, stripped passage strings. Returns an empty list
        for blank input.
    """
    chunks: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= _MAX_CHUNK_CHARS:
            chunks.append(para)
            continue

        # Paragraph too long — pack its sentences into chunks.
        current = ""
        for sentence in _SENTENCE_SPLIT.split(para):
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and len(current) + 1 + len(sentence) > chunk_chars:
                chunks.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence
        if current:
            chunks.append(current)

    return chunks


class SemanticRetriever:
    """Cosine-similarity retrieval over corpus passages using model embeddings.

    Build one by passing an ``embed_fn`` that maps a list of strings to an
    L2-normalised ``(n, d_model)`` tensor — ``InferenceEngine.embed`` provides
    exactly this. Ingest documents with ``add_text``, call ``index`` once to
    embed them, then ``query`` repeatedly.

    Attributes:
        size: Number of indexed passages (0 until ``index`` is called).
    """

    def __init__(
        self,
        embed_fn: Callable[[list[str]], torch.Tensor],
        chunk_chars: int = _CHUNK_CHARS,
    ) -> None:
        """Initialise an empty retriever.

        Args:
            embed_fn: Callable that embeds a list of texts into an
                L2-normalised ``(n, d_model)`` float tensor on CPU. Normally
                ``InferenceEngine.embed``.
            chunk_chars: Target passage size passed to ``chunk_text``.
        """
        self._embed_fn = embed_fn
        self._chunk_chars = chunk_chars
        self._pending: list[tuple[str, Optional[str]]] = []
        self._excerpts: list[str] = []
        self._sources: list[Optional[str]] = []
        self._vectors: Optional[torch.Tensor] = None

    def add_text(self, text: str, source: Optional[str] = None) -> int:
        """Chunk a document and queue its passages for indexing.

        Passages are not embedded until ``index`` is called, so ingesting
        many documents and indexing once is efficient.

        Args:
            text: Raw document text.
            source: Optional provenance label stored with every passage from
                this document and surfaced in ``QueryResult.source``.

        Returns:
            The number of passages queued from this document.
        """
        chunks = chunk_text(text, self._chunk_chars)
        for chunk in chunks:
            self._pending.append((chunk, source))
        return len(chunks)

    def index(self, batch_size: int = 32) -> int:
        """Embed all queued passages and build the searchable vector matrix.

        Safe to call repeatedly: passages queued since the last call are
        embedded and appended to the existing index. Cleared queue after.

        Args:
            batch_size: Number of passages embedded per forward pass.

        Returns:
            The total number of indexed passages after this call.
        """
        if self._pending:
            texts = [c for c, _ in self._pending]
            sources = [s for _, s in self._pending]

            new_vectors: list[torch.Tensor] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                new_vectors.append(self._embed_fn(batch))
            stacked = torch.cat(new_vectors, dim=0)

            if self._vectors is None:
                self._vectors = stacked
            else:
                self._vectors = torch.cat([self._vectors, stacked], dim=0)
            self._excerpts.extend(texts)
            self._sources.extend(sources)
            self._pending.clear()

        return self.size

    def query(self, text: str, top_k: int = 5) -> list[QueryResult]:
        """Retrieve the ``top_k`` passages most semantically similar to a query.

        Embeds the query with the same ``embed_fn``, computes cosine
        similarity against every indexed passage (a single matrix-vector
        product, since all vectors are L2-normalised), and returns the
        highest-scoring passages.

        Args:
            text: Plain-language query.
            top_k: Maximum number of passages to return.

        Returns:
            A list of ``QueryResult`` objects sorted by descending cosine
            score. ``multi_token`` is an empty tuple and ``next_token`` is
            ``None`` — those fields are artefacts of the lexical index and
            carry no meaning for semantic retrieval; ``excerpt`` holds the
            passage and is what ``PromptBuilder`` consumes. Returns an empty
            list when nothing has been indexed.
        """
        if self._vectors is None or self.size == 0:
            return []

        query_vec = self._embed_fn([text])  # (1, d_model), normalised
        scores = (self._vectors @ query_vec.squeeze(0))  # (n,)

        k = min(top_k, self.size)
        top_scores, top_idx = torch.topk(scores, k)

        results: list[QueryResult] = []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            results.append(
                QueryResult(
                    multi_token=(),
                    next_token=None,
                    score=float(score),
                    source=self._sources[idx],
                    excerpt=self._excerpts[idx],
                )
            )
        return results

    def save_cache(self, path) -> None:
        """Save indexed vectors, excerpts, and sources to a .pt cache file.

        Uses a write-to-temp-then-rename pattern so a concurrent or interrupted
        write never leaves a corrupt file at the target path.
        """
        import os
        import tempfile
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        try:
            os.close(fd)
            torch.save({
                "vectors": self._vectors,
                "excerpts": self._excerpts,
                "sources": self._sources,
                "chunk_chars": self._chunk_chars,
            }, tmp)
            os.replace(tmp, str(p))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @classmethod
    def from_cache(cls, path, embed_fn: Callable[[list[str]], torch.Tensor]) -> "SemanticRetriever":
        """Restore a retriever from a file written by ``save_cache``."""
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        obj = cls(embed_fn=embed_fn, chunk_chars=data["chunk_chars"])
        obj._vectors = data["vectors"]
        obj._excerpts = data["excerpts"]
        obj._sources = data["sources"]
        return obj

    @property
    def size(self) -> int:
        """Number of passages currently indexed (excludes the pending queue)."""
        return 0 if self._vectors is None else self._vectors.shape[0]
