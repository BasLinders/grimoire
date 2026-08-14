"""Tests for semantic (embedding-cosine) corpus retrieval.

Covers the passage chunker, the SemanticRetriever ranking, and the
InferenceEngine.embed / build_semantic_corpus integration. A deterministic
fake embed_fn is used for the retriever unit tests so ranking assertions do
not depend on a trained model; a tiny real model is used for the end-to-end
engine test.
"""

import torch

from grimoire_ai.corpus.corpus import QueryResult
from grimoire_ai.llm.inference.semantic import SemanticRetriever, chunk_text


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunk_text_splits_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph here."
    chunks = chunk_text(text)
    assert chunks == ["First paragraph here.", "Second paragraph here."]


def test_chunk_text_skips_blank() -> None:
    assert chunk_text("\n\n   \n\n") == []


def test_chunk_text_breaks_long_paragraph() -> None:
    """A paragraph well over the limit is split into multiple chunks."""
    sentence = "A grappled creature has its speed reduced to zero. "
    long_para = sentence * 30  # ~1500 chars, single paragraph
    chunks = chunk_text(long_para, chunk_chars=200)
    assert len(chunks) > 1
    assert all(len(c) <= 600 for c in chunks)


# ---------------------------------------------------------------------------
# SemanticRetriever ranking with a deterministic fake embedder
# ---------------------------------------------------------------------------

def _keyword_embed(texts: list[str]) -> torch.Tensor:
    """Map each text to a 3-d unit vector based on keyword presence.

    Axis 0 = 'grapple', axis 1 = 'fire', axis 2 = 'stealth'. This lets the
    cosine ranking be asserted deterministically without a trained model.
    """
    rows = []
    for t in texts:
        low = t.lower()
        v = torch.tensor(
            [
                1.0 if "grapple" in low or "speed" in low else 0.0,
                1.0 if "fire" in low or "burn" in low else 0.0,
                1.0 if "stealth" in low or "hide" in low else 0.0,
            ]
        )
        if v.sum() == 0:
            v = torch.ones(3)
        rows.append(v)
    mat = torch.stack(rows)
    return torch.nn.functional.normalize(mat, p=2, dim=-1)


def test_retriever_ranks_semantically() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text(
        "A grappled creature has its speed reduced to zero.", source="srd"
    )
    retriever.add_text("The fireball ignites and burns everything nearby.", source="srd")
    retriever.add_text("A rogue can hide and move with stealth.", source="srd")
    retriever.index()

    results = retriever.query("how does grapple affect movement speed", top_k=1)
    assert len(results) == 1
    assert "grappled creature" in results[0].excerpt
    assert isinstance(results[0], QueryResult)
    assert results[0].source == "srd"


def test_retriever_empty_returns_empty() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    assert retriever.query("anything", top_k=5) == []
    assert retriever.size == 0


def test_query_batch_matches_per_call_query() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature has its speed reduced to zero.", source="srd")
    retriever.add_text("The fireball ignites and burns everything nearby.", source="srd")
    retriever.add_text("A rogue can hide and move with stealth.", source="srd")
    retriever.index()

    queries = ["grapple and speed", "fire and burning", "hide with stealth"]
    batched = retriever.query_batch(queries, top_k=2)
    assert len(batched) == 3
    for query_text, row in zip(queries, batched):
        assert row == retriever.query(query_text, top_k=2)


def test_query_batch_embeds_in_one_call() -> None:
    """The whole point of query_batch: one embed_fn call for the batch, not
    one per query text."""
    calls: list[int] = []

    def counting_embed(texts: list[str]) -> torch.Tensor:
        calls.append(len(texts))
        return _keyword_embed(texts)

    retriever = SemanticRetriever(embed_fn=counting_embed)
    retriever.add_text("A grappled creature is held fast.")
    retriever.add_text("The fire spreads quickly.")
    retriever.index()

    calls.clear()
    retriever.query_batch(["grapple", "fire", "stealth"], top_k=1)
    assert calls == [3]


def test_query_batch_empty_index_returns_empty_per_row() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    assert retriever.query_batch(["a", "b"], top_k=5) == [[], []]


def test_query_batch_empty_texts_returns_empty_list() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.")
    retriever.index()
    assert retriever.query_batch([], top_k=5) == []


def test_retriever_incremental_index() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.")
    retriever.index()
    retriever.add_text("The fire spreads quickly.")
    retriever.index()
    assert retriever.size == 2


def test_retriever_index_stops_early_and_keeps_progress() -> None:
    """A stop_event set before indexing leaves nothing embedded but the
    queued passages intact for a later index() call to resume."""
    import threading

    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.")
    retriever.add_text("The fire spreads quickly.")

    stop_event = threading.Event()
    stop_event.set()
    retriever.index(batch_size=1, stop_event=stop_event)

    assert retriever.size == 0
    assert len(retriever._pending) == 2

    # A later call without the stop_event resumes from where it left off.
    retriever.index(batch_size=1)
    assert retriever.size == 2


# ---------------------------------------------------------------------------
# End-to-end with a tiny real model
# ---------------------------------------------------------------------------

def test_engine_build_semantic_corpus(tmp_path) -> None:
    """build_semantic_corpus indexes passages and attaches a working retriever."""
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.model.config import TransformerConfig
    from grimoire_ai.llm.model.transformer import GrimoireTransformer
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder

    cfg = TransformerConfig(
        vocab_size=512, d_model=64, n_layers=2, n_heads=4,
        n_kv_heads=2, d_ff=128, max_seq_len=64, dropout=0.0,
    )
    model = GrimoireTransformer(cfg)

    # Train a tiny BPE tokenizer on toy text so encode() works.
    tok = BytePairEncoder()
    tok.train(
        ["a grappled creature has its speed reduced to zero",
         "the fireball burns and ignites everything nearby",
         "a rogue can hide and move with stealth quietly"],
        vocab_size=300,
    )
    tok_path = tmp_path / "bpe.json"
    tok.save(str(tok_path))

    ckpt_path = tmp_path / "step.pt"
    torch.save(
        {"step": 0, "config": cfg.to_dict(), "model": model.state_dict()},
        str(ckpt_path),
    )

    engine = InferenceEngine(
        checkpoint_path=str(ckpt_path), tokenizer_path=str(tok_path), device="cpu",
    )

    retriever = engine.build_semantic_corpus(
        [
            ("A grappled creature has its speed reduced to zero.", "srd"),
            ("The fireball burns everything nearby.", "srd"),
        ]
    )
    assert retriever.size == 2
    assert engine.corpus is retriever

    results = engine.corpus.query("grapple speed", top_k=2)
    assert len(results) == 2
    assert all(isinstance(r, QueryResult) for r in results)
    # Embeddings are L2-normalised, so cosine scores live in [-1, 1].
    assert all(-1.01 <= r.score <= 1.01 for r in results)


# ---------------------------------------------------------------------------
# Retrieval threshold router
# ---------------------------------------------------------------------------

def test_threshold_blocks_low_scores() -> None:
    """A threshold above the top score should return no results (pure-chat)."""
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature has its speed reduced to zero.")
    retriever.index()

    # Query about fire — cosine similarity to grapple passage will be low.
    results_unthresholded = retriever.query("fire damage and burning", top_k=1)
    assert len(results_unthresholded) == 1

    # Simulate engine threshold: score below 0.99 threshold → caller drops context.
    top_score = results_unthresholded[0].score
    # Threshold above top score means the engine would route to pure-chat.
    assert top_score < 0.99, f"Expected low cross-domain score, got {top_score}"


def test_threshold_passes_high_scores() -> None:
    """A matching query should clear a reasonable threshold."""
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature has its speed reduced to zero.")
    retriever.index()

    results = retriever.query("grapple speed movement", top_k=1)
    assert len(results) == 1
    assert results[0].score > 0.5, f"Expected high on-domain score, got {results[0].score}"


# ---------------------------------------------------------------------------
# indexed_sources / checkpointed resume
# ---------------------------------------------------------------------------

def test_indexed_sources_tracks_distinct_sources() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.", source="doc_a")
    retriever.add_text("The fire spreads quickly.", source="doc_b")
    retriever.index()
    assert retriever.indexed_sources == {"doc_a", "doc_b"}


def test_indexed_sources_excludes_pending() -> None:
    """A queued-but-not-yet-indexed document must not count as done."""
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.", source="doc_a")
    assert retriever.indexed_sources == set()
    retriever.index()
    assert retriever.indexed_sources == {"doc_a"}


def test_resume_from_checkpoint_skips_already_indexed_documents(tmp_path) -> None:
    """The save_index / from_index / indexed_sources combination must let a
    caller resume without re-embedding documents already indexed -- the
    pattern evaluate.py's --encoder lora checkpointing relies on.
    """
    calls: list[int] = []

    def counting_embed(texts: list[str]) -> torch.Tensor:
        calls.append(len(texts))
        return _keyword_embed(texts)

    retriever = SemanticRetriever(embed_fn=counting_embed)
    retriever.add_text("A grappled creature is held fast.", source="doc_a")
    retriever.add_text("The fire spreads quickly.", source="doc_b")
    retriever.index()
    assert retriever.size == 2

    checkpoint_dir = tmp_path / "checkpoint"
    retriever.save_index(checkpoint_dir, build_faiss=False)

    # Simulate a fresh process resuming: reload from the checkpoint, then
    # only re-add documents NOT already indexed.
    calls.clear()
    resumed = SemanticRetriever.from_index(checkpoint_dir, embed_fn=counting_embed)
    assert resumed.indexed_sources == {"doc_a", "doc_b"}

    all_documents = [
        ("A grappled creature is held fast.", "doc_a"),
        ("The fire spreads quickly.", "doc_b"),
        ("A rogue can hide and move with stealth.", "doc_c"),
    ]
    remaining = [(t, s) for t, s in all_documents if s not in resumed.indexed_sources]
    assert remaining == [("A rogue can hide and move with stealth.", "doc_c")]

    for text, source in remaining:
        resumed.add_text(text, source=source)
    resumed.index()

    assert resumed.size == 3
    assert resumed.indexed_sources == {"doc_a", "doc_b", "doc_c"}
    # Only the new document's passage should have been embedded post-resume.
    assert calls == [1]
