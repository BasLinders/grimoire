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


def test_retriever_incremental_index() -> None:
    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text("A grappled creature is held fast.")
    retriever.index()
    retriever.add_text("The fire spreads quickly.")
    retriever.index()
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
