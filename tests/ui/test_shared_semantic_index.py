"""Tests for shared.py's semantic-index path/freshness helpers.

Covers the pure directory-naming logic (no live checkpoint needed) and an
end-to-end save/freshness round-trip using SemanticRetriever + RagIndex --
consistent with the rest of this test module, which unit-tests pure UI
helpers but leaves live-checkpoint UI callables (load_agent, load_engine)
untested (see test_app_mla_fields.py's module docstring for the same
scoping choice on the training side).
"""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")

from grimoire_ai.ui.shared import (  # noqa: E402
    _index_is_fresh,
    _semantic_index_dir,
    _semantic_index_dir_external,
)


# ---------------------------------------------------------------------------
# _semantic_index_dir_external
# ---------------------------------------------------------------------------

def test_external_index_dir_distinct_per_encoder(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    minilm_dir = _semantic_index_dir_external([str(corpus_dir)], "all-MiniLM-L6-v2")
    mpnet_dir = _semantic_index_dir_external([str(corpus_dir)], "all-mpnet-base-v2")
    assert minilm_dir != mpnet_dir


def test_external_index_dir_distinct_from_native(tmp_path):
    """The external-encoder cache must never collide with the native
    model-embeddings cache -- the vector spaces aren't even the same
    dimensionality."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    native_dir = _semantic_index_dir([str(corpus_dir)])
    external_dir = _semantic_index_dir_external([str(corpus_dir)], "all-MiniLM-L6-v2")
    assert native_dir != external_dir


def test_external_index_dir_stable_for_same_encoder(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    first = _semantic_index_dir_external([str(corpus_dir)], "all-MiniLM-L6-v2")
    second = _semantic_index_dir_external([str(corpus_dir)], "all-MiniLM-L6-v2")
    assert first == second


def test_external_index_dir_none_for_missing_corpus_dir(tmp_path):
    missing = tmp_path / "does_not_exist"
    assert _semantic_index_dir_external([str(missing)], "all-MiniLM-L6-v2") is None


def test_external_index_dir_none_for_empty_corpus_dirs():
    assert _semantic_index_dir_external([], "all-MiniLM-L6-v2") is None


# ---------------------------------------------------------------------------
# End-to-end: save_index / _index_is_fresh round-trip with checkpoint=""
# ---------------------------------------------------------------------------

def _keyword_embed(texts: list[str]):
    import torch

    rows = []
    for t in texts:
        low = t.lower()
        v = torch.tensor([1.0 if "grapple" in low else 0.0, 1.0 if "fire" in low else 0.0])
        if v.sum() == 0:
            v = torch.ones(2)
        rows.append(v)
    return torch.nn.functional.normalize(torch.stack(rows), p=2, dim=-1)


def test_external_index_freshness_round_trip(tmp_path):
    """checkpoint="" (as the external-encoder call sites use) must still
    correctly detect staleness from corpus content changes alone."""
    from grimoire_ai.llm.inference.rag_index import RagIndex
    from grimoire_ai.llm.inference.semantic import SemanticRetriever

    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    doc_path = corpus_dir / "doc.txt"
    doc_path.write_text("A grappled creature is held fast.", encoding="utf-8")

    index_dir = _semantic_index_dir_external([str(corpus_dir)], "all-MiniLM-L6-v2")
    assert index_dir is not None

    retriever = SemanticRetriever(embed_fn=_keyword_embed)
    retriever.add_text(doc_path.read_text(encoding="utf-8"), source="doc")
    retriever.index()
    hashes = RagIndex.compute_source_hashes([str(corpus_dir)], "", cache_dir=index_dir)
    retriever.save_index(index_dir, source_hashes=hashes)

    assert _index_is_fresh(index_dir, [str(corpus_dir)], "")

    doc_path.write_text("The fireball ignites everything nearby.", encoding="utf-8")
    assert not _index_is_fresh(index_dir, [str(corpus_dir)], "")
