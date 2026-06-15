"""Tests for the persistent RAG index (RagIndex) and SemanticRetriever integration.

Covers:
- MD5 hash computation and staleness detection
- Round-trip save / load via memmap + JSON
- Brute-force query correctness
- FAISS index (conditional: skipped when faiss-cpu is not installed)
- SemanticRetriever.save_index / from_index round-trip
- FAISS query path through SemanticRetriever when available
"""

import json

import numpy as np
import pytest
import torch

from grimoire_ai.llm.inference.rag_index import RagIndex
from grimoire_ai.llm.inference.semantic import SemanticRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_index(n: int = 4, d: int = 8) -> RagIndex:
    """Return a small RagIndex with random L2-normalised vectors."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((n, d)).astype("float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= norms
    excerpts = [f"passage {i}" for i in range(n)]
    sources: list = [f"src_{i % 2}" for i in range(n)]
    return RagIndex(
        vectors=vecs,
        excerpts=excerpts,
        sources=sources,
        chunk_chars=400,
        source_hashes={"doc_a.txt": "aabbcc", "doc_b.txt": "ddeeff"},
    )


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

class TestStaleness:
    def test_missing_dir_is_stale(self, tmp_path):
        assert RagIndex.is_stale(tmp_path / "nonexistent", {}) is True

    def test_missing_meta_is_stale(self, tmp_path):
        (tmp_path / "vectors.dat").write_bytes(b"\x00" * 8)
        assert RagIndex.is_stale(tmp_path, {}) is True

    def test_missing_vec_is_stale(self, tmp_path):
        (tmp_path / "meta.json").write_text(
            json.dumps({"source_hashes": {}}), encoding="utf-8"
        )
        assert RagIndex.is_stale(tmp_path, {}) is True

    def test_matching_hashes_fresh(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        assert RagIndex.is_stale(tmp_path, idx._source_hashes) is False

    def test_changed_hash_stale(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        changed = dict(idx._source_hashes, **{"doc_a.txt": "000000"})
        assert RagIndex.is_stale(tmp_path, changed) is True

    def test_added_file_stale(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        added = dict(idx._source_hashes, **{"new_file.txt": "112233"})
        assert RagIndex.is_stale(tmp_path, added) is True

    def test_removed_file_stale(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        removed = {k: v for k, v in idx._source_hashes.items() if k != "doc_a.txt"}
        assert RagIndex.is_stale(tmp_path, removed) is True

    def test_corrupt_meta_stale(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        (tmp_path / "meta.json").write_text("not json", encoding="utf-8")
        assert RagIndex.is_stale(tmp_path, idx._source_hashes) is True

    def test_compute_source_hashes_real_files(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world", encoding="utf-8")
        hashes = RagIndex.compute_source_hashes([tmp_path])
        assert "doc.txt" in hashes
        assert len(hashes["doc.txt"]) == 32  # MD5 hex

    def test_compute_source_hashes_includes_checkpoint(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"\x00" * 64)
        hashes = RagIndex.compute_source_hashes([], ckpt)
        assert "__checkpoint__" in hashes

    def test_content_change_detected(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("version 1", encoding="utf-8")
        h1 = RagIndex.compute_source_hashes([tmp_path])
        f.write_text("version 2", encoding="utf-8")
        h2 = RagIndex.compute_source_hashes([tmp_path])
        assert h1 != h2


# ---------------------------------------------------------------------------
# Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        loaded = RagIndex.load(tmp_path)
        assert loaded.size == idx.size
        assert loaded.excerpts == idx.excerpts
        assert loaded.sources == idx.sources
        assert loaded.chunk_chars == idx.chunk_chars
        np.testing.assert_allclose(loaded.vectors, idx._vectors, atol=1e-6)

    def test_source_hashes_preserved(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        loaded = RagIndex.load(tmp_path)
        assert loaded._source_hashes == idx._source_hashes

    def test_meta_json_written(self, tmp_path):
        _make_index().save(tmp_path)
        meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
        assert meta["version"] == 1
        assert meta["n_passages"] == 4
        assert meta["d_model"] == 8

    def test_vectors_dat_written(self, tmp_path):
        _make_index().save(tmp_path)
        assert (tmp_path / "vectors.dat").is_file()

    def test_load_missing_meta_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RagIndex.load(tmp_path)

    def test_load_corrupt_meta_raises(self, tmp_path):
        idx = _make_index()
        idx.save(tmp_path)
        (tmp_path / "meta.json").write_text("corrupted", encoding="utf-8")
        with pytest.raises(ValueError):
            RagIndex.load(tmp_path)

    def test_save_creates_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        _make_index().save(nested)
        assert (nested / "meta.json").is_file()

    def test_vectors_written_atomically_no_tmp_dat_left(self, tmp_path):
        """save() must not leave a .tmp.dat file behind on success."""
        _make_index().save(tmp_path)
        assert not any(tmp_path.glob("*.tmp.dat"))

    def test_stale_after_simulated_partial_write(self, tmp_path):
        """A corrupt vectors.dat with a missing meta.json is detected as stale."""
        # Simulate crash: vectors.dat truncated, meta.json not yet written.
        (tmp_path / "vectors.dat").write_bytes(b"\x00" * 8)
        hashes = {"doc_a.txt": "aabbcc"}
        assert RagIndex.is_stale(tmp_path, hashes) is True  # meta.json absent → stale


# ---------------------------------------------------------------------------
# Brute-force query
# ---------------------------------------------------------------------------

class TestQuery:
    def test_returns_correct_count(self):
        idx = _make_index(n=6, d=8)
        results = idx.query(idx._vectors[0], top_k=3)
        assert len(results) == 3

    def test_top_result_is_self(self):
        idx = _make_index(n=4, d=8)
        results = idx.query(idx._vectors[2], top_k=1)
        score, passage_idx = results[0]
        assert passage_idx == 2
        assert score > 0.99  # cosine of identical L2-normalised vector ≈ 1

    def test_top_k_clamped_to_size(self):
        idx = _make_index(n=3, d=8)
        results = idx.query(idx._vectors[0], top_k=100)
        assert len(results) == 3

    def test_empty_index_returns_empty(self):
        idx = RagIndex(
            vectors=np.zeros((0, 8), dtype="float32"),
            excerpts=[],
            sources=[],
            chunk_chars=400,
            source_hashes={},
        )
        assert idx.query(np.zeros(8, dtype="float32"), top_k=5) == []

    def test_scores_descending(self):
        idx = _make_index(n=6, d=8)
        results = idx.query(idx._vectors[0], top_k=6)
        scores = [s for s, _ in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# FAISS (conditional)
# ---------------------------------------------------------------------------

try:
    import faiss as _faiss_mod  # noqa: F401
    _faiss_available = True
except ImportError:
    _faiss_available = False

_skip_no_faiss = pytest.mark.skipif(not _faiss_available, reason="faiss-cpu not installed")


@_skip_no_faiss
class TestFaiss:
    def test_build_faiss_returns_true(self):
        idx = _make_index()
        assert idx.build_faiss() is True
        assert idx._faiss_index is not None

    def test_faiss_query_matches_brute_force(self):
        idx = _make_index(n=10, d=16)
        idx_faiss = _make_index(n=10, d=16)
        idx_faiss.build_faiss()
        query = idx._vectors[3]
        bf = idx.query(query, top_k=5)
        fq = idx_faiss.query(query, top_k=5)
        assert [i for _, i in bf] == [i for _, i in fq]

    def test_faiss_index_saved_and_loaded(self, tmp_path):
        idx = _make_index()
        idx.build_faiss()
        idx.save(tmp_path)
        assert (tmp_path / "faiss.index").is_file()
        loaded = RagIndex.load(tmp_path)
        assert loaded._faiss_index is not None

    def test_faiss_query_after_load(self, tmp_path):
        idx = _make_index(n=8, d=16)
        idx.build_faiss()
        idx.save(tmp_path)
        loaded = RagIndex.load(tmp_path)
        results = loaded.query(idx._vectors[0], top_k=1)
        assert results[0][1] == 0  # self is top-1


# ---------------------------------------------------------------------------
# SemanticRetriever.save_index / from_index
# ---------------------------------------------------------------------------

def _keyword_embed(texts: list[str]) -> torch.Tensor:
    rows = []
    for t in texts:
        low = t.lower()
        v = torch.tensor([
            1.0 if "grapple" in low else 0.0,
            1.0 if "fire" in low else 0.0,
            1.0 if "stealth" in low else 0.0,
        ])
        if v.sum() == 0:
            v = torch.ones(3)
        rows.append(v)
    mat = torch.stack(rows)
    return torch.nn.functional.normalize(mat, p=2, dim=-1)


class TestRetrieverIndex:
    def test_save_index_raises_before_index_call(self, tmp_path):
        r = SemanticRetriever(embed_fn=_keyword_embed)
        r.add_text("some text")
        with pytest.raises(RuntimeError):
            r.save_index(tmp_path / "idx")

    def test_round_trip(self, tmp_path):
        r = SemanticRetriever(embed_fn=_keyword_embed)
        r.add_text("A grappled creature has its speed reduced to zero.", source="srd")
        r.add_text("The fireball ignites everything nearby.", source="srd")
        r.index()
        r.save_index(tmp_path / "idx", source_hashes={"a.txt": "deadbeef"})

        r2 = SemanticRetriever.from_index(tmp_path / "idx", embed_fn=_keyword_embed)
        assert r2.size == 2
        assert r2._excerpts == r._excerpts
        assert r2._sources == r._sources

    def test_from_index_retrieves_correctly(self, tmp_path):
        r = SemanticRetriever(embed_fn=_keyword_embed)
        r.add_text("A grappled creature has its speed reduced to zero.")
        r.add_text("The fireball ignites everything nearby.")
        r.add_text("A rogue uses stealth to hide.")
        r.index()
        r.save_index(tmp_path / "idx")

        r2 = SemanticRetriever.from_index(tmp_path / "idx", embed_fn=_keyword_embed)
        results = r2.query("grapple speed", top_k=1)
        assert "grappled creature" in results[0].excerpt

    def test_from_index_uses_faiss_when_available(self, tmp_path):
        if not _faiss_available:
            pytest.skip("faiss-cpu not installed")
        r = SemanticRetriever(embed_fn=_keyword_embed)
        r.add_text("A grappled creature has its speed reduced to zero.")
        r.index()
        r.save_index(tmp_path / "idx", build_faiss=True)
        r2 = SemanticRetriever.from_index(tmp_path / "idx", embed_fn=_keyword_embed)
        assert r2._rag_index is not None
        assert r2._rag_index._faiss_index is not None
