"""Persistent semantic index: numpy memmap vectors + JSON metadata + optional FAISS.

Replaces the ephemeral in-memory index in ``SemanticRetriever`` with an
on-disk representation that survives across sessions.  Staleness is detected
by comparing MD5 hashes of every source ``.txt`` file and the model checkpoint
stored in ``meta.json`` against the current files — a hash change (content
update) triggers a rebuild; a timestamp change alone does not.

On-disk layout inside the index directory
-----------------------------------------
vectors.dat   float32 memmap, shape (N, d_model)
meta.json     JSON: excerpts, sources, chunk_chars, d_model, n_passages,
              source_hashes, version
faiss.index   optional FAISS IndexFlatIP (only written when faiss-cpu is
              installed and ``build_faiss()`` is called)

Typical usage
-------------
    # Build once after training or corpus update:
    hashes = RagIndex.compute_source_hashes(["data/corpus/saga"], ckpt_path)
    if RagIndex.is_stale(".semantic_index", hashes):
        retriever.save_index(".semantic_index", source_hashes=hashes)

    # Load on every session start:
    retriever = SemanticRetriever.from_index(".semantic_index", embed_fn=engine.embed)
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

_META_FILE = "meta.json"
_VEC_FILE = "vectors.dat"
_FAISS_FILE = "faiss.index"
_HASH_CACHE_FILE = ".hash_cache.json"
_VERSION = 1


def _md5(path: "str | Path") -> str:
    """Return the MD5 hex-digest of a file without loading it fully into memory."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _stat_stamp(path: "Path") -> str:
    """Return a cheap (mtime, size) fingerprint for a file, no content read."""
    st = path.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


class RagIndex:
    """Persistent semantic index: memmap vectors + JSON metadata + optional FAISS.

    Attributes:
        size: Number of indexed passages.
        excerpts: Passage strings in index order.
        sources: Optional source labels in index order.
        chunk_chars: Target chunk size used when the index was built.
        vectors: Numpy float32 array of shape ``(N, d_model)``, L2-normalised.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        excerpts: "list[str]",
        sources: "list[Optional[str]]",
        chunk_chars: int,
        source_hashes: "dict[str, str]",
    ) -> None:
        self._vectors = vectors.astype("float32", copy=False)
        self._excerpts = excerpts
        self._sources = sources
        self._chunk_chars = chunk_chars
        self._source_hashes = source_hashes
        self._faiss_index = None  # set by build_faiss()

    @property
    def size(self) -> int:
        return len(self._excerpts)

    @property
    def excerpts(self) -> "list[str]":
        return self._excerpts

    @property
    def sources(self) -> "list[Optional[str]]":
        return self._sources

    @property
    def chunk_chars(self) -> int:
        return self._chunk_chars

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    # ------------------------------------------------------------------ #
    # Staleness detection                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_source_hashes(
        corpus_dirs: "list[str | Path]",
        checkpoint_path: "Optional[str | Path]" = None,
        lora_path: "Optional[str | Path]" = None,
        cache_dir: "Optional[str | Path]" = None,
    ) -> "dict[str, str]":
        """Compute MD5 hashes for every ``.txt`` file, the checkpoint, and a LoRA adapter.

        Hashing thousands of corpus files (plus a multi-hundred-MB checkpoint)
        on every freshness check is expensive even when nothing changed. When
        *cache_dir* is given, a small ``(mtime, size) -> md5`` cache is kept
        alongside the index so a file is only re-read when its mtime or size
        has actually changed; an unchanged file reuses its last-known digest.

        Args:
            corpus_dirs: Directories whose ``.txt`` files form the corpus.
            checkpoint_path: Model checkpoint to include in the hash so that
                switching checkpoints automatically invalidates the index.
            lora_path: LoRA adapter file to include in the hash, stored under
                ``"__lora__"``, so loading/changing/removing an adapter also
                invalidates an index built from a different set of weights.
            cache_dir: Directory to read/write the stat-based hash cache from.
                Typically the same directory the index itself lives in.

        Returns:
            A ``{filename: md5_hex}`` dict.  The checkpoint and LoRA adapter
            are stored under the fixed keys ``"__checkpoint__"`` / ``"__lora__"``.
        """
        cache_path = Path(cache_dir) / _HASH_CACHE_FILE if cache_dir else None
        stat_cache: dict = {}
        if cache_path and cache_path.is_file():
            try:
                stat_cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                stat_cache = {}

        new_cache: dict = {}

        def _hash_with_cache(path: Path, key: str) -> str:
            stamp = _stat_stamp(path)
            cached = stat_cache.get(key)
            digest = cached["md5"] if cached and cached.get("stamp") == stamp else _md5(path)
            new_cache[key] = {"stamp": stamp, "md5": digest}
            return digest

        # The stat cache is keyed by resolved absolute path, not by basename:
        # the returned `hashes` dict groups files by `f.name` (so corpus_dirs
        # containing same-named files collapse to one entry, as before this
        # cache existed), but two *different* files that happen to share a
        # name across different corpus_dirs must never share a cache entry —
        # otherwise a coincidental (mtime, size) match on one file could
        # return a cached digest that actually belongs to the other file.
        hashes: "dict[str, str]" = {}
        for d in corpus_dirs:
            for f in sorted(Path(d).glob("*.txt")):
                hashes[f.name] = _hash_with_cache(f, str(f.resolve()))
        if checkpoint_path and Path(checkpoint_path).is_file():
            ckpt = Path(checkpoint_path)
            hashes["__checkpoint__"] = _hash_with_cache(ckpt, str(ckpt.resolve()))
        if lora_path and Path(lora_path).is_file():
            lora = Path(lora_path)
            hashes["__lora__"] = _hash_with_cache(lora, str(lora.resolve()))

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(new_cache), encoding="utf-8")
            except OSError:
                pass

        return hashes

    @staticmethod
    def is_stale(
        index_dir: "str | Path",
        current_hashes: "dict[str, str]",
    ) -> bool:
        """Return ``True`` when the on-disk index is missing or out of date.

        Reads ``source_hashes`` from ``meta.json`` and compares against
        ``current_hashes``.  Any difference — added file, removed file, or
        changed content — counts as stale.

        Args:
            index_dir: Directory produced by :meth:`save`.
            current_hashes: Hashes returned by :meth:`compute_source_hashes`.

        Returns:
            ``True`` when a rebuild is required.
        """
        meta_path = Path(index_dir) / _META_FILE
        vec_path = Path(index_dir) / _VEC_FILE
        if not meta_path.is_file() or not vec_path.is_file():
            return True
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return True
        return meta.get("source_hashes", {}) != current_hashes

    # ------------------------------------------------------------------ #
    # FAISS                                                                #
    # ------------------------------------------------------------------ #

    def build_faiss(self) -> bool:
        """Build a FAISS ``IndexFlatIP`` over the stored vectors.

        An inner-product index is correct here because every vector is already
        L2-normalised, so inner product equals cosine similarity.

        Returns:
            ``True`` when faiss-cpu is installed and the index was built;
            ``False`` when the package is absent (brute-force fallback used).
        """
        try:
            import faiss
        except ImportError:
            return False
        n, d = self._vectors.shape
        idx = faiss.IndexFlatIP(d)
        idx.add(np.ascontiguousarray(self._vectors))
        self._faiss_index = idx
        return True

    # ------------------------------------------------------------------ #
    # Query                                                                #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int,
    ) -> "list[tuple[float, int]]":
        """Return ``(score, passage_idx)`` pairs ordered by descending cosine score.

        Uses the FAISS index when available, brute-force numpy dot product
        otherwise.  ``query_vec`` must be an L2-normalised float32 array of
        shape ``(d_model,)`` or ``(1, d_model)``.

        Args:
            query_vec: Normalised query embedding.
            top_k: Maximum number of results to return.

        Returns:
            List of ``(score, idx)`` tuples, best first.
        """
        k = min(top_k, self.size)
        if k == 0:
            return []
        qv = query_vec.reshape(1, -1).astype("float32")
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(qv, k)
            return [
                (float(s), int(i))
                for s, i in zip(scores[0], indices[0])
                if i >= 0
            ]
        flat = (self._vectors @ qv.squeeze())  # (N,)
        top_idx = np.argsort(-flat)[:k]
        return [(float(flat[i]), int(i)) for i in top_idx]

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, dir_path: "str | Path") -> None:
        """Write the index to *dir_path* (created if absent).

        Vectors are written as a flat float32 numpy memmap (``vectors.dat``).
        Metadata — excerpts, sources, chunk_chars, d_model, source_hashes — is
        written atomically via a rename to ``meta.json``.  If a FAISS index has
        been built (via :meth:`build_faiss`) it is written to ``faiss.index``.

        Args:
            dir_path: Target directory.  Created recursively if missing.
        """
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)
        n, d = self._vectors.shape
        # Write vectors atomically: write to a temp file, then rename into place
        # so a crash mid-write never leaves a corrupt vectors.dat alongside a
        # valid meta.json that would pass the staleness check.
        fd, tmp_vec = tempfile.mkstemp(dir=p, suffix=".tmp.dat")
        try:
            os.close(fd)
            mm = np.memmap(tmp_vec, dtype="float32", mode="w+", shape=(n, d))
            mm[:] = self._vectors
            mm.flush()
            del mm
            os.replace(tmp_vec, str(p / _VEC_FILE))
        except Exception:
            try:
                os.unlink(tmp_vec)
            except OSError:
                pass
            raise
        # Write metadata atomically.
        meta: dict = {
            "version": _VERSION,
            "d_model": d,
            "n_passages": n,
            "chunk_chars": self._chunk_chars,
            "source_hashes": self._source_hashes,
            "excerpts": self._excerpts,
            "sources": self._sources,
        }
        meta_path = p / _META_FILE
        fd, tmp = tempfile.mkstemp(dir=p, suffix=".tmp")
        try:
            os.close(fd)
            Path(tmp).write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )
            os.replace(tmp, str(meta_path))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        # Write FAISS index when available.
        if self._faiss_index is not None:
            try:
                import faiss
                faiss.write_index(self._faiss_index, str(p / _FAISS_FILE))
            except ImportError:
                pass

    @classmethod
    def load(cls, dir_path: "str | Path") -> "RagIndex":
        """Load a previously saved index from *dir_path*.

        Vectors are read from the memmap file into a contiguous in-memory
        array so that subsequent numpy and FAISS operations work on normal RAM
        without keeping the file descriptor open.

        Args:
            dir_path: Directory produced by :meth:`save`.

        Returns:
            A fully populated :class:`RagIndex`.

        Raises:
            FileNotFoundError: When ``meta.json`` or ``vectors.dat`` is absent.
            ValueError: When ``meta.json`` is malformed.
        """
        p = Path(dir_path)
        meta_path = p / _META_FILE
        vec_path = p / _VEC_FILE
        if not meta_path.is_file():
            raise FileNotFoundError(f"No {_META_FILE} found in {p}")
        if not vec_path.is_file():
            raise FileNotFoundError(f"No {_VEC_FILE} found in {p}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed {_META_FILE} in {p}: {exc}") from exc
        n: int = meta["n_passages"]
        d: int = meta["d_model"]
        mm = np.memmap(vec_path, dtype="float32", mode="r", shape=(n, d))
        vectors = np.array(mm)  # copy into normal RAM
        del mm
        obj = cls(
            vectors=vectors,
            excerpts=meta["excerpts"],
            sources=meta["sources"],
            chunk_chars=meta["chunk_chars"],
            source_hashes=meta.get("source_hashes", {}),
        )
        faiss_path = p / _FAISS_FILE
        if faiss_path.is_file():
            try:
                import faiss
                obj._faiss_index = faiss.read_index(str(faiss_path))
            except ImportError:
                pass
        return obj
