"""Near-duplicate document detection via MinHash + LSH.

Large scraped corpora contain many near-duplicate documents — multiple
encyclopedia articles reciting the same event, boilerplate intros, mirrored
pages.  Training on them wastes tokens and over-weights the duplicated content
without adding signal.  Exact-match deduplication misses these because the
documents differ in a few words; what we want is *near*-duplicate removal by
content similarity.

Method
------
1. **Shingling.** Each document becomes a set of overlapping word k-grams
   (shingles).  Two documents that share most of their k-grams are near
   duplicates; the overlap is their Jaccard similarity.
2. **MinHash.** Computing exact Jaccard over every pair is O(n²) in documents
   and O(shingles) per pair.  MinHash replaces each shingle set with a short
   fixed-length signature whose per-position agreement is an unbiased estimator
   of the Jaccard similarity — cheap to store and compare.
3. **LSH banding.** Even comparing all signature pairs is O(n²).  Locality
   sensitive hashing splits each signature into ``bands`` bands of ``rows``
   rows and buckets documents by each band; only documents that collide in at
   least one band become candidate pairs.  Similar documents collide with high
   probability, dissimilar ones almost never.
4. **Confirm + cluster.** Candidate pairs whose full-signature agreement meets
   the threshold are linked with a union-find; each connected component is a
   near-duplicate cluster, from which we keep one representative (the longest
   document) and drop the rest.

Pure ``numpy`` — no external MinHash/LSH dependency.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_MAX_HASH = np.uint64((1 << 32) - 1)

_WORD_RE = re.compile(r"\w+")


def _shingles(text: str, k: int) -> set[int]:
    """Return the set of hashed word k-gram shingles for a document.

    Args:
        text: Raw document text.
        k: Number of words per shingle.

    Returns:
        A set of 32-bit shingle hashes.  Short documents (fewer than ``k``
        words) yield a single shingle of their whole word sequence so they are
        still comparable.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    # Use SHA-1 (deterministic, PYTHONHASHSEED-independent) so Jaccard estimates
    # are stable across Python versions, hash seeds, and repeated runs.
    def _h(gram: list[str]) -> int:
        return int.from_bytes(
            hashlib.sha1("\x00".join(gram).encode()).digest()[:4], "little"
        )
    if len(words) < k:
        return {_h(words)}
    return {_h(words[i: i + k]) for i in range(len(words) - k + 1)}


def _signature_matrix(
    shingle_sets: Sequence[set[int]],
    num_perm: int,
    seed: int,
) -> np.ndarray:
    """Compute the MinHash signature matrix for a list of shingle sets.

    Args:
        shingle_sets: One hashed-shingle set per document.
        num_perm: Number of hash permutations (signature length).
        seed: RNG seed for the hash coefficients.

    Returns:
        An ``(n_docs, num_perm)`` uint64 array.  A document with no shingles
        gets an all-``_MAX_HASH`` row so it never matches another document.
    """
    rng = np.random.default_rng(seed)
    # Multiplicative hashing mod 2^32: f(x) = (a*x + b) & 0xFFFFFFFF.
    # a must be ODD so that x → ax mod 2^32 is a bijection on Z_{2^32},
    # giving the min-wise independence that MinHash requires.
    # With a, b, x all < 2^32: a*x + b ≤ (2^32-1)^2 + (2^32-1) = 2^64 - 2^32,
    # which fits in uint64 without overflow — no Mersenne tricks needed.
    a = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 1 << 32, size=num_perm, dtype=np.uint64)

    n = len(shingle_sets)
    sig = np.full((n, num_perm), _MAX_HASH, dtype=np.uint64)
    for i, shingles in enumerate(shingle_sets):
        if not shingles:
            continue
        h = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
        # Shape (num_perm, n_shingles); AND keeps only the low 32 bits (≡ mod 2^32).
        hashed = (np.outer(a, h) + b[:, None]) & _MAX_HASH
        sig[i] = hashed.min(axis=1)
    return sig


class _UnionFind:
    """Minimal union-find for clustering near-duplicate documents."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def find_duplicate_clusters(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 32,
    k: int = 5,
    seed: int = 0,
) -> list[list[int]]:
    """Group documents into near-duplicate clusters.

    Args:
        texts: Documents to compare.
        threshold: Minimum estimated Jaccard similarity (signature agreement)
            for two documents to be treated as near duplicates.
        num_perm: MinHash signature length.  Must be divisible by ``bands``.
        bands: Number of LSH bands.  More bands → more candidate pairs (higher
            recall, lower precision before the threshold confirmation step).
        k: Shingle size in words.
        seed: RNG seed.

    Returns:
        A list of clusters, each a sorted list of document indices, containing
        only clusters with two or more members (singletons are omitted).

    Raises:
        ValueError: If ``num_perm`` is not divisible by ``bands``.
    """
    if num_perm % bands != 0:
        raise ValueError(f"num_perm ({num_perm}) must be divisible by bands ({bands}).")
    n = len(texts)
    if n < 2:
        return []

    rows = num_perm // bands
    shingle_sets = [_shingles(t, k) for t in texts]
    sig = _signature_matrix(shingle_sets, num_perm, seed)

    # --- LSH banding: bucket documents by each band's row-tuple -------------
    candidate_pairs: set[tuple[int, int]] = set()
    for band in range(bands):
        block = sig[:, band * rows: (band + 1) * rows]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            key = block[i].tobytes()
            buckets.setdefault(key, []).append(i)
        for members in buckets.values():
            if len(members) > 1:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        candidate_pairs.add((members[x], members[y]))

    # --- Confirm candidates by full-signature agreement, then cluster -------
    uf = _UnionFind(n)
    for i, j in candidate_pairs:
        agreement = float(np.mean(sig[i] == sig[j]))
        if agreement >= threshold:
            uf.union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(uf.find(i), []).append(i)

    return [sorted(members) for members in clusters.values() if len(members) > 1]


def deduplicate_indices(
    texts: Sequence[str],
    threshold: float = 0.8,
    num_perm: int = 128,
    bands: int = 32,
    k: int = 5,
    seed: int = 0,
) -> tuple[list[int], list[list[int]]]:
    """Return the indices to keep after near-duplicate removal.

    For each near-duplicate cluster the longest document (most characters) is
    kept and the rest are dropped.  Documents not in any cluster are always
    kept.

    Args:
        texts: Documents to deduplicate.
        threshold: Minimum signature agreement for a near-duplicate match.
        num_perm: MinHash signature length.
        bands: Number of LSH bands.
        k: Shingle size in words.
        seed: RNG seed.

    Returns:
        ``(kept_indices, clusters)`` where ``kept_indices`` is the sorted list
        of surviving document indices and ``clusters`` is the list of
        near-duplicate clusters that were collapsed.
    """
    clusters = find_duplicate_clusters(
        texts, threshold=threshold, num_perm=num_perm, bands=bands, k=k, seed=seed
    )
    drop: set[int] = set()
    for cluster in clusters:
        # Keep the longest document in the cluster; drop the others.
        keep = max(cluster, key=lambda idx: len(texts[idx]))
        drop.update(idx for idx in cluster if idx != keep)

    kept = [i for i in range(len(texts)) if i not in drop]
    return kept, clusters
