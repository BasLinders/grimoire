"""Near-duplicate detection for corpus text files via MinHash/shingling.

Estimates Jaccard similarity between documents using word-shingle MinHash
signatures, without needing an external LSH library. Scoped to compare a
"new" set of files (e.g. a fresh scrape) against the rest of the corpus and
against each other, rather than doing a full O(n^2) sweep over everything —
that's the check called for before merging newly scraped material into an
existing corpus.

Usage
-----
    python scripts/dedup_corpus.py --corpus-dir data/corpus/saga/ --new-glob "gutenberg_*.txt"
    python scripts/dedup_corpus.py --corpus-dir data/corpus/saga/ --new-glob "gutenberg_*.txt" --threshold 0.3
"""

import argparse
import re
from pathlib import Path

import numpy as np

_WORD_RE = re.compile(r"[a-z0-9']+")
_PRIME = (1 << 61) - 1


def _shingle_hashes(text: str, k: int) -> np.ndarray:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        shingles = [" ".join(words)] if words else []
    else:
        shingles = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
    if not shingles:
        return np.array([0], dtype=np.uint64)
    hashes = np.array(
        [hash(s) & 0xFFFFFFFFFFFFFFFF for s in shingles], dtype=np.uint64
    )
    return hashes


def _minhash_signature(hashes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = hashes.astype(np.uint64) % _PRIME
    sig = np.empty(len(a), dtype=np.uint64)
    for i in range(len(a)):
        permuted = (a[i] * h + b[i]) % _PRIME
        sig[i] = permuted.min()
    return sig


def build_signatures(paths: list[Path], k: int, num_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, _PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)

    sigs = {}
    sizes = {}
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="ignore")
        hashes = _shingle_hashes(text, k)
        sigs[p] = _minhash_signature(hashes, a, b)
        sizes[p] = len(text)
    return sigs, sizes


def estimate_jaccard(sig1: np.ndarray, sig2: np.ndarray) -> float:
    return float((sig1 == sig2).mean())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MinHash near-duplicate detection for corpus text files"
    )
    parser.add_argument("--corpus-dir", default="data/corpus/saga/")
    parser.add_argument(
        "--new-glob", required=True,
        help="Glob (relative to --corpus-dir) matching the newly added files to check",
    )
    parser.add_argument("--shingle-size", type=int, default=8, help="Words per shingle")
    parser.add_argument("--num-perm", type=int, default=64, help="MinHash permutations")
    parser.add_argument("--threshold", type=float, default=0.4,
                         help="Report pairs with estimated Jaccard >= this")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    all_files = sorted(corpus_dir.rglob("*.txt"))
    new_files = sorted(corpus_dir.glob(args.new_glob))
    new_set = set(new_files)

    if not new_files:
        print(f"No files matched --new-glob {args.new_glob!r} under {corpus_dir}")
        return

    print(f"Comparing {len(new_files)} new files against {len(all_files)} corpus files "
          f"(shingle_size={args.shingle_size}, num_perm={args.num_perm})")

    sigs, sizes = build_signatures(all_files, args.shingle_size, args.num_perm, args.seed)

    pairs = []
    seen = set()
    for i, f1 in enumerate(new_files):
        for f2 in all_files:
            if f1 == f2:
                continue
            key = tuple(sorted((str(f1), str(f2))))
            if key in seen:
                continue
            seen.add(key)
            sim = estimate_jaccard(sigs[f1], sigs[f2])
            if sim >= args.threshold:
                pairs.append((sim, f1.name, f2.name, sizes[f1], sizes[f2]))

    pairs.sort(reverse=True)
    if not pairs:
        print(f"\nNo near-duplicate pairs found at threshold {args.threshold}.")
        return

    print(f"\n{len(pairs)} near-duplicate pair(s) found (threshold={args.threshold}):")
    for sim, n1, n2, s1, s2 in pairs:
        print(f"  {sim:.3f}  {n1} ({s1} chars)  <->  {n2} ({s2} chars)")


if __name__ == "__main__":
    main()
