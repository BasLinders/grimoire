"""Build a per-window retrieval_neighbors.npy for RETRO-style Chunked Cross-Attention.

Companion to ``TransformerConfig.retro_layers`` / ``GrimoireTransformer.forward``'s
``neighbor_ids`` parameter (see ``chunked_cross_attention.py`` and item #3 in
docs/architecture_optimization.md). Precomputes, for each ``TokenizedDataset``
training window, the token ids of its top-``n_neighbors`` nearest-neighbor
passages from a ``SemanticRetriever`` index built over the same corpus — so
``Trainer`` never needs a live embedding+search call during training (which
would mean an extra forward pass through the model on every training step;
the retrieval database is frozen during training anyway, so there is nothing
to gain from recomputing neighbors on the fly).

Same idiom as ``scripts/build_source_weights.py``: produces a ``.npy`` array
aligned to ``TokenizedDataset.offsets``, consumed via a ``"neighbor_ids_path"``
key in the training config (``Trainer(neighbor_ids=np.load(...))``).

Self-retrieval exclusion
-------------------------
A training window (``seq_len`` tokens, e.g. 1024) is much longer than an
indexed retrieval passage (``chunk_chars`` characters, ~400 by default,
roughly 100 tokens) — if the same corpus was both indexed for retrieval and
used for pretraining windows, the window's own text almost certainly
*contains* several of the index's own passages verbatim (they were chunked
from it). Querying with the full window text would then retrieve the
window's own constituent passages as "neighbors", teaching the model to
predict text it can already see rather than anything useful. Any candidate
passage that is a substring of the query window's text is dropped before
taking the top-``n_neighbors``.

Usage
-----
Using an existing persistent index (see ``RagIndex`` / the Chat tab's
semantic index):

    python scripts/build_retrieval_neighbors.py \\
        --corpus      data/processed/corpus.bin \\
        --checkpoint  checkpoints/pretrain/step_0010000.pt \\
        --vocab       data/tokenizer/bpe.json \\
        --seq-len     1024 --stride 512 \\
        --index-dir   data/processed/.semantic_index \\
        --n-neighbors 2 --neighbor-len 64 \\
        --output      data/processed/retrieval_neighbors.npy

Building a fresh index from raw text files instead of an existing one:

    python scripts/build_retrieval_neighbors.py \\
        --corpus      data/processed/corpus.bin \\
        --checkpoint  checkpoints/pretrain/step_0010000.pt \\
        --vocab       data/tokenizer/bpe.json \\
        --seq-len     1024 --stride 512 \\
        --corpus-dir  data/corpus/saga/ \\
        --n-neighbors 2 --neighbor-len 64 \\
        --output      data/processed/retrieval_neighbors.npy

``--seq-len``/``--stride`` must match what training will actually use (the
same requirement ``build_source_weights.py`` documents for
``sample_weights.npy``) — the array is only valid for windows built with
that exact configuration.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.llm.inference.semantic import SemanticRetriever
from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID


def _tokenize_neighbor(tokenizer, text: str, neighbor_len: int) -> list[int]:
    """Encode *text* to exactly *neighbor_len* token ids, truncated or padded."""
    ids = tokenizer.encode(text)[:neighbor_len]
    if len(ids) < neighbor_len:
        ids = ids + [PAD_ID] * (neighbor_len - len(ids))
    return ids


def build(
    corpus_path: str,
    checkpoint_path: str,
    vocab_path: str,
    seq_len: int,
    stride: int,
    output_path: str,
    n_neighbors: int = 2,
    neighbor_len: int = 64,
    index_dir: str | None = None,
    corpus_dir: str | None = None,
    embed_batch_size: int = 32,
) -> None:
    """Build and save the per-window neighbor-ids array.

    Args:
        corpus_path: Tokenized corpus (.bin) — must be the same file, with
            the same seq_len/stride, that training will use.
        checkpoint_path: Model checkpoint used both to embed windows/query
            the retrieval index and to tokenize (via ``vocab_path``).
        vocab_path: BPE vocabulary JSON.
        seq_len: Window length — must match training.
        stride: Window stride — must match training.
        output_path: Destination ``.npy`` file.
        n_neighbors: Neighbors retrieved per window.
        neighbor_len: Token length each neighbor chunk is truncated/padded
            to — must match ``neighbor_len`` used wherever this array is
            consumed (``Trainer``/``GrimoireTransformer.forward``).
        index_dir: Directory of a persistent ``RagIndex`` (see
            ``SemanticRetriever.save_index``) to load and query. Mutually
            exclusive with ``corpus_dir``.
        corpus_dir: Directory of raw ``.txt`` files to build a fresh index
            from instead of loading an existing one. Mutually exclusive
            with ``index_dir``.
        embed_batch_size: Windows embedded per forward pass.

    Raises:
        ValueError: If neither or both of ``index_dir``/``corpus_dir`` are
            given, or if ``n_neighbors``/``neighbor_len`` are not positive.
    """
    if bool(index_dir) == bool(corpus_dir):
        raise ValueError("Exactly one of --index-dir or --corpus-dir must be given.")
    if n_neighbors <= 0:
        raise ValueError(f"n_neighbors ({n_neighbors}) must be positive.")
    if neighbor_len <= 0:
        raise ValueError(f"neighbor_len ({neighbor_len}) must be positive.")

    print(f"Loading checkpoint: {checkpoint_path}")
    engine = InferenceEngine(checkpoint_path=checkpoint_path, tokenizer_path=vocab_path)
    print(f"Model ready on {engine.device.upper()}.")

    if index_dir:
        print(f"Loading retrieval index from {index_dir} ...")
        retriever = SemanticRetriever.from_index(index_dir, embed_fn=engine.embed)
    else:
        print(f"Building retrieval index from {corpus_dir} ...")
        retriever = SemanticRetriever(embed_fn=engine.embed)
        for txt_file in sorted(Path(corpus_dir).glob("*.txt")):
            retriever.add_text(txt_file.read_text(encoding="utf-8"), source=txt_file.stem)
        retriever.index(batch_size=embed_batch_size)
    print(f"Index ready: {retriever.size:,} passage(s).")

    dataset = TokenizedDataset(corpus_path, seq_len=seq_len, stride=stride)
    print(f"Train windows: {len(dataset):,} (seq_len={seq_len}, stride={stride})")

    # Query top n_neighbors + a small margin so there's still something left
    # after dropping self-retrieval hits, without a second query round-trip
    # in the common case.
    query_k = n_neighbors + 3

    all_neighbor_ids = np.zeros((len(dataset), n_neighbors, neighbor_len), dtype=np.int32)
    for start in range(0, len(dataset), embed_batch_size):
        batch_indices = range(start, min(start + embed_batch_size, len(dataset)))
        window_texts = []
        for idx in batch_indices:
            input_ids, _ = dataset[idx]
            window_texts.append(engine.tokenizer.decode(input_ids.tolist()))

        batch_results = retriever.query_batch(window_texts, top_k=query_k)

        for local_i, idx in enumerate(batch_indices):
            window_text = window_texts[local_i]
            results = batch_results[local_i]
            # Self-retrieval exclusion — see module docstring.
            kept = [r for r in results if r.excerpt not in window_text][:n_neighbors]
            for n_i in range(n_neighbors):
                if n_i < len(kept):
                    ids = _tokenize_neighbor(engine.tokenizer, kept[n_i].excerpt, neighbor_len)
                else:
                    ids = [PAD_ID] * neighbor_len  # fewer than n_neighbors real hits
                all_neighbor_ids[idx, n_i] = ids

        if (start // embed_batch_size) % 20 == 0 or start + embed_batch_size >= len(dataset):
            done = min(start + embed_batch_size, len(dataset))
            print(f"  processed {done:,}/{len(dataset):,} window(s) …")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), all_neighbor_ids)
    print(f"\nDone. Wrote {all_neighbor_ids.shape} neighbor-ids array to {out_path}")
    print(
        'Set "neighbor_ids_path" in your training config to this file. It must '
        "be rebuilt if you change the corpus, seq_len, stride, retrieval index, "
        "n_neighbors, or neighbor_len."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a per-window retrieval_neighbors.npy for RETRO-style Chunked Cross-Attention.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--corpus", required=True, help="Tokenized corpus (.bin).")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint (.pt).")
    parser.add_argument("--vocab", required=True, help="BPE vocabulary JSON.")
    parser.add_argument("--seq-len", type=int, required=True,
                         help="Window length — must match what training uses.")
    parser.add_argument("--stride", type=int, required=True,
                         help="Window stride — must match what training uses.")
    parser.add_argument("--output", default="data/processed/retrieval_neighbors.npy",
                         help="Destination .npy neighbor-ids file.")
    parser.add_argument("--n-neighbors", type=int, default=2,
                         help="Neighbors retrieved per training window.")
    parser.add_argument("--neighbor-len", type=int, default=64,
                         help="Token length each neighbor chunk is truncated/padded to.")
    parser.add_argument("--index-dir", default=None,
                         help="Directory of an existing persistent RagIndex to load.")
    parser.add_argument("--corpus-dir", default=None,
                         help="Directory of raw .txt files to build a fresh index from.")
    parser.add_argument("--embed-batch-size", type=int, default=32,
                         help="Windows embedded per forward pass.")
    args = parser.parse_args()

    try:
        build(
            corpus_path=args.corpus,
            checkpoint_path=args.checkpoint,
            vocab_path=args.vocab,
            seq_len=args.seq_len,
            stride=args.stride,
            output_path=args.output,
            n_neighbors=args.n_neighbors,
            neighbor_len=args.neighbor_len,
            index_dir=args.index_dir,
            corpus_dir=args.corpus_dir,
            embed_batch_size=args.embed_batch_size,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
