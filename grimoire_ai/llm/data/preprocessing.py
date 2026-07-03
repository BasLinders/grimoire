"""Offline corpus preprocessing: raw text files → binary token array.

Run this script once before training to convert your domain text files
into a memory-mapped binary that ``TokenizedDataset`` can read efficiently
during training.  Tokenization is the slow step; doing it offline means
the hot training loop reads pre-computed integers rather than running BPE
on every epoch.

Output format
-------------
A flat numpy int32 array written to a ``.bin`` file via ``numpy.memmap``.
Documents are separated by the EOS token (id 2) so the model learns that
a response ends at EOS and the next document is a fresh context.

Usage
-----
Train the tokenizer first (if you haven't already):

    python -m grimoire.llm.data.preprocessing \\
        --input  data/raw/ \\
        --output data/processed/corpus.bin \\
        --vocab  data/tokenizer/bpe.json \\
        --vocab-size 16384

If ``--vocab`` points to a file that does not yet exist, the BPE encoder
is trained on the input files first and saved to that path automatically.
If the file already exists it is loaded directly (training is skipped).

Arguments
---------
--input      Path to a directory of .txt files, or a single .txt file.
--output     Destination path for the .bin token array.
--vocab      Path to the BPE vocabulary JSON (read or write).
--vocab-size  Target BPE vocabulary size (used only when training anew).
--encoding   Text encoding of the input files (default: utf-8).
"""

import argparse
import fnmatch
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID


def _collect_text_files(input_path: Path) -> list[Path]:
    """Collect all .txt files under a directory (or return a single file).

    Args:
        input_path: A directory or a single ``.txt`` file.

    Returns:
        A sorted list of ``.txt`` file paths found under ``input_path``.

    Raises:
        ValueError: If no ``.txt`` files are found.
    """
    if input_path.is_file():
        return [input_path]
    # Force the OS to flush its directory cache before globbing.
    # On Windows, a long-running process may not see files added after startup
    # without explicitly re-reading the directory via os.scandir.
    import os
    for dirpath, _, _ in os.walk(input_path):
        with os.scandir(dirpath):
            pass
    files = sorted(input_path.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found under {input_path}")
    return files


def _load_texts(files: list[Path], encoding: str) -> list[str]:
    """Read all files into memory — only used when deduplication is requested."""
    return [
        path.read_text(encoding=encoding, errors="replace")
        for path in files
    ]


def _sample_texts_for_bpe(
    files: list[Path],
    texts: Optional[list[str]],
    max_files: Optional[int],
    encoding: str,
) -> list[str]:
    """Return up to *max_files* texts for BPE vocabulary training.

    When *texts* is already loaded (dedup path), samples from that list to
    avoid a second disk read.  Otherwise reads a random sample from disk.
    A good vocabulary only needs to see a representative slice of the corpus —
    sampling 500 files already covers the long tail for typical domain corpora.
    """
    import random
    n = len(files)
    if max_files is None or n <= max_files:
        if texts is not None:
            return texts
        return [p.read_text(encoding=encoding, errors="replace") for p in files]
    # Sample without replacement; sort for deterministic disk-read ordering.
    indices = sorted(random.sample(range(n), max_files))
    if texts is not None:
        return [texts[i] for i in indices]
    return [files[i].read_text(encoding=encoding, errors="replace") for i in indices]


def _resolve_weight(path: Path, weight_rules: list[tuple[str, float]]) -> float:
    """Return the weight for *path* from the first matching glob rule.

    Rules are checked in the order given; the first pattern whose
    ``fnmatch`` matches ``path.name`` wins. Files matching no rule get the
    neutral weight ``1.0`` — identical to today's unweighted behaviour.
    """
    for pattern, weight in weight_rules:
        if fnmatch.fnmatch(path.name, pattern):
            return weight
    return 1.0


def preprocess(
    input_path: str,
    output_path: str,
    vocab_path: str,
    vocab_size: int = 16384,
    encoding: str = "utf-8",
    dedup: bool = False,
    dedup_threshold: float = 0.8,
    bpe_sample_size: Optional[int] = 500,
    extend_vocab: bool = False,
    weight_rules: Optional[list[tuple[str, float]]] = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Tokenize text files and write a flat binary array of token ids.

    If ``vocab_path`` does not exist, trains a new BPE encoder on up to
    ``bpe_sample_size`` randomly-sampled source files and saves it.  Otherwise
    loads the existing encoder.  All files are then encoded and written to
    ``output_path`` one at a time — no full corpus is ever held in RAM, so
    the function scales to corpora of arbitrary size.

    Encoding pipeline per document:
    1. ``BytePairEncoder.encode(text)`` → list of int ids
    2. Append ``EOS_ID`` to signal end of document
    3. Write the int32 array directly to the output file

    Args:
        input_path: Path to a directory of ``.txt`` files or a single file.
        output_path: Destination path for the ``.bin`` output file.
        vocab_path: Path to the BPE vocabulary JSON.  Trained and saved here
            if it does not exist; loaded from here otherwise.
        vocab_size: Target vocabulary size when training a new encoder.
            Ignored if ``vocab_path`` already exists.
        encoding: Text encoding of the source files.
        dedup: When ``True``, near-duplicate documents are detected with
            MinHash + LSH and only one representative of each duplicate cluster
            is kept before tokenisation.  Requires loading all files into RAM.
        dedup_threshold: Minimum estimated Jaccard similarity for two documents
            to be treated as near duplicates (only used when ``dedup`` is set).
        bpe_sample_size: Maximum number of files loaded into RAM for BPE
            vocabulary training.  A random sample is drawn when the corpus is
            larger.  ``None`` loads all files (original behaviour).  Ignored
            when ``vocab_path`` already exists and ``extend_vocab`` is False.
        extend_vocab: When ``True`` and ``vocab_path`` already exists, grow
            the existing vocabulary up to ``vocab_size`` by learning
            additional merges (see ``BytePairEncoder.extend``) instead of
            using it unchanged.  Existing token ids are preserved exactly,
            so checkpoints trained against the old vocabulary remain valid —
            only the newly appended ids need fresh embedding rows.  This is
            an explicit opt-in; the default behaviour (load as-is) is
            unchanged.  Ignored if ``vocab_size`` is not larger than the
            existing vocabulary.
        weight_rules: Optional list of ``(glob_pattern, weight)`` pairs used
            to mark which source documents matter more for training, e.g.
            ``[("rpg_*", 3.0), ("dnd_*", 3.0), ("gutenberg_*", 1.0)]``. Each
            file is matched against its filename by the first pattern that
            fits (``fnmatch``); unmatched files default to weight ``1.0`` —
            identical to today's behaviour. This does *not* change the
            token stream itself (no data is duplicated or dropped); it
            writes two small sidecar files next to ``output_path`` —
            ``<output>.doc_end_offsets.npy`` (cumulative end token-index of
            each document) and ``<output>.doc_weights.npy`` (weight per
            document) — that ``scripts/build_source_weights.py`` later
            converts into a per-window ``sample_weights_path`` array for
            ``WeightedRandomSampler`` (the same mechanism
            ``scripts/score_difficulty.py`` already uses for difficulty-based
            weighting). When ``None`` (default), no sidecar files are
            written and behaviour is unchanged.
        on_progress: Optional callable invoked with a progress message string
            at each major step and during BPE merge iterations.  When
            ``None`` messages are printed to stdout — existing CLI behaviour
            is unchanged.

    Returns:
        Total number of tokens written to the output file.

    Raises:
        FileNotFoundError: If ``input_path`` does not exist.
        ValueError: If no ``.txt`` files are found under ``input_path``.
    """
    def _emit(msg: str) -> None:
        if on_progress is not None:
            on_progress(msg)
        else:
            print(msg)

    input_p  = Path(input_path)
    output_p = Path(output_path)
    vocab_p  = Path(vocab_path)

    if bpe_sample_size is not None and bpe_sample_size <= 0:
        bpe_sample_size = None

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # --- Collect source files -------------------------------------------
    _emit(f"[1/3] Collecting text files from {input_p} ...")
    files = _collect_text_files(input_p)
    _emit(f"      Found {len(files)} file(s).")

    # --- Optional near-duplicate removal --------------------------------
    # Dedup requires all texts in RAM; accept that cost only when requested.
    texts: Optional[list[str]] = None
    if dedup:
        _emit(f"      Loading {len(files)} files for deduplication ...")
        texts = _load_texts(files, encoding)
        from grimoire_ai.llm.data.dedup import deduplicate_indices
        _emit(f"      Deduplicating (threshold={dedup_threshold}) ...")
        kept, clusters = deduplicate_indices(texts, threshold=dedup_threshold)
        removed = len(texts) - len(kept)
        if removed:
            files = [files[i] for i in kept]
            texts = [texts[i] for i in kept]
            _emit(f"      Removed {removed} near-duplicate document(s) in "
                  f"{len(clusters)} cluster(s); {len(texts)} remain.")
        else:
            _emit("      No near-duplicates found.")

    # --- Train or load BPE encoder -------------------------------------
    encoder = BytePairEncoder()
    if vocab_p.exists():
        _emit(f"[2/3] Loading existing vocabulary from {vocab_p} ...")
        encoder = BytePairEncoder.load(str(vocab_p))
        _emit(f"      Vocabulary size: {len(encoder):,}")

        if extend_vocab and vocab_size > len(encoder):
            _emit(f"      Extending vocabulary to {vocab_size:,} "
                  f"(existing token ids are preserved) ...")

            def _bpe_progress(step: int, total: int) -> None:
                pct = 100 * step // total
                _emit(f"      New merges: {step:,} / {total:,}  ({pct}%)")

            sample = _sample_texts_for_bpe(files, texts, bpe_sample_size, encoding)
            encoder.extend(
                sample,
                vocab_size=vocab_size,
                on_progress=_bpe_progress if on_progress is not None else None,
            )
            encoder.save(str(vocab_p))
            _emit(f"      Vocabulary size: {len(encoder):,}")
            _emit(f"      Saved to {vocab_p}")
    else:
        _emit(f"[2/3] Training BPE encoder (vocab_size={vocab_size:,}) ...")
        _emit(f"      This may take several minutes on CPU.")

        def _bpe_progress(step: int, total: int) -> None:
            pct = 100 * step // total
            _emit(f"      BPE merges: {step:,} / {total:,}  ({pct}%)")

        sample = _sample_texts_for_bpe(files, texts, bpe_sample_size, encoding)
        n_sample = len(sample)
        _emit(f"      Sampling {n_sample}/{len(files)} file(s) for vocabulary training."
              if n_sample < len(files) else
              f"      Using all {n_sample} file(s) for vocabulary training.")
        encoder.train(
            sample,
            vocab_size=vocab_size,
            on_progress=_bpe_progress if on_progress is not None else None,
        )
        vocab_p.parent.mkdir(parents=True, exist_ok=True)
        encoder.save(str(vocab_p))
        _emit(f"      Vocabulary size: {len(encoder):,}")
        _emit(f"      Saved to {vocab_p}")

    # --- Encode and write documents (streaming — no full buffer in RAM) ---
    # Write to a sibling temp file and rename on success so a failed run never
    # leaves a partial/corrupt .bin at the destination path.
    _emit(f"[3/3] Encoding and writing {output_p} ...")
    output_p.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_p.with_suffix(".bin.tmp")
    total_tokens = 0
    doc_end_offsets: list[int] = []
    doc_weights: list[float] = []
    try:
        with open(str(tmp_path), "wb") as f_out:
            for i, path in enumerate(files):
                text = texts[i] if texts is not None else path.read_text(
                    encoding=encoding, errors="replace"
                )
                ids = encoder.encode(text) + [EOS_ID]   # +[EOS_ID] avoids mutating encoder cache
                np.array(ids, dtype=np.int32).tofile(f_out)
                total_tokens += len(ids)
                if weight_rules is not None:
                    doc_end_offsets.append(total_tokens)
                    doc_weights.append(_resolve_weight(path, weight_rules))
                _emit(f"      [{i+1}/{len(files)}] {path.name}: {len(ids):,} tokens")
        tmp_path.replace(output_p)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    size_mb = output_p.stat().st_size / (1024 ** 2)
    _emit(f"      Total: {total_tokens:,} tokens ({size_mb:.1f} MB) -> {output_p}")

    if weight_rules is not None:
        offsets_path = output_p.with_suffix(output_p.suffix + ".doc_end_offsets.npy")
        weights_path = output_p.with_suffix(output_p.suffix + ".doc_weights.npy")
        np.save(str(offsets_path), np.array(doc_end_offsets, dtype=np.int64))
        np.save(str(weights_path), np.array(doc_weights, dtype=np.float32))
        _emit(
            f"      Wrote per-document weight metadata for {len(doc_weights)} "
            f"document(s) -> {offsets_path.name}, {weights_path.name}"
        )
        _emit(
            "      Run scripts/build_source_weights.py against these to produce "
            "a per-window sample_weights.npy for training."
        )

    return total_tokens


def main() -> None:
    """Command-line entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description="Tokenize text files into a binary corpus for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", required=True,
        help="Directory of .txt files or a single .txt file.",
    )
    parser.add_argument(
        "--output", default="data/processed/corpus.bin",
        help="Output path for the binary token array.",
    )
    parser.add_argument(
        "--vocab", default="data/tokenizer/bpe.json",
        help="Path to the BPE vocabulary JSON (trained if missing).",
    )
    parser.add_argument(
        "--vocab-size", type=int, default=16384,
        help="Target vocabulary size when training a new encoder.",
    )
    parser.add_argument(
        "--encoding", default="utf-8",
        help="Text encoding of the input files.",
    )
    parser.add_argument(
        "--dedup", action="store_true",
        help="Remove near-duplicate documents (MinHash + LSH) before tokenising.",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.8,
        help="Min estimated Jaccard similarity to treat documents as duplicates.",
    )
    parser.add_argument(
        "--bpe-sample", type=int, default=500, metavar="N",
        help="Max files sampled for BPE vocabulary training (0 = all files).",
    )
    parser.add_argument(
        "--extend-vocab", action="store_true",
        help="If --vocab already exists, grow it to --vocab-size by learning "
             "new merges instead of using it unchanged. Existing token ids "
             "are preserved, so old checkpoints stay loadable (just resize "
             "the embedding/output layers and randomly init the new rows). "
             "Opt-in only — default behaviour loads the existing vocab as-is.",
    )
    parser.add_argument(
        "--weight-pattern", action="append", metavar="GLOB:WEIGHT", default=None,
        help="Mark files matching a glob (against filename only) with a "
             "training weight, e.g. --weight-pattern 'rpg_*:3' --weight-pattern "
             "'dnd_*:3' --weight-pattern 'gutenberg_*:1'. Repeatable; first "
             "matching pattern wins; unmatched files get weight 1.0. Writes "
             "<output>.doc_end_offsets.npy and <output>.doc_weights.npy "
             "sidecar files for scripts/build_source_weights.py to consume — "
             "does not change the corpus.bin itself. Omit entirely for "
             "today's unweighted behaviour.",
    )
    args = parser.parse_args()

    weight_rules = None
    if args.weight_pattern:
        weight_rules = []
        for spec in args.weight_pattern:
            pattern, _, weight_str = spec.rpartition(":")
            if not pattern:
                print(f"\nError: --weight-pattern must be GLOB:WEIGHT, got {spec!r}", file=sys.stderr)
                sys.exit(1)
            try:
                weight_rules.append((pattern, float(weight_str)))
            except ValueError:
                print(f"\nError: weight in --weight-pattern must be a number, got {spec!r}", file=sys.stderr)
                sys.exit(1)

    try:
        preprocess(
            input_path=args.input,
            output_path=args.output,
            vocab_path=args.vocab,
            vocab_size=args.vocab_size,
            encoding=args.encoding,
            dedup=args.dedup,
            dedup_threshold=args.dedup_threshold,
            bpe_sample_size=args.bpe_sample,
            extend_vocab=args.extend_vocab,
            weight_rules=weight_rules,
        )
        print("\nPreprocessing complete.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
