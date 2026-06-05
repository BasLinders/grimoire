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
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import EOS_ID


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
    files = sorted(input_path.rglob("*.txt"))
    if not files:
        raise ValueError(f"No .txt files found under {input_path}")
    return files


def _read_texts(files: list[Path], encoding: str) -> list[str]:
    """Read and return the contents of each file as a string.

    Args:
        files: List of file paths to read.
        encoding: Text encoding (e.g. ``"utf-8"``).

    Returns:
        A list of raw text strings, one per file.
    """
    return [
        path.read_text(encoding=encoding, errors="replace")
        for path in files
    ]


def preprocess(
    input_path: str,
    output_path: str,
    vocab_path: str,
    vocab_size: int = 16384,
    encoding: str = "utf-8",
    on_progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Tokenize text files and write a flat binary array of token ids.

    If ``vocab_path`` does not exist, trains a new BPE encoder on the
    collected texts and saves it.  Otherwise loads the existing encoder.

    Encoding pipeline per document:
    1. ``BytePairEncoder.encode(text)`` → list of int ids
    2. Append ``EOS_ID`` to signal end of document
    3. Concatenate all documents into one long token sequence
    4. Write as a numpy int32 array to ``output_path``

    Args:
        input_path: Path to a directory of ``.txt`` files or a single file.
        output_path: Destination path for the ``.bin`` output file.
        vocab_path: Path to the BPE vocabulary JSON.  Trained and saved here
            if it does not exist; loaded from here otherwise.
        vocab_size: Target vocabulary size when training a new encoder.
            Ignored if ``vocab_path`` already exists.
        encoding: Text encoding of the source files.
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

    if not input_p.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    # --- Collect source files -------------------------------------------
    _emit(f"[1/4] Collecting text files from {input_p} ...")
    files = _collect_text_files(input_p)
    _emit(f"      Found {len(files)} file(s).")
    for path in files:
        _emit(f"      Reading {path.name} ({path.stat().st_size / 1024:.1f} KB)")
    texts = _read_texts(files, encoding)

    # --- Train or load BPE encoder -------------------------------------
    encoder = BytePairEncoder()
    if vocab_p.exists():
        _emit(f"[2/4] Loading existing vocabulary from {vocab_p} ...")
        encoder = BytePairEncoder.load(str(vocab_p))
        _emit(f"      Vocabulary size: {len(encoder):,}")
    else:
        _emit(f"[2/4] Training BPE encoder (vocab_size={vocab_size:,}) ...")
        _emit(f"      This may take 10–30 minutes on CPU.")

        def _bpe_progress(step: int, total: int) -> None:
            pct = 100 * step // total
            _emit(f"      BPE merges: {step:,} / {total:,}  ({pct}%)")

        encoder.train(
            texts,
            vocab_size=vocab_size,
            on_progress=_bpe_progress if on_progress is not None else None,
        )
        vocab_p.parent.mkdir(parents=True, exist_ok=True)
        encoder.save(str(vocab_p))
        _emit(f"      Vocabulary size: {len(encoder):,}")
        _emit(f"      Saved to {vocab_p}")

    # --- Encode all documents -----------------------------------------
    _emit(f"[3/4] Encoding documents ...")
    all_ids: list[int] = []
    for i, (path, text) in enumerate(zip(files, texts)):
        ids = encoder.encode(text)
        ids.append(EOS_ID)          # document boundary marker
        all_ids.extend(ids)
        _emit(f"      [{i+1}/{len(files)}] {path.name}: {len(ids):,} tokens")
    _emit(f"      Total tokens: {len(all_ids):,}")

    # --- Write binary file --------------------------------------------
    _emit(f"[4/4] Writing {output_p} ...")
    output_p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(all_ids, dtype=np.int32)
    fp = np.memmap(str(output_p), dtype=np.int32, mode="w+", shape=(len(arr),))
    fp[:] = arr
    fp.flush()
    del fp
    size_mb = output_p.stat().st_size / (1024 ** 2)
    _emit(f"      Written {len(arr):,} tokens ({size_mb:.1f} MB) to {output_p}")
    return len(arr)


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
    args = parser.parse_args()

    try:
        preprocess(
            input_path=args.input,
            output_path=args.output,
            vocab_path=args.vocab,
            vocab_size=args.vocab_size,
            encoding=args.encoding,
        )
        print("\nPreprocessing complete.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
