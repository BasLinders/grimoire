"""Preview what --quality-filter would drop from a corpus, without touching any files.

Non-destructive companion to ``grimoire_ai.llm.data.quality_filter``, in
the same relationship ``dedup_corpus.py`` already has to
``grimoire_ai.llm.data.dedup``: run this against a ``--corpus-dir`` before
trusting the automatic ``--quality-filter`` flag on ``grimoire-preprocess``,
especially right after adding a new scraper whose output hasn't been
eyeballed yet.

Usage
-----
    python scripts/score_corpus_quality.py --corpus-dir data/corpus/saga/
    python scripts/score_corpus_quality.py --corpus-dir data/corpus/saga/ --report quality_report.jsonl
    python scripts/score_corpus_quality.py --corpus-dir data/corpus/saga/ --min-chars 500
"""

import argparse
import json
from pathlib import Path

from grimoire_ai.llm.data.quality_filter import QualityThresholds, score_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview what --quality-filter would drop from a corpus "
                     "directory, without touching any files.",
    )
    parser.add_argument("--corpus-dir", default="data/corpus/saga/")
    parser.add_argument(
        "--pattern", default="*.txt",
        help="Glob (relative to --corpus-dir) of files to check.",
    )
    parser.add_argument("--min-chars", type=int, default=QualityThresholds.min_chars)
    parser.add_argument("--min-words", type=int, default=QualityThresholds.min_words)
    parser.add_argument("--min-alpha-ratio", type=float, default=QualityThresholds.min_alpha_ratio)
    parser.add_argument("--max-symbol-ratio", type=float, default=QualityThresholds.max_symbol_ratio)
    parser.add_argument("--max-mean-word-length", type=float, default=QualityThresholds.max_mean_word_length)
    parser.add_argument("--max-short-line-ratio", type=float, default=QualityThresholds.max_short_line_ratio)
    parser.add_argument(
        "--max-top-ngram-repetition-ratio", type=float,
        default=QualityThresholds.max_top_ngram_repetition_ratio,
    )
    parser.add_argument(
        "--report", default=None, metavar="PATH",
        help="Write a JSONL audit log of dropped files (name, reasons, stats) to PATH.",
    )
    args = parser.parse_args()

    thresholds = QualityThresholds(
        min_chars=args.min_chars,
        min_words=args.min_words,
        min_alpha_ratio=args.min_alpha_ratio,
        max_symbol_ratio=args.max_symbol_ratio,
        max_mean_word_length=args.max_mean_word_length,
        max_short_line_ratio=args.max_short_line_ratio,
        max_top_ngram_repetition_ratio=args.max_top_ngram_repetition_ratio,
    )

    corpus_dir = Path(args.corpus_dir)
    files = sorted(corpus_dir.glob(args.pattern))
    if not files:
        print(f"No files matched --pattern {args.pattern!r} under {corpus_dir}")
        return

    print(f"Scoring {len(files)} file(s) under {corpus_dir} (pattern={args.pattern!r}) ...\n")

    dropped_entries: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        report = score_document(text, thresholds)
        if not report.keep:
            reasons_str = "; ".join(report.reasons)
            print(f"  DROP  {path.name}: {reasons_str}")
            dropped_entries.append({
                "file": path.name,
                "reasons": report.reasons,
                "stats": report.stats,
            })

    if not dropped_entries:
        print(f"No documents would be dropped at these thresholds ({len(files)} checked).")
        return

    print(f"\n{len(dropped_entries)}/{len(files)} document(s) would be dropped by --quality-filter.")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            for entry in dropped_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(dropped_entries)} dropped-document report(s) to {args.report}")


if __name__ == "__main__":
    main()
