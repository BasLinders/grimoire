"""Strip Stack Exchange dump scaffolding (vote scores, tags, Markdown headings,
chunk separators) out of ``rpg_se_*.txt`` corpus files.

``scripts/scrape_stackexchange_rpg.py`` writes each Q&A pair as:

    # Question title
    Tags: [tag1] [tag2]
    Score: N

    <question body>

    ## Answer  (score: N)

    <answer body>

    ---

The ``Tags:``/``Score:``/``## Answer ... (score: N)``/``---`` lines are pure
forum bookkeeping, not natural-language content -- and it shows: a model
pretrained on this corpus has been observed emitting literal
``## Answer (score: 4)`` fragments mid-generation. This script removes that
scaffolding in place, keeping the actual question/answer prose (including the
question title itself, with its leading ``# `` heading marker stripped, since
the title text is real content worth keeping).

Originals are backed up to ``<corpus_dir>/../saga_se_qa_source/`` before
anything is overwritten, since ``data/`` is gitignored and there is no git
history to fall back on. This backup turned out to serve an ongoing purpose,
not just a one-off revert point: ``grimoire_ai.llm.data.qa_pairs.load_qa_pairs``
(used by ``scripts/build_finetune_data_from_qa.py`` and ``embed_tune.py``'s
``--qa-corpus-dir``) parses Q&A structure by keying off exactly the markers
this script strips -- pointed at the live, cleaned corpus, it silently
returns zero pairs. Point those tools at this directory instead of
``data/corpus/saga/`` going forward.

Usage
-----
    python scripts/clean_stackexchange_markup.py --dry-run
    python scripts/clean_stackexchange_markup.py
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

_TAGS_LINE = re.compile(r"^Tags: .*\n?", re.MULTILINE)
_SCORE_LINE = re.compile(r"^Score: -?\d+\n?", re.MULTILINE)
_ANSWER_HEADER = re.compile(r"^## .*\(score: -?\d+\)\s*\n?", re.MULTILINE)
_SEPARATOR_LINE = re.compile(r"^---\n?", re.MULTILINE)
_TITLE_HEADING = re.compile(r"^# (.+)$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    text = _TAGS_LINE.sub("", text)
    text = _SCORE_LINE.sub("", text)
    text = _ANSWER_HEADER.sub("", text)
    text = _SEPARATOR_LINE.sub("", text)
    text = _TITLE_HEADING.sub(r"\1", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", default="data/corpus/saga")
    parser.add_argument("--pattern", default="rpg_se_*.txt")
    parser.add_argument(
        "--backup-dir", default=None,
        help="Where originals are copied before overwriting. "
             "Default: <corpus-dir>/../saga_se_qa_source/",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    files = sorted(corpus_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern!r} in {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    backup_dir = Path(args.backup_dir) if args.backup_dir else corpus_dir.parent / "saga_se_qa_source"

    total_before = 0
    total_after = 0
    changed = 0

    for path in files:
        original = path.read_text(encoding="utf-8")
        cleaned = clean_text(original)
        total_before += len(original)
        total_after += len(cleaned)
        if cleaned == original:
            continue
        changed += 1
        if args.dry_run:
            continue
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / path.name
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
        path.write_text(cleaned, encoding="utf-8")

    verb = "Would change" if args.dry_run else "Changed"
    print(f"{verb} {changed}/{len(files)} file(s).")
    print(f"Total size: {total_before:,} -> {total_after:,} chars "
          f"({100 * (total_before - total_after) / total_before:.1f}% removed).")
    if not args.dry_run and changed:
        print(f"Originals backed up to {backup_dir}")
        print("Re-run grimoire-preprocess (and rebuild sample_weights.npy) "
              "to pick up the cleaned corpus.")


if __name__ == "__main__":
    main()
