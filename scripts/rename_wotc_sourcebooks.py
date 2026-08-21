"""Prefix official D&D sourcebook files so --weight-pattern can target them.

87 official sourcebook/adventure files in data/corpus/saga/ (Monster
Manual, PHB/DMG, Curse of Strahd, Storm King's Thunder, etc. -- see
docs/expansion_PLAN.md's corpus categorization) have no consistent
filename prefix ("Monster Manual (2025).txt", "Curse of Strahd.txt", ...).
`grimoire-preprocess --weight-pattern GLOB:WEIGHT` matches against
filename only, so these files are untargetable by any specific glob and
silently fall through to the final catch-all pattern -- despite being
arguably the highest-value canonical content in the corpus.

This renames them in place to "wotc_book_<slug>.txt", so a
"wotc_book_*:WEIGHT" --weight-pattern entry can finally target them
deliberately instead of by accident.

Selection rule: any .txt file directly under --corpus-dir whose name does
NOT already start with `word_` (the convention every other tagged source
in this corpus follows -- gutenberg_*, rpg_*, wp_*, srd_*, dnd_*, etc.).
This is exactly the same rule used to identify these files during corpus
categorization, and it makes a second run idempotent for free: once a
file is renamed to wotc_book_*, it matches the "already has a prefix"
exclusion and is skipped on any future run.

Usage
-----
    python scripts/rename_wotc_sourcebooks.py --dry-run   # preview only
    python scripts/rename_wotc_sourcebooks.py              # actually rename
    python scripts/rename_wotc_sourcebooks.py --corpus-dir data/corpus/saga/ --prefix wotc_book_
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

_PREFIXED = re.compile(r"^[A-Za-z0-9]+_")
_SLUG_JUNK = re.compile(r"[^a-z0-9]+")


def _slugify(stem: str) -> str:
    """Lowercase, collapse anything non-alphanumeric to a single underscore.

    "Monster Manual (2025)" -> "monster_manual_2025"
    "Curse of Strahd" -> "curse_of_strahd"
    "Baldur's Gate_ Descent Into Avernus" -> "baldur_s_gate_descent_into_avernus"
    """
    slug = _SLUG_JUNK.sub("_", stem.lower()).strip("_")
    return slug


def find_unprefixed(corpus_dir: Path) -> list[Path]:
    """Every .txt file directly under corpus_dir with no existing word_ prefix."""
    return sorted(
        p for p in corpus_dir.glob("*.txt")
        if not _PREFIXED.match(p.stem)
    )


def plan_renames(files: list[Path], prefix: str) -> list[tuple[Path, Path]]:
    """Build (source, destination) pairs, skipping any destination collision."""
    plan = []
    seen_dest: set[Path] = set()
    for src in files:
        dest = src.with_name(f"{prefix}{_slugify(src.stem)}.txt")
        if dest in seen_dest or dest.exists():
            print(f"  ! Skipping {src.name} -- destination {dest.name} already exists.")
            continue
        seen_dest.add(dest)
        plan.append((src, dest))
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefix official D&D sourcebook files with a consistent "
                     "'wotc_book_' name so --weight-pattern can target them.",
    )
    parser.add_argument("--corpus-dir", default="data/corpus/saga/", metavar="DIR")
    parser.add_argument(
        "--prefix", default="wotc_book_", metavar="PREFIX",
        help="Filename prefix to apply (default: wotc_book_).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the rename plan without touching any files.",
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        raise SystemExit(f"Not a directory: {corpus_dir}")

    unprefixed = find_unprefixed(corpus_dir)
    if not unprefixed:
        print(f"No unprefixed .txt files found in {corpus_dir} -- nothing to do "
              f"(already renamed, or run again after a fresh sourcebook scrape).")
        return

    plan = plan_renames(unprefixed, args.prefix)
    print(f"{len(plan)} file(s) to rename in {corpus_dir}"
          + (" (dry run, no changes made):" if args.dry_run else ":"))
    for src, dest in plan:
        print(f"  {src.name}  ->  {dest.name}")
        if not args.dry_run:
            src.rename(dest)

    if args.dry_run:
        print("\nDry run only -- rerun without --dry-run to actually rename.")
    else:
        print(f"\nDone. Add --weight-pattern \"{args.prefix}*:WEIGHT\" to your next "
              f"grimoire-preprocess run to target these files deliberately.")


if __name__ == "__main__":
    main()
