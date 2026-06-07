"""Build the Saga corpus from the D&D 5e SRD and hand-authored math references.

Downloads the CC-licensed D&D 5e SRD (BTMorton/dnd-5e-srd) as a single
Markdown file, splits it into sections by top-level heading, converts each
section to plain text via grimoire's ingest pipeline, and writes one .txt
file per section to the output directory.

Also copies the hand-authored probability and encounter-math reference files
from scripts/saga_references/ into the same output directory.

Output: data/corpus/saga/*.txt  (one file per SRD section + references)

Usage
-----
    python scripts/build_saga_corpus.py
    python scripts/build_saga_corpus.py --output-dir path/to/corpus/
    python scripts/build_saga_corpus.py --skip-download   # only copy references

The script is idempotent: re-running it overwrites existing files.

License
-------
The D&D 5e SRD is published by Wizards of the Coast under the Creative Commons
Attribution 4.0 International License (CC-BY 4.0).
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

_SRD_URL = (
    "https://raw.githubusercontent.com/BTMorton/dnd-5e-srd/master/5esrd.md"
)

_REFERENCES_DIR = Path(__file__).parent / "saga_references"

# Sections to keep (case-insensitive prefix match on heading text).
# Omit lore-heavy narrative sections that add noise without rules content.
_KEEP_SECTIONS: set[str] = {
    "races",
    "barbarian",
    "bard",
    "cleric",
    "druid",
    "fighter",
    "monk",
    "paladin",
    "ranger",
    "rogue",
    "sorcerer",
    "warlock",
    "wizard",
    "beyond 1st level",
    "equipment",
    "feats",
    "using ability scores",
    "adventuring",
    "combat",
    "spellcasting",
    "traps",
    "poisons",
    "magic items",
    "monsters",
    "appendix ph-a: conditions",
    "appendix mm-b: nonplayer characters",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch(url: str, timeout: int = 30) -> str:
    """Fetch raw text from *url*."""
    import urllib.request
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


def _split_by_heading(markdown: str) -> list[tuple[str, str]]:
    """Split *markdown* into (heading_slug, content) pairs.

    Handles both ATX headings (# Heading) and setext headings
    (Heading\\n=======).  Returns one entry per top-level section.
    The heading line is included in the content.
    """
    # Normalise CRLF → LF
    markdown = markdown.replace("\r\n", "\n")

    # Match setext-style (underlined with =) or ATX-style (# ) headings.
    pattern = re.compile(
        r"^(.+)\n=+[ \t]*$"   # setext h1
        r"|^(# .+)$",          # ATX h1
        re.MULTILINE,
    )

    positions: list[tuple[int, str]] = []
    for m in pattern.finditer(markdown):
        heading_text = (m.group(1) or m.group(2)).strip().lstrip("# ")
        positions.append((m.start(), heading_text))

    sections: list[tuple[str, str]] = []
    for i, (start, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(markdown)
        slug = re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_")
        sections.append((slug, markdown[start:end]))

    return sections


def _convert(markdown_text: str) -> str:
    """Convert a markdown string to plain text."""
    import tempfile
    from grimoire_ai.corpus.ingest import CleaningLevel, from_markdown

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(markdown_text)
        tmp = f.name

    try:
        return from_markdown(tmp, cleaning=CleaningLevel.STANDARD)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _should_keep(slug: str) -> bool:
    """Return True if this section slug matches one of the keep patterns."""
    normalized = slug.replace("_", " ")
    return any(normalized.startswith(k) for k in _KEEP_SECTIONS)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(output_dir: str = "data/corpus/saga/", skip_download: bool = False) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- SRD download and split ---------------------------------------------
    if not skip_download:
        print(f"Fetching D&D 5e SRD from {_SRD_URL} …")
        try:
            raw = _fetch(_SRD_URL)
            print(f"  Downloaded {len(raw):,} bytes.  Splitting into sections …")
        except Exception as exc:
            print(f"  FAILED to download SRD: {exc}")
            raw = None

        if raw:
            sections = _split_by_heading(raw)
            kept = [(s, c) for s, c in sections if _should_keep(s)]
            skipped = len(sections) - len(kept)
            print(
                f"  {len(sections)} sections found; keeping {len(kept)}, "
                f"skipping {skipped} (lore/appendix)."
            )

            for i, (slug, content) in enumerate(kept, 1):
                dest = out / f"srd_{slug}.txt"
                print(
                    f"  [{i:02d}/{len(kept)}] {slug} …",
                    end=" ",
                    flush=True,
                )
                try:
                    text = _convert(content)
                    dest.write_text(text, encoding="utf-8")
                    print(f"✓  ({len(text.split()):,} words)")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                time.sleep(0.05)
    else:
        print("Skipping SRD download (--skip-download).")

    # ---- Hand-authored references -------------------------------------------
    if _REFERENCES_DIR.exists():
        ref_files = sorted(_REFERENCES_DIR.glob("*.txt"))
        if ref_files:
            print(f"\nCopying {len(ref_files)} reference files → {out}/")
            for src in ref_files:
                dest = out / src.name
                dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"  {src.name}  ✓")
    else:
        print(f"\nNo reference directory found at {_REFERENCES_DIR} — skipping.")

    total = len(list(out.glob("*.txt")))
    print(f"\nDone.  {total} corpus files in {out.resolve()}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Saga corpus from D&D 5e SRD.")
    parser.add_argument(
        "--output-dir",
        default="data/corpus/saga/",
        help="Directory to write corpus .txt files (default: data/corpus/saga/).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip SRD download; only copy hand-authored reference files.",
    )
    args = parser.parse_args(argv)
    build(output_dir=args.output_dir, skip_download=args.skip_download)


if __name__ == "__main__":
    main()
