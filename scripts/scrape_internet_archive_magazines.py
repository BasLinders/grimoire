"""Download Dragon / Dungeon Magazine text from the Internet Archive.

The Internet Archive hosts scanned issues of Dragon Magazine (1976–2007) and
Dungeon Magazine (1986–2007).  For every issue it OCRs, it generates a
``_djvu.txt`` file containing the extracted plain text.  This script queries
the Archive's search API, filters to freely accessible items (skipping
borrow-only controlled-digital-lending entries), downloads the pre-extracted
text, cleans it, and writes one ``.txt`` file per issue into the output
directory.

Important note on copyright
---------------------------
Copyright status of individual issues varies.  Issues published before 1978
without a timely renewal may be in the public domain; later issues are still
under copyright.  This script skips anything the Internet Archive marks as
access-restricted (borrow-only).  Review the access status of downloaded
files against your jurisdiction's fair-use / research-use rules before
distributing the resulting corpus.

Usage
-----
    python scripts/scrape_internet_archive_magazines.py
    python scripts/scrape_internet_archive_magazines.py --magazine dragon --limit 30
    python scripts/scrape_internet_archive_magazines.py --out data/corpus/saga/ --delay 2.5

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote as urlquote

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IA_SEARCH   = "https://archive.org/advancedsearch.php"
IA_METADATA = "https://archive.org/metadata"
IA_DOWNLOAD = "https://archive.org/download"

USER_AGENT  = (
    "grimoire-corpus-scraper/1.0 (academic/research use; "
    "github.com/BasLinders/grimoire)"
)

# Search queries — broad enough to catch variant titles in the Archive's
# metadata, e.g. "Dragon Magazine #50" or "Dragon (magazine) issue 100".
QUERIES: dict[str, str] = {
    "dragon":  '(title:"Dragon Magazine" OR title:"Dragon (magazine)") AND mediatype:texts',
    "dungeon": '(title:"Dungeon Magazine" OR title:"Dungeon Adventures") AND mediatype:texts',
}

# ---------------------------------------------------------------------------
# OCR text cleaning
# ---------------------------------------------------------------------------

# Form-feed characters mark page boundaries in djvu.txt files.
_FORMFEED = re.compile(r"\x0c")

# Repeated magazine name headers printed at the top of each page.
_HEADER = re.compile(
    r"^[ \t]*(DRAGON|DUNGEON|TSR|WIZARDS? OF THE COAST|PAIZO)[ \t\d]*$",
    re.MULTILINE | re.IGNORECASE,
)

# Isolated page numbers (a bare integer on its own line).
_PAGENUM = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)

# Long runs of dashes / rules that separate columns/sections.
_HRULE = re.compile(r"[-─═━]{5,}")

# Collapse 3+ consecutive blank lines to 2.
_BLANK_RUNS = re.compile(r"\n{3,}")

# Strip non-printable control characters (except newline/tab).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(raw: str) -> str:
    """Remove OCR noise common in scanned magazine text."""
    text = _FORMFEED.sub("\n\n", raw)
    text = _CONTROL.sub("", text)
    text = _HEADER.sub("", text)
    text = _PAGENUM.sub("", text)
    text = _HRULE.sub("", text)
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Internet Archive helpers
# ---------------------------------------------------------------------------

def _get(session: requests.Session, url: str, delay: float, **kwargs) -> requests.Response:
    """GET with rate-limiting and error propagation."""
    time.sleep(delay)
    resp = session.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


def _search_identifiers(
    session: requests.Session,
    query: str,
    limit: int,
    delay: float,
) -> list[str]:
    """Return up to *limit* archive.org identifiers matching *query*."""
    identifiers: list[str] = []
    page = 1
    page_size = 50

    while len(identifiers) < limit:
        want = min(page_size, limit - len(identifiers))
        params = {
            "q":      query,
            "fl[]":   "identifier",
            "rows":   want,
            "page":   page,
            "output": "json",
            "sort[]": "date asc",   # chronological order
        }
        data = _get(session, IA_SEARCH, delay, params=params).json()
        docs = data.get("response", {}).get("docs", [])
        if not docs:
            break
        identifiers.extend(d["identifier"] for d in docs)
        page += 1

    return identifiers[:limit]


def _is_freely_accessible(meta: dict) -> bool:
    """Return False for borrow-only controlled-digital-lending items."""
    restricted = meta.get("metadata", {}).get("access-restricted-item", "")
    return str(restricted).lower() != "true"


def _find_text_filename(files: list[dict]) -> str | None:
    """Prefer _djvu.txt (OCR output); fall back to any plain .txt."""
    djvu  = [f["name"] for f in files if f["name"].endswith("_djvu.txt")]
    plain = [
        f["name"] for f in files
        if f["name"].endswith(".txt") and not f["name"].endswith("_djvu.txt")
        # skip meta-files the Archive adds
        and not any(f["name"].endswith(s) for s in ("_files.xml.txt",))
    ]
    return (djvu or plain or [None])[0]


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------

def scrape_magazine(
    magazine: str,
    out_dir: Path,
    limit: int,
    delay: float,
    min_chars: int,
) -> None:
    query = QUERIES[magazine]
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    print(f"\n── {magazine.upper()} ──────────────────────────────────────────")
    print(f"Querying archive.org (up to {limit} issues) …")

    try:
        identifiers = _search_identifiers(session, query, limit, delay)
    except requests.RequestException as exc:
        print(f"Search failed: {exc}")
        return

    print(f"Found {len(identifiers)} candidate(s). Processing …\n")

    saved = skipped = errors = 0

    for ident in identifiers:
        out_path = out_dir / f"{magazine}_{ident}.txt"

        if out_path.exists():
            print(f"  [skip]   {ident}  (already saved)")
            skipped += 1
            continue

        # -- Fetch metadata ------------------------------------------------
        try:
            meta = _get(session, f"{IA_METADATA}/{ident}", delay).json()
        except requests.RequestException as exc:
            print(f"  [error]  {ident}  metadata: {exc}")
            errors += 1
            continue

        # -- Check access --------------------------------------------------
        if not _is_freely_accessible(meta):
            print(f"  [skip]   {ident}  (borrow-only / access-restricted)")
            skipped += 1
            continue

        # -- Find text file ------------------------------------------------
        files = meta.get("files", [])
        txt_name = _find_text_filename(files)
        if not txt_name:
            print(f"  [skip]   {ident}  (no text file in item)")
            skipped += 1
            continue

        # -- Download text file --------------------------------------------
        txt_url = f"{IA_DOWNLOAD}/{ident}/{urlquote(txt_name)}"
        try:
            raw = _get(session, txt_url, delay).text
        except requests.RequestException as exc:
            print(f"  [error]  {ident}  download: {exc}")
            errors += 1
            continue

        # -- Clean and save ------------------------------------------------
        cleaned = _clean(raw)
        if len(cleaned) < min_chars:
            print(
                f"  [skip]   {ident}  "
                f"(only {len(cleaned):,} chars after cleaning — likely a cover/index page)"
            )
            skipped += 1
            continue

        out_path.write_text(cleaned, encoding="utf-8")
        title = meta.get("metadata", {}).get("title", ident)
        print(f"  [saved]  {out_path.name}  {len(cleaned):>9,} chars  «{title}»")
        saved += 1

    print(
        f"\n{magazine.upper()}: {saved} saved · {skipped} skipped · {errors} errors"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Dragon/Dungeon Magazine text from Internet Archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--magazine",
        choices=["dragon", "dungeon", "both"],
        default="both",
        help="Which magazine to scrape",
    )
    parser.add_argument(
        "--out",
        default="data/corpus/saga",
        help="Output directory for .txt files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of issues to attempt per magazine type",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between requests (be a good citizen)",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=2000,
        help="Minimum cleaned character count to keep a file (filters covers/index pages)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")

    magazines = ["dragon", "dungeon"] if args.magazine == "both" else [args.magazine]
    for mag in magazines:
        scrape_magazine(
            magazine=mag,
            out_dir=out_dir,
            limit=args.limit,
            delay=args.delay,
            min_chars=args.min_chars,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
