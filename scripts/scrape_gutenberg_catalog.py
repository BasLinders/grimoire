"""Discover and download public-domain fantasy/mythology/folklore texts from
Project Gutenberg using its official bulk CSV catalog, instead of a
hand-curated list of book IDs.

Why this exists
----------------
scrape_gutenberg.py, scrape_gutenberg_extended.py, and
scrape_gutenberg_expansion2.py all use (book_id, title, author) tuples
compiled from memory. That works for well-known classics but doesn't scale,
and a misremembered ID either 404s or silently downloads the wrong book.

An earlier attempt to fix this by crawling gutenberg.org's search-results
pages was abandoned after live-testing found the pages themselves carry an
explicit notice: "DON'T USE THIS PAGE FOR SCRAPING. Seriously. You'll only
get your IP blocked." That page recommends Gutenberg's own bulk catalog feed
instead -- which is exactly what this script uses.

This script downloads (and caches) https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
once -- a single ~21MB file listing every Gutenberg text with its real,
exact Title / Authors / Language / Subjects / Bookshelves -- filters it
locally by subject keywords and language, excludes anything already present
in the corpus, and downloads the matching plain-text editions through the
same _download / _clean pipeline as the other scrape_gutenberg_*.py scripts.
No HTML scraping, no guessed IDs, no repeated search requests.

As of this writing, the default keyword set matches ~3,400 English-language
texts tagged fantasy fiction / fairy tales / mythology / folklore / legends /
fables / epic literature / fantasy literature / ghost stories / supernatural
in Gutenberg's own subject and bookshelf metadata -- far more than any
hand-curated list could realistically cover, and with exact rather than
guessed metadata.

Usage
-----
    python scripts/scrape_gutenberg_catalog.py
    python scripts/scrape_gutenberg_catalog.py --max-books 300
    python scripts/scrape_gutenberg_catalog.py --keywords "mythology,folklore"
    python scripts/scrape_gutenberg_catalog.py --refresh-catalog

Do not reduce --delay below 1.0 to respect Gutenberg's rate limits. The
catalog CSV itself is only ever downloaded once per --refresh-catalog.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import csv
import io
import re
import time
from pathlib import Path

import requests

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
CATALOG_CACHE = Path("data/catalogs/pg_catalog.csv")

_TEXT_URLS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

DEFAULT_KEYWORDS = [
    "fantasy fiction",
    "fairy tales",
    "mythology",
    "folklore",
    "legends",
    "fables",
    "epic literature",
    "fantasy literature",
    "ghost stories",
    "supernatural",
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)

_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*{3}",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*",
    re.IGNORECASE | re.DOTALL,
)


def _fetch_catalog(refresh: bool) -> str:
    if CATALOG_CACHE.exists() and not refresh:
        print(f"Using cached catalog at {CATALOG_CACHE}")
        return CATALOG_CACHE.read_text(encoding="utf-8")

    print(f"Downloading Gutenberg catalog from {CATALOG_URL} ...")
    resp = SESSION.get(CATALOG_URL, timeout=120)
    resp.raise_for_status()
    text = resp.content.decode("utf-8")

    CATALOG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_CACHE.write_text(text, encoding="utf-8")
    print(f"Cached catalog to {CATALOG_CACHE} ({len(resp.content)} bytes)")
    return text


def _find_candidates(catalog_csv: str, keywords: list[str]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(catalog_csv))
    keywords_lower = [k.lower() for k in keywords]

    candidates = []
    for row in reader:
        if row.get("Type") != "Text" or row.get("Language") != "en":
            continue
        blob = (row.get("Subjects", "") + " " + row.get("Bookshelves", "")).lower()
        if any(k in blob for k in keywords_lower):
            candidates.append(row)
    return candidates


def _already_scraped(out: Path, book_id: str) -> bool:
    return any(out.glob(f"gutenberg_{book_id}_*.txt"))


def _download(book_id: str) -> str | None:
    for pattern in _TEXT_URLS:
        url = pattern.format(id=book_id)
        try:
            resp = SESSION.get(url, timeout=60)
            if resp.status_code == 200:
                try:
                    return resp.content.decode("utf-8")
                except UnicodeDecodeError:
                    return resp.content.decode("latin-1")
        except requests.RequestException:
            continue
    return None


def _clean(text: str) -> str:
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def scrape(
    output_dir: str,
    keywords: list[str],
    max_books: int,
    delay: float,
    refresh_catalog: bool,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog_csv = _fetch_catalog(refresh_catalog)
    candidates = _find_candidates(catalog_csv, keywords)
    print(
        f"\n{len(candidates)} candidate texts match keywords {keywords} "
        f"(English, Type=Text) in the catalog."
    )

    written = skipped = failed = 0
    for row in candidates:
        if written >= max_books:
            print(f"\nReached --max-books limit ({max_books}); stopping.")
            break

        book_id = row["Text#"]
        title = (row.get("Title") or f"Untitled Work {book_id}").splitlines()[0].strip()
        author = (row.get("Authors") or "Unknown").strip() or "Unknown"

        if _already_scraped(out, book_id):
            skipped += 1
            continue

        raw = _download(book_id)
        if raw is None:
            failed += 1
            print(f"  [fail] {title} (#{book_id}) -- no URL worked")
            continue

        text = _clean(raw)
        if len(text) < 500:
            failed += 1
            print(f"  [skip] {title} (#{book_id}) -- too short after cleaning")
            continue

        safe = re.sub(r"[^\w\s-]", "", title)[:70].strip()
        safe = re.sub(r"\s+", "_", safe)
        dest = out / f"gutenberg_{book_id}_{safe}.txt"

        header = f"# {title}\nAuthor: {author}\nSource: Project Gutenberg #{book_id}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        written += 1
        print(f"  [{written}] {title} -- {author} (#{book_id})")
        time.sleep(delay)

    print(
        f"\nDone. {written} new, {skipped} already present, {failed} failed "
        f"-> {out}"
    )
    print("Run the Preprocess tab to tokenize the new files.")
    print(
        'Then check for near-duplicates before merging: '
        'python scripts/dedup_corpus.py --corpus-dir data/corpus/ '
        '--new-glob "saga/gutenberg_*.txt"'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Discover and download Gutenberg texts by subject/keyword, using "
            "the official bulk catalog CSV instead of a hand-curated list."
        )
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--keywords", default=",".join(DEFAULT_KEYWORDS),
        help="Comma-separated substrings to match against each book's "
             "Subjects + Bookshelves fields, case-insensitive "
             f"(default: {', '.join(DEFAULT_KEYWORDS)})",
    )
    parser.add_argument(
        "--max-books", type=int, default=200,
        help="Maximum number of new books to download this run (default: 200). "
             "Candidates typically number in the thousands -- this cap keeps "
             "a single run reviewable rather than downloading everything at once.",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between text downloads (default: 2.0 -- do not go below 1.0)",
    )
    parser.add_argument(
        "--refresh-catalog", action="store_true",
        help="Re-download the catalog CSV even if a cached copy exists at "
             f"{CATALOG_CACHE}",
    )
    args = parser.parse_args()
    scrape(
        args.output,
        [k.strip() for k in args.keywords.split(",") if k.strip()],
        args.max_books,
        args.delay,
        args.refresh_catalog,
    )
