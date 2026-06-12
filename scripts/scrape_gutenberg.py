"""Scrape public-domain texts from Project Gutenberg via the Gutendex API.

Fetches books matching fantasy/mythology/folklore subjects and writes one
.txt file per book into the output directory.  Uses gutendex.com as a
JSON search layer on top of the Gutenberg catalogue.

Usage
-----
    python scripts/scrape_gutenberg.py
    python scripts/scrape_gutenberg.py --output data/corpus/saga/ --max-books 200

The script respects Gutenberg's robots.txt by sleeping between downloads.
Do not reduce --delay below 1.0 to avoid being banned.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
from pathlib import Path

import requests

GUTENDEX_URL = "https://gutendex.com/books"

# Subject queries — each produces a separate search pass.
# Gutendex matches against Gutenberg subject headings.
SUBJECTS = [
    "mythology",
    "folklore",
    "legends",
    "fairy tales",
    "Norse mythology",
    "Arthurian romances",
    "magic",
    "medieval",
    "demonology",
    "witchcraft",
    "alchemy",
    "chivalry",
    "epic poetry",
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)


# ---------------------------------------------------------------------------
# Gutendex helpers
# ---------------------------------------------------------------------------

def _search_books(subject: str, language: str = "en") -> list[dict]:
    """Return all book metadata records for a subject query."""
    books = []
    url = GUTENDEX_URL
    params = {"topic": subject, "languages": language}
    while url:
        try:
            resp = SESSION.get(url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  Search error ({subject}): {exc}")
            break
        data = resp.json()
        books.extend(data.get("results", []))
        url = data.get("next")
        params = {}   # next URL already contains all params
        time.sleep(0.5)
    return books


def _best_text_url(book: dict) -> str | None:
    """Pick the best plain-text download URL from a book's format map."""
    formats = book.get("formats", {})
    # Prefer UTF-8 plain text, fall back to ASCII plain text
    for mime in ("text/plain; charset=utf-8", "text/plain; charset=us-ascii", "text/plain"):
        if mime in formats:
            return formats[mime]
    # Last resort: any key containing 'text/plain'
    for mime, url in formats.items():
        if mime.startswith("text/plain"):
            return url
    return None


def _download_text(url: str) -> str:
    """Download and return the raw text of a Gutenberg book."""
    resp = SESSION.get(url, timeout=60)
    resp.raise_for_status()
    # Gutenberg serves latin-1 sometimes; decode carefully
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("latin-1")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Gutenberg books start with a header ending at "*** START OF" and end at
# "*** END OF".  Strip both to keep only the literary content.
_START_RE = re.compile(
    r"\*{3}\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*{3}",
    re.IGNORECASE,
)
_END_RE = re.compile(
    r"\*{3}\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Gutenberg header and footer license blocks."""
    m = _START_RE.search(text)
    if m:
        text = text[m.end():]
    m = _END_RE.search(text)
    if m:
        text = text[: m.start()]
    return text.strip()


def _clean(text: str) -> str:
    text = _strip_gutenberg_boilerplate(text)
    # Collapse excessive blank lines
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(output_dir: str, delay: float, max_books: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen_ids: set[int] = set()   # deduplicate across subject passes
    total_written = 0

    for subject in SUBJECTS:
        if max_books and total_written >= max_books:
            break
        print(f"\nSubject: {subject}")
        books = _search_books(subject)
        print(f"  Found {len(books)} books.")

        for book in books:
            if max_books and total_written >= max_books:
                break

            book_id = book.get("id")
            if book_id in seen_ids:
                continue
            seen_ids.add(book_id)

            title = book.get("title", f"book_{book_id}").strip()
            authors = ", ".join(
                a.get("name", "") for a in book.get("authors", [])
            )

            url = _best_text_url(book)
            if not url:
                print(f"  No plain-text URL for: {title}")
                continue

            try:
                raw = _download_text(url)
                text = _clean(raw)
                if len(text) < 500:   # skip tiny/empty results
                    continue
            except Exception as exc:
                print(f"  Failed to download '{title}': {exc}")
                continue

            # Sanitise title for use as a filename
            safe_title = re.sub(r"[^\w\s-]", "", title)[:80].strip()
            safe_title = re.sub(r"\s+", "_", safe_title)
            filename = f"gutenberg_{book_id}_{safe_title}.txt"

            header = f"# {title}\nAuthor: {authors}\nSource: Project Gutenberg #{book_id}\n\n"
            (out / filename).write_text(header + text, encoding="utf-8")
            total_written += 1
            print(f"  [{total_written}] {title} ({authors}) -> {filename}")
            time.sleep(delay)

    print(f"\nDone. {total_written} books written to {out}")
    print("Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape public-domain fantasy/mythology texts from Project Gutenberg"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory for .txt files (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between book downloads (default: 2.0 -- do not go below 1.0)",
    )
    parser.add_argument(
        "--max-books", type=int, default=0,
        help="Max total books to download (default: 0 = no limit)",
    )
    args = parser.parse_args()
    scrape(args.output, args.delay, args.max_books)
