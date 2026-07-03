"""Second expansion of public-domain Gutenberg texts — supplement to
scrape_gutenberg.py and scrape_gutenberg_extended.py.

Every book ID here is cross-checked against the 103 IDs already present in
data/corpus/saga/ (gutenberg_*.txt) as of this writing, so there should be no
collisions with the two earlier curated scripts. IDs were compiled from
training-data knowledge, not verified against Gutenberg at write time — some
may 404, point to a different edition than expected, or resolve to the wrong
book entirely. That's an accepted, low-cost failure mode already established
by this project (3 known failures in the earlier scripts): the downloader
skips and reports anything that doesn't resolve rather than guessing further.
Recommend running once, reviewing the [fail] lines, and pruning/replacing
bad entries in a follow-up commit rather than assuming this list is perfect
as written.

Usage
-----
    python scripts/scrape_gutenberg_expansion2.py
    python scripts/scrape_gutenberg_expansion2.py --output data/corpus/saga/

Do not reduce --delay below 1.0 to respect Gutenberg's rate limits.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
from pathlib import Path

import requests

_TEXT_URLS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

BOOKS = [
    # Lewis Carroll — nonsense fantasy, high-confidence classics
    (11,    "Alice's Adventures in Wonderland",         "Lewis Carroll"),
    (12,    "Through the Looking-Glass",                "Lewis Carroll"),

    # Fable/folklore staples
    (21,    "Aesop's Fables",                            "Aesop"),
    (5407,  "Celtic Fairy Tales",                        "Joseph Jacobs"),
    (7439,  "English Fairy Tales",                       "Joseph Jacobs"),

    # L. Frank Baum — American fairy-tale fantasy
    (55,    "The Wonderful Wizard of Oz",                "L. Frank Baum"),

    # Medieval / chivalric romance, distinct from Le Morte d'Arthur already used
    (82,    "Ivanhoe",                                   "Walter Scott"),
    (10148, "The Merry Adventures of Robin Hood",        "Howard Pyle"),

    # Swashbuckling / adventure classics — vocabulary diversity, matching the
    # precedent set by Treasure Island / Scarlet Letter / Little Women in the
    # earlier curated lists
    (829,   "Gulliver's Travels",                        "Jonathan Swift"),
    (86,    "A Connecticut Yankee in King Arthur's Court","Mark Twain"),
    (1837,  "The Prince and the Pauper",                 "Mark Twain"),
    (521,   "Robinson Crusoe",                           "Daniel Defoe"),
    (1257,  "The Three Musketeers",                       "Alexandre Dumas"),
    (1184,  "The Count of Monte Cristo",                  "Alexandre Dumas"),
    (996,   "Don Quixote",                                "Miguel de Cervantes"),
    (60,    "The Scarlet Pimpernel",                       "Baroness Orczy"),
    (95,    "The Prisoner of Zenda",                       "Anthony Hope"),
    (46,    "A Christmas Carol",                           "Charles Dickens"),

    # H. Rider Haggard — lost-world adventure fantasy, distinct from Burroughs
    (2166,  "King Solomon's Mines",                       "H. Rider Haggard"),
    (3155,  "She",                                        "H. Rider Haggard"),
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


def _download(book_id: int) -> str | None:
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


def scrape(output_dir: str, delay: float) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = skipped = failed = 0
    for book_id, title, author in BOOKS:
        safe = re.sub(r"[^\w\s-]", "", title)[:70].strip()
        safe = re.sub(r"\s+", "_", safe)
        dest = out / f"gutenberg_{book_id}_{safe}.txt"

        if dest.exists():
            skipped += 1
            print(f"  [skip] {title} (already exists)")
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

        header = f"# {title}\nAuthor: {author}\nSource: Project Gutenberg #{book_id}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        written += 1
        print(f"  [{written}] {title} -- {author}")
        time.sleep(delay)

    print(f"\nDone. {written} new, {skipped} skipped, {failed} failed -> {out}")
    print("Run the Preprocess tab to tokenize the new files.")
    if failed:
        print(
            f"\n{failed} book(s) failed to resolve. IDs in this script were "
            "compiled from memory, not verified beforehand -- review the "
            "[fail] lines above and consider pruning or replacing those "
            "entries in a follow-up commit."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download second-expansion public-domain texts from Project Gutenberg"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between downloads (default: 2.0 -- do not go below 1.0)",
    )
    args = parser.parse_args()
    scrape(args.output, args.delay)
