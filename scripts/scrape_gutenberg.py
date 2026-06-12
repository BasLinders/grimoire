"""Download public-domain fantasy/mythology texts from Project Gutenberg.

Uses a curated list of book IDs rather than the Gutendex search API, which
proved unreliable.  Books are downloaded directly from gutenberg.org as
plain text and stripped of their header/footer boilerplate.

Usage
-----
    python scripts/scrape_gutenberg.py
    python scripts/scrape_gutenberg.py --output data/corpus/saga/

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

# Direct plain-text download base URLs (tried in order).
_TEXT_URLS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/files/{id}/{id}.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

# Curated list of (book_id, title, author) tuples.
# All are public domain; selected for fantasy/mythology/medieval vocabulary.
BOOKS = [
    # Norse / Germanic mythology
    (4785,  "Prose Edda",                              "Snorri Sturluson"),
    (23265, "Poetic Edda",                             "Anonymous"),
    (16328, "Beowulf",                                 "Anonymous"),
    (557,   "Nibelungenlied",                          "Anonymous"),
    # Arthurian / Medieval romance
    (1251,  "Le Morte d'Arthur Vol 1",                 "Thomas Malory"),
    (1252,  "Le Morte d'Arthur Vol 2",                 "Thomas Malory"),
    (831,   "The Once and Future King (excerpt)",      "T.H. White"),
    (9934,  "Parzival",                                "Wolfram von Eschenbach"),
    (583,   "The Canterbury Tales",                    "Geoffrey Chaucer"),
    # Greek / Roman mythology
    (6130,  "The Iliad",                               "Homer"),
    (1727,  "The Odyssey",                             "Homer"),
    (228,   "The Aeneid",                              "Virgil"),
    (348,   "Metamorphoses",                           "Ovid"),
    (22381, "Theogony and Works and Days",             "Hesiod"),
    (2680,  "Myths and Legends of Ancient Greece and Rome", "E.M. Berens"),
    # Arabian Nights / Eastern mythology
    (128,   "One Thousand and One Nights Vol 1",       "Anonymous"),
    (7517,  "One Thousand and One Nights Vol 4",       "Anonymous"),
    # Fairy tales / Folklore
    (2591,  "Grimms' Fairy Tales",                     "Brothers Grimm"),
    (5314,  "Household Tales by Brothers Grimm",       "Brothers Grimm"),
    (1597,  "Fairy Tales by Hans Christian Andersen",  "Hans Christian Andersen"),
    (6997,  "More Fairy Tales by Andersen",            "Hans Christian Andersen"),
    (7452,  "The Blue Fairy Book",                     "Andrew Lang"),
    (7153,  "The Red Fairy Book",                      "Andrew Lang"),
    (7111,  "The Green Fairy Book",                    "Andrew Lang"),
    (7154,  "The Yellow Fairy Book",                   "Andrew Lang"),
    (558,   "The Arabian Nights Entertainments",       "Anonymous"),
    # Epic poetry / heroic literature
    (20,    "The Divine Comedy (Inferno)",             "Dante Alighieri"),
    (8789,  "Paradise Lost",                           "John Milton"),
    (3071,  "The Faerie Queene Book 1",                "Edmund Spenser"),
    # Mythology compilations
    (14637, "Bulfinch's Mythology",                    "Thomas Bulfinch"),
    (22696, "The Golden Bough (abridged)",             "James George Frazer"),
    (7882,  "Hero Tales and Legends of the Rhine",     "Lewis Spence"),
    (9267,  "Myths of the Norsemen",                   "H.A. Guerber"),
    (9880,  "The Mythology of the British Islands",    "Charles Squire"),
    # Demonology / occult lore (flavour for D&D)
    (46849, "The Lesser Key of Solomon",               "Anonymous"),
    (2021,  "The Witch-cult in Western Europe",        "Margaret Murray"),
    # Fantasy fiction (public domain)
    (16,    "Peter Pan",                               "J.M. Barrie"),
    (25525, "The King of Elfland's Daughter",          "Lord Dunsany"),
    (7132,  "The Book of Wonder",                      "Lord Dunsany"),
    (7133,  "Time and the Gods",                       "Lord Dunsany"),
    (8432,  "The Gods of Pegana",                      "Lord Dunsany"),
    (30,    "The Scarlet Letter",                      "Nathaniel Hawthorne"),
    (113,   "Treasure Island",                         "Robert Louis Stevenson"),
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
    """Try each URL pattern and return raw text, or None on failure."""
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

    written = 0
    for book_id, title, author in BOOKS:
        safe = re.sub(r"[^\w\s-]", "", title)[:70].strip()
        safe = re.sub(r"\s+", "_", safe)
        dest = out / f"gutenberg_{book_id}_{safe}.txt"

        if dest.exists():
            print(f"  [skip] {title} (already exists)")
            continue

        raw = _download(book_id)
        if raw is None:
            print(f"  [fail] {title} (#{book_id}) — no URL worked")
            continue

        text = _clean(raw)
        if len(text) < 500:
            print(f"  [skip] {title} (#{book_id}) — too short after cleaning")
            continue

        header = f"# {title}\nAuthor: {author}\nSource: Project Gutenberg #{book_id}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        written += 1
        print(f"  [{written}] {title} — {author}")
        time.sleep(delay)

    print(f"\nDone. {written} books written to {out}")
    print("Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download public-domain fantasy/mythology texts from Project Gutenberg"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds between downloads (default: 2.0 — do not go below 1.0)",
    )
    args = parser.parse_args()
    scrape(args.output, args.delay)
