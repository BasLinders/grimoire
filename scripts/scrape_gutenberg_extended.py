"""Extended Project Gutenberg downloader — supplement to scrape_gutenberg.py.

Downloads additional public-domain texts not in the original scraper:
- George MacDonald fantasy novels
- William Morris medieval romances
- Edgar Rice Burroughs planetary romances
- Andrew Lang fairy books (colour series continuation)
- Mathematical/scientific texts for the data-science side of the corpus

Usage
-----
    python scripts/scrape_gutenberg_extended.py
    python scripts/scrape_gutenberg_extended.py --output data/corpus/saga/

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
    # George MacDonald — Victorian fantasy, rich symbolic worldbuilding
    (325,   "Phantastes",                               "George MacDonald"),
    (1085,  "Lilith",                                   "George MacDonald"),
    (947,   "The Princess and the Goblin",              "George MacDonald"),
    (948,   "The Princess and Curdie",                  "George MacDonald"),
    (4791,  "At the Back of the North Wind",            "George MacDonald"),
    (9873,  "The Light Princess and Other Fairy Tales", "George MacDonald"),

    # William Morris — medieval prose romances, invented archaic diction
    (7143,  "The Wood Beyond the World",                "William Morris"),
    (15646, "The Well at the World's End Vol 1",        "William Morris"),
    (15679, "The Well at the World's End Vol 2",        "William Morris"),
    (4712,  "The Story of the Glittering Plain",        "William Morris"),
    (4325,  "The Water of the Wondrous Isles",          "William Morris"),
    (4324,  "The Sundering Flood",                      "William Morris"),

    # Edgar Rice Burroughs — planetary romance / sword-and-planet
    (62,    "A Princess of Mars",                       "Edgar Rice Burroughs"),
    (64,    "The Gods of Mars",                         "Edgar Rice Burroughs"),
    (68,    "The Warlord of Mars",                      "Edgar Rice Burroughs"),
    (1153,  "Thuvia Maid of Mars",                      "Edgar Rice Burroughs"),
    (72,    "Tarzan of the Apes",                       "Edgar Rice Burroughs"),
    (81,    "The Return of Tarzan",                     "Edgar Rice Burroughs"),

    # Andrew Lang — Fairy Books (colour series, beyond the four in original)
    (7830,  "The Orange Fairy Book",                    "Andrew Lang"),
    (7783,  "The Olive Fairy Book",                     "Andrew Lang"),
    (8188,  "The Lilac Fairy Book",                     "Andrew Lang"),
    (27227, "The Crimson Fairy Book",                   "Andrew Lang"),
    (8290,  "The Brown Fairy Book",                     "Andrew Lang"),
    (8107,  "The Violet Fairy Book",                    "Andrew Lang"),
    (7749,  "The Grey Fairy Book",                      "Andrew Lang"),
    (544,   "The Book of Romance",                      "Andrew Lang"),

    # Lord Dunsany — additional fantasy short fiction
    (7972,  "Fifty-One Tales",                          "Lord Dunsany"),
    (8725,  "The Sword of Welleran",                    "Lord Dunsany"),
    (8168,  "A Dreamer's Tales",                        "Lord Dunsany"),

    # Celtic mythology and folklore
    (14272, "Myths and Folk Tales of the Russians, Western Slavs, and Teutons",
            "Jeremiah Curtin"),
    (10329, "Heroic Romances of Ireland Vol 1",         "A.H. Leahy"),
    (10462, "Heroic Romances of Ireland Vol 2",         "A.H. Leahy"),
    (4486,  "The Mabinogion",                           "Lady Charlotte Guest"),
    (21451, "Irish Fairy Tales",                        "James Stephens"),
    (13993, "Myths and Legends of the Celtic Race",     "T.W. Rolleston"),

    # Additional mythology
    (7154,  "The Yellow Fairy Book",                    "Andrew Lang"),
    (17034, "Hawaiian Folk Tales",                      "Thomas G. Thrum"),
    (3972,  "The Mahabharata of Krishna-Dwaipayana Vyasa Book 1",
            "Kisari Mohan Ganguli"),
    (7864,  "The Ramayana",                             "M.N. Dutt"),

    # Mathematics and science — data-science side of corpus
    (28233, "A Course of Pure Mathematics",             "G.H. Hardy"),
    (33320, "Introduction to Mathematical Philosophy",  "Bertrand Russell"),
    (9108,  "The Logic of Chance",                      "John Venn"),
    (38986, "Symbolic Logic",                           "Lewis Carroll"),
    (35497, "Mathematical Puzzles and Diversions",      "Henry Dudeney"),
    (4763,  "Flatland: A Romance of Many Dimensions",   "Edwin Abbott"),
    (21076, "An Introduction to Mathematics",           "Alfred North Whitehead"),
    (20693, "The Algebra of Logic",                     "Louis Couturat"),

    # H.G. Wells — science fiction with speculative/world-building vocabulary
    (36,    "The War of the Worlds",                    "H.G. Wells"),
    (718,   "The Time Machine",                         "H.G. Wells"),
    (159,   "The Island of Doctor Moreau",              "H.G. Wells"),
    (5230,  "The First Men in the Moon",                "H.G. Wells"),
    (13084, "The Invisible Man",                        "H.G. Wells"),

    # Jules Verne — adventure and speculative science
    (164,   "Twenty Thousand Leagues under the Sea",    "Jules Verne"),
    (103,   "Around the World in Eighty Days",          "Jules Verne"),
    (1268,  "Journey to the Centre of the Earth",       "Jules Verne"),
    (83,    "The Mysterious Island",                    "Jules Verne"),

    # Mary Shelley / Gothic / Horror — monster / undead vocabulary
    (84,    "Frankenstein",                             "Mary Shelley"),
    (345,   "Dracula",                                  "Bram Stoker"),
    (174,   "The Picture of Dorian Gray",               "Oscar Wilde"),

    # E.R. Eddison — high fantasy precursor
    (39058, "The Worm Ouroboros",                       "E.R. Eddison"),

    # Louisa May Alcott / classic American — sentence variety
    (514,   "Little Women",                             "Louisa May Alcott"),

    # David Hume — philosophy / logic / reasoning vocabulary
    (9662,  "An Enquiry Concerning Human Understanding", "David Hume"),
    (4705,  "A Treatise of Human Nature",               "David Hume"),

    # John Stuart Mill — logic / reasoning
    (1684,  "A System of Logic",                        "John Stuart Mill"),
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
            print(f"  [skip] {title}")
            continue

        raw = _download(book_id)
        if raw is None:
            failed += 1
            print(f"  [fail] {title} (#{book_id}) — no URL worked")
            continue

        text = _clean(raw)
        if len(text) < 500:
            failed += 1
            print(f"  [skip] {title} (#{book_id}) — too short after cleaning")
            continue

        header = f"# {title}\nAuthor: {author}\nSource: Project Gutenberg #{book_id}\n\n"
        dest.write_text(header + text, encoding="utf-8")
        written += 1
        print(f"  [{written}] {title} — {author}")
        time.sleep(delay)

    print(f"\nDone. {written} new, {skipped} skipped, {failed} failed → {out}")
    print("Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download extended public-domain texts from Project Gutenberg"
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
