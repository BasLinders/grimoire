"""Scrape D&D Wiki (dandwiki.com) homebrew content via the MediaWiki API.

Fetches homebrew classes, spells, monsters, races, feats, and magic items
and writes one .txt file per category into the output directory.

Usage
-----
    python scripts/scrape_dndwiki.py
    python scripts/scrape_dndwiki.py --output data/corpus/saga/ --delay 0.5
    python scripts/scrape_dndwiki.py --max-articles 200

Use a delay of at least 0.5 s to avoid being rate-limited.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
from pathlib import Path

import requests

API_URL = "https://www.dandwiki.com/w/api.php"

# (category_title, output_filename)
# Using the 5e SRD and homebrew categories that have the most useful content.
CATEGORIES = [
    ("5e Classes",                  "dndwiki_classes.txt"),
    ("5e Subclasses",               "dndwiki_subclasses.txt"),
    ("5e Spells",                   "dndwiki_spells.txt"),
    ("5e Monsters",                 "dndwiki_monsters.txt"),
    ("5e Races",                    "dndwiki_races.txt"),
    ("5e Feats",                    "dndwiki_feats.txt"),
    ("5e Magic Items",              "dndwiki_magic_items.txt"),
    ("5e Backgrounds",              "dndwiki_backgrounds.txt"),
    ("5e Weapons",                  "dndwiki_weapons.txt"),
    ("5e Armor",                    "dndwiki_armor.txt"),
]

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)


def _get_category_members(category: str, delay: float) -> list[str]:
    """Return all page titles in a wiki category (paginated)."""
    titles = []
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype":  "page",
        "cmlimit": "500",
        "format":  "json",
    }
    while True:
        try:
            resp = SESSION.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  Error fetching category members: {exc}")
            break
        data = resp.json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
        time.sleep(delay)
    return titles


def _get_page_text(title: str) -> str:
    """Fetch plain text from a wiki page via raw wikitext."""
    params = {
        "action":  "query",
        "titles":  title,
        "prop":    "revisions",
        "rvprop":  "content",
        "rvslots": "main",
        "format":  "json",
    }
    resp = SESSION.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return ""
        slots = page.get("revisions", [{}])[0].get("slots", {})
        wikitext = slots.get("main", {}).get("*", "")
        return _wikitext_to_text(wikitext)
    return ""


def _wikitext_to_text(wikitext: str) -> str:
    """Strip wikitext markup to plain text."""
    # Two passes for shallow template nesting
    text = re.sub(r"\{\{[^{}]*\}\}", "", wikitext)
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    text = re.sub(r"\[\[(?:File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", text)
    text = re.sub(r"\[https?://\S+\]", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"^\s*[|!{].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"=+\s*(.+?)\s*=+", r"\1", text)
    text = re.sub(r"'{2,3}", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scrape(output_dir: str, delay: float, max_articles: int) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for category, filename in CATEGORIES:
        print(f"\nCategory: {category}")
        try:
            titles = _get_category_members(category, delay=delay)
        except Exception as exc:
            print(f"  Could not fetch category members: {exc}")
            continue

        if not titles:
            print("  No articles found, skipping.")
            continue

        if max_articles and len(titles) > max_articles:
            print(f"  Capping at {max_articles} of {len(titles)} articles.")
            titles = titles[:max_articles]
        else:
            print(f"  Found {len(titles)} articles.")

        articles = []
        for i, title in enumerate(titles, 1):
            try:
                text = _get_page_text(title)
                if len(text) > 100:
                    articles.append(f"# {title}\n\n{text}")
                if i % 50 == 0:
                    print(f"  {i}/{len(titles)} fetched...")
                time.sleep(delay)
            except Exception as exc:
                print(f"  {title}: {exc}")
                continue

        if not articles:
            print("  No usable content retrieved.")
            continue

        out_path = out / filename
        out_path.write_text("\n\n---\n\n".join(articles), encoding="utf-8")
        print(f"  {len(articles)} articles -> {out_path}")

    print("\nDone. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape D&D Wiki homebrew content to corpus .txt files"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between requests (default: 0.5)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=0,
        help="Max articles per category (default: 0 = no limit)",
    )
    args = parser.parse_args()
    scrape(args.output, args.delay, args.max_articles)
