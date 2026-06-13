"""Download Wikipedia articles via the MediaWiki action API.

Recursively traverses subcategories from a set of seed categories and writes
one .txt file per article to the output directory.  Two topic groups are
covered: mathematics/data-science and fantasy/mythology.

Usage
-----
    python scripts/scrape_wikipedia.py
    python scripts/scrape_wikipedia.py --output data/corpus/saga/ --depth 2
    python scripts/scrape_wikipedia.py --group math --max-articles 2000

The script is idempotent: already-downloaded files are skipped.  Respect
Wikipedia's rate limits — do not set --delay below 0.1.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
from pathlib import Path

import requests

API_URL = "https://en.wikipedia.org/w/api.php"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)

_MAX_RETRIES = 6
_RETRY_BASE  = 2.0   # seconds; doubled on each attempt


def _get(params: dict, delay: float, timeout: int = 60) -> requests.Response:
    """GET with exponential-backoff retry on 429 / 5xx responses."""
    wait = _RETRY_BASE
    for attempt in range(_MAX_RETRIES):
        try:
            resp = SESSION.get(API_URL, params=params, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(wait)
            wait *= 2
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            retry_after = resp.headers.get("Retry-After")
            sleep_for = float(retry_after) if retry_after else wait
            time.sleep(max(sleep_for, wait))
            wait *= 2
            continue

        resp.raise_for_status()
        time.sleep(delay)
        return resp

    resp.raise_for_status()
    return resp

# ---------------------------------------------------------------------------
# Seed categories
# ---------------------------------------------------------------------------

MATH_CATEGORIES = [
    "Statistics",
    "Probability theory",
    "Machine learning",
    "Linear algebra",
    "Calculus",
    "Information theory",
    "Bayesian statistics",
    "Mathematical logic",
    "Graph theory",
    "Combinatorics",
    "Number theory",
    "Numerical analysis",
    "Optimization (mathematics)",
    "Data mining",
    "Artificial intelligence",
    "Neural networks",
    "Natural language processing",
    "Time series",
    "Stochastic processes",
    "Mathematical statistics",
]

FANTASY_CATEGORIES = [
    "Greek mythology",
    "Norse mythology",
    "Celtic mythology",
    "Arthurian legend",
    "Folklore",
    "Fairy tales",
    "Demonology",
    "Angels",
    "Fantasy literature",
    "Epic poetry",
    "Mythological creatures",
    "Dragons in mythology",
    "Elves",
    "Wizards",
    "Magic (supernatural)",
    "Necromancy",
    "Alchemy",
    "Divination",
    "Curses in mythology",
    "Legendary weapons",
    "Underworld",
    "Heroes in Greek mythology",
    "Arabian mythology",
    "Mesopotamian mythology",
    "Egyptian mythology",
]

ALL_GROUPS = {
    "math":    MATH_CATEGORIES,
    "fantasy": FANTASY_CATEGORIES,
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_subcategories(category: str, delay: float) -> list[str]:
    """Return immediate subcategory names within a category."""
    subs = []
    params = {
        "action":  "query",
        "list":    "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype":  "subcat",
        "cmlimit": "500",
        "format":  "json",
    }
    while True:
        data = _get(params, delay=delay, timeout=30).json()
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            subs.append(m["title"].removeprefix("Category:"))
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
    return subs


def _get_category_page_titles(category: str, delay: float) -> list[str]:
    """Return page titles (not subcategories) directly in a category."""
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
        data = _get(params, delay=delay, timeout=30).json()
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
    return titles


def _collect_titles(seeds: list[str], depth: int, delay: float) -> list[str]:
    """BFS over category tree up to *depth* levels deep; return unique page titles."""
    seen_cats: set[str] = set()
    seen_titles: set[str] = set()
    all_titles: list[str] = []

    frontier = list(seeds)
    for level in range(depth + 1):
        if not frontier:
            break
        next_frontier: list[str] = []
        for cat in frontier:
            if cat in seen_cats:
                continue
            seen_cats.add(cat)
            try:
                page_titles = _get_category_page_titles(cat, delay)
                new_pages = 0
                for t in page_titles:
                    if t not in seen_titles:
                        seen_titles.add(t)
                        all_titles.append(t)
                        new_pages += 1
                subs = []
                if level < depth:
                    subs = _get_subcategories(cat, delay)
                    next_frontier.extend(s for s in subs if s not in seen_cats)
                print(f"  [depth {level}] {cat}: {new_pages} articles, {len(subs)} subcategories "
                      f"(total so far: {len(all_titles)})")
                time.sleep(delay)
            except Exception as exc:
                print(f"  [warn] category '{cat}': {exc}")
        frontier = next_frontier

    return all_titles


def _fetch_extracts(titles: list[str], delay: float) -> dict[str, str]:
    """Fetch plain-text extracts for a batch of titles; return {title: text}."""
    results: dict[str, str] = {}
    params = {
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     "1",
        "exsectionformat": "plain",
        "titles":          "|".join(titles),
        "format":          "json",
    }
    try:
        data = _get(params, delay=delay, timeout=60).json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            title = page.get("title", "")
            extract = page.get("extract", "")
            if extract and "missing" not in page:
                results[title] = extract
    except Exception as exc:
        print(f"  [warn] batch fetch failed: {exc}")
    return results


def _clean(text: str) -> str:
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _safe_filename(title: str, prefix: str) -> str:
    safe = re.sub(r"[^\w\s-]", "", title)[:60].strip()
    safe = re.sub(r"\s+", "_", safe)
    return f"wp_{prefix}_{safe}.txt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(
    output_dir: str,
    groups: list[str],
    depth: int,
    delay: float,
    max_articles: int,
    batch_size: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for group in groups:
        seeds = ALL_GROUPS[group]
        print(f"\n=== Group: {group} ({len(seeds)} seed categories, depth={depth}) ===")

        print("  Collecting article titles...")
        titles = _collect_titles(seeds, depth=depth, delay=delay)
        print(f"  Found {len(titles)} unique articles.")

        if max_articles and len(titles) > max_articles:
            titles = titles[:max_articles]
            print(f"  Capped at {max_articles}.")

        written = skipped = failed = 0
        for start in range(0, len(titles), batch_size):
            batch = titles[start: start + batch_size]

            # Skip titles whose files already exist
            needed = [t for t in batch if not (out / _safe_filename(t, group)).exists()]
            skipped += len(batch) - len(needed)
            if not needed:
                continue

            extracts = _fetch_extracts(needed, delay=delay)
            for title in needed:
                text = extracts.get(title, "")
                text = _clean(text)
                if len(text) < 200:
                    failed += 1
                    continue
                dest = out / _safe_filename(title, group)
                header = f"# {title}\nSource: Wikipedia\n\n"
                dest.write_text(header + text, encoding="utf-8")
                written += 1

            done = start + len(batch)
            if done % 200 == 0 or done >= len(titles):
                print(f"  {done}/{len(titles)} processed — {written} written, "
                      f"{skipped} skipped, {failed} failed")

        print(f"  Done: {written} new articles written.")

    print("\nFinished. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wikipedia articles for the Grimoire corpus"
    )
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--group", choices=["math", "fantasy", "all"], default="all",
        help="Topic group to fetch (default: all)",
    )
    parser.add_argument(
        "--depth", type=int, default=2,
        help="Subcategory recursion depth (default: 2)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds between requests (default: 0.5 — do not go below 0.2)",
    )
    parser.add_argument(
        "--max-articles", type=int, default=0,
        help="Max articles per group (default: 0 = no limit)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10,
        help="Titles per API batch request (default: 10; max 50 for Wikipedia)",
    )
    args = parser.parse_args()

    groups = ["math", "fantasy"] if args.group == "all" else [args.group]
    scrape(
        output_dir=args.output,
        groups=groups,
        depth=args.depth,
        delay=args.delay,
        max_articles=args.max_articles,
        batch_size=min(args.batch_size, 50),
    )
