"""Download Wikibooks pages via the MediaWiki action API.

Wikibooks hosts free, collaboratively written textbooks.  Pages tend to be
much longer than Wikipedia articles — a single chapter can run several thousand
words — making this a token-dense source.  Two topic groups are covered:

  math     — mathematics, statistics, probability, linear algebra, calculus,
              logic, combinatorics, data science, machine learning
  fantasy  — fantasy writing, mythology, folklore, world-building guides,
              role-playing game manuals, historical fiction craft

Usage
-----
    python scripts/scrape_wikibooks.py
    python scripts/scrape_wikibooks.py --output data/corpus/saga/ --depth 2
    python scripts/scrape_wikibooks.py --group math --max-pages 3000

The script is idempotent: already-downloaded files are skipped.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
from pathlib import Path

import requests

API_URL = "https://en.wikibooks.org/w/api.php"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)

_MAX_RETRIES = 6
_RETRY_BASE  = 2.0


def _get(params: dict, delay: float, timeout: int = 60) -> requests.Response:
    """GET with exponential-backoff retry on 429 / 5xx responses."""
    wait = _RETRY_BASE
    for attempt in range(_MAX_RETRIES):
        try:
            resp = SESSION.get(API_URL, params=params, timeout=timeout)
        except requests.RequestException:
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
    "Mathematics",
    "Algebra",
    "Calculus",
    "Statistics",
    "Probability",
    "Linear Algebra",
    "Discrete Mathematics",
    "Mathematical Analysis",
    "Number Theory",
    "Combinatorics",
    "Logic",
    "Numerical Methods",
    "Data Science",
    "Machine Learning",
    "Artificial Intelligence",
    "Computer Science",
    "Algorithms",
    "Information Theory",
    "Optimization",
]

FANTASY_CATEGORIES = [
    "Fiction Writing",
    "Fantasy",
    "Mythology",
    "Folklore",
    "Dungeons and Dragons",
    "Role-playing Games",
    "World Building",
    "Speculative Fiction",
    "Writing",
    "History",
    "Classical Studies",
    "Linguistics",
]

ALL_GROUPS = {
    "math":    MATH_CATEGORIES,
    "fantasy": FANTASY_CATEGORIES,
}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get_subcategories(category: str, delay: float) -> list[str]:
    subs: list[str] = []
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
        for m in data.get("query", {}).get("categorymembers", []):
            subs.append(m["title"].removeprefix("Category:"))
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
    return subs


def _get_category_page_titles(category: str, delay: float) -> list[str]:
    titles: list[str] = []
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
        for m in data.get("query", {}).get("categorymembers", []):
            titles.append(m["title"])
        cont = data.get("continue", {}).get("cmcontinue")
        if not cont:
            break
        params["cmcontinue"] = cont
    return titles


def _collect_titles(seeds: list[str], depth: int, delay: float) -> list[str]:
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
                subs: list[str] = []
                if level < depth:
                    subs = _get_subcategories(cat, delay)
                    next_frontier.extend(s for s in subs if s not in seen_cats)
                print(f"  [depth {level}] {cat}: {new_pages} pages, "
                      f"{len(subs)} subcategories (total: {len(all_titles)})")
            except Exception as exc:
                print(f"  [warn] category '{cat}': {exc}")
        frontier = next_frontier

    return all_titles


def _fetch_extracts(titles: list[str], delay: float) -> dict[str, str]:
    results: dict[str, str] = {}
    params = {
        "action":          "query",
        "prop":            "extracts",
        "explaintext":     "1",
        "exsectionformat": "plain",
        "exlimit":         "max",
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
    # Remove Wikibooks boilerplate navigation lines
    text = re.sub(r"^(Previous|Next|Back|Home|Contents|Navigation)\s*:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _safe_filename(title: str, prefix: str) -> str:
    safe = re.sub(r"[^\w\s/-]", "", title)[:60].strip()
    safe = re.sub(r"[\s/]+", "_", safe)
    return f"wb_{prefix}_{safe}.txt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(
    output_dir: str,
    groups: list[str],
    depth: int,
    delay: float,
    max_pages: int,
    batch_size: int,
    min_chars: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for group in groups:
        seeds = ALL_GROUPS[group]
        print(f"\n=== Wikibooks group: {group} ({len(seeds)} seed categories, depth={depth}) ===")

        print("  Collecting page titles...")
        titles = _collect_titles(seeds, depth=depth, delay=delay)
        print(f"  Found {len(titles)} unique pages.")

        if max_pages and len(titles) > max_pages:
            titles = titles[:max_pages]
            print(f"  Capped at {max_pages}.")

        written = skipped = failed = 0
        for start in range(0, len(titles), batch_size):
            batch = titles[start: start + batch_size]
            needed = [t for t in batch if not (out / _safe_filename(t, group)).exists()]
            skipped += len(batch) - len(needed)
            if not needed:
                continue

            extracts = _fetch_extracts(needed, delay=delay)
            for title in needed:
                text = _clean(extracts.get(title, ""))
                if len(text) < min_chars:
                    failed += 1
                    continue
                dest = out / _safe_filename(title, group)
                header = f"# {title}\nSource: Wikibooks (en.wikibooks.org)\n\n"
                dest.write_text(header + text, encoding="utf-8")
                written += 1

            done = start + len(batch)
            if done % 200 == 0 or done >= len(titles):
                print(f"  {done}/{len(titles)} processed — "
                      f"{written} written, {skipped} skipped, {failed} too short")

        print(f"  Done: {written} new pages written.")

    print("\nFinished. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download Wikibooks pages for the Grimoire corpus"
    )
    parser.add_argument("--output", default="data/corpus/saga/")
    parser.add_argument("--group", choices=["math", "fantasy", "all"], default="all")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Max pages per group (0 = no limit)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=300,
                        help="Drop pages shorter than this (default 300)")
    args = parser.parse_args()

    groups = ["math", "fantasy"] if args.group == "all" else [args.group]
    scrape(
        output_dir=args.output,
        groups=groups,
        depth=args.depth,
        delay=args.delay,
        max_pages=args.max_pages,
        batch_size=min(args.batch_size, 50),
        min_chars=args.min_chars,
    )
