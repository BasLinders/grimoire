"""Download arXiv paper abstracts via the public API.

arXiv hosts over 2 million papers across mathematics, physics, computer
science, statistics, and related fields.  Abstracts average 200-300 words
each — at 2M papers that is 400-600 million tokens from abstracts alone,
making this one of the highest-yield text sources accessible without
bulk downloads.  The API returns structured metadata including title,
authors, abstract, and subject categories.

Two groups are fetched:

  math       — all mathematics subject classes (math.*, stat.*)
  datascience — machine learning, AI, NLP, information retrieval (cs.LG,
                cs.AI, cs.CL, cs.IR, cs.CV, stat.ML)

Each paper is written as:

    # Title
    Authors: ...
    Categories: ...
    Published: YYYY-MM-DD

    Abstract text here.

Usage
-----
    python scripts/scrape_arxiv.py
    python scripts/scrape_arxiv.py --group math --max-papers 50000
    python scripts/scrape_arxiv.py --start-year 2010 --output data/corpus/saga/

Rate limits
-----------
arXiv asks for at most 3 requests per second.  The default --delay of 0.4s
keeps well within that.  For large runs (> 10 000 papers) prefer --delay 1.0.

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

_ATOM_NS   = "http://www.w3.org/2005/Atom"
_ARXIV_NS  = "http://arxiv.org/schemas/atom"
_API_BASE  = "https://export.arxiv.org/api/query"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = (
    "GrimoireCorpusScraper/1.0 (educational corpus builder; "
    "contact: local research project)"
)

_MAX_RETRIES = 6
_RETRY_BASE  = 2.0

# Subject-class queries accepted by the arXiv search API.
# Each string is passed as search_query=cat:<value>.
MATH_QUERIES = [
    "math.ST",   # Statistics Theory
    "math.PR",   # Probability
    "math.CO",   # Combinatorics
    "math.NT",   # Number Theory
    "math.LO",   # Logic
    "math.GR",   # Group Theory
    "math.NA",   # Numerical Analysis
    "math.OC",   # Optimization and Control
    "math.IT",   # Information Theory
    "stat.TH",   # Statistics Theory
    "stat.ME",   # Methodology
    "stat.CO",   # Computation
]

DATASCIENCE_QUERIES = [
    "cs.LG",     # Machine Learning
    "cs.AI",     # Artificial Intelligence
    "cs.CL",     # Computation and Language (NLP)
    "cs.IR",     # Information Retrieval
    "cs.CV",     # Computer Vision
    "cs.NE",     # Neural and Evolutionary Computing
    "stat.ML",   # Machine Learning (statistics)
]

ALL_GROUPS = {
    "math":        MATH_QUERIES,
    "datascience": DATASCIENCE_QUERIES,
}


def _get(url: str, params: dict, delay: float) -> requests.Response:
    wait = _RETRY_BASE
    for attempt in range(_MAX_RETRIES):
        try:
            resp = SESSION.get(url, params=params, timeout=60)
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


def _safe_filename(arxiv_id: str, prefix: str) -> str:
    safe = re.sub(r"[^\w.-]", "_", arxiv_id)
    return f"arxiv_{prefix}_{safe}.txt"


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


def _fetch_batch(
    query: str,
    start: int,
    max_results: int,
    delay: float,
    start_year: int | None,
) -> list[dict]:
    """Fetch one page of results from the arXiv API; return list of paper dicts."""
    search_query = f"cat:{query}"
    if start_year:
        search_query += f" AND submittedDate:[{start_year}01010000 TO 99991231235900]"

    params = {
        "search_query": search_query,
        "start":        start,
        "max_results":  max_results,
        "sortBy":       "submittedDate",
        "sortOrder":    "descending",
    }
    resp = _get(_API_BASE, params, delay=delay)
    root = ET.fromstring(resp.text)

    papers: list[dict] = []
    for entry in root.findall(f"{{{_ATOM_NS}}}entry"):
        arxiv_id_url = _text(entry.find(f"{{{_ATOM_NS}}}id"))
        arxiv_id = arxiv_id_url.split("/abs/")[-1].replace("/", "_")
        title    = re.sub(r"\s+", " ", _text(entry.find(f"{{{_ATOM_NS}}}title")))
        abstract = re.sub(r"\s+", " ", _text(entry.find(f"{{{_ATOM_NS}}}summary")))
        published = _text(entry.find(f"{{{_ATOM_NS}}}published"))[:10]
        authors  = [
            _text(a.find(f"{{{_ATOM_NS}}}name"))
            for a in entry.findall(f"{{{_ATOM_NS}}}author")
        ]
        cats = [
            t.get("term", "")
            for t in entry.findall(f"{{{_ATOM_NS}}}category")
        ]
        papers.append({
            "id":        arxiv_id,
            "title":     title,
            "abstract":  abstract,
            "published": published,
            "authors":   authors,
            "cats":      cats,
        })

    return papers


def _format_paper(p: dict) -> str:
    author_str = "; ".join(p["authors"][:6])
    if len(p["authors"]) > 6:
        author_str += f" et al. ({len(p['authors'])} authors)"
    cat_str = ", ".join(p["cats"])
    return (
        f"# {p['title']}\n"
        f"Authors: {author_str}\n"
        f"Categories: {cat_str}\n"
        f"Published: {p['published']}\n"
        f"Source: arXiv ({p['id']})\n\n"
        f"{p['abstract']}\n"
    )


def scrape(
    output_dir: str,
    groups: list[str],
    max_papers: int,
    batch_size: int,
    delay: float,
    start_year: int | None,
    min_abstract_chars: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for group in groups:
        queries = ALL_GROUPS[group]
        print(f"\n=== arXiv group: {group} ({len(queries)} subject classes) ===")
        group_written = 0

        for query in queries:
            if max_papers and group_written >= max_papers:
                break
            print(f"  Fetching {query}...")
            offset = 0
            query_written = query_skipped = query_failed = 0

            while True:
                if max_papers and group_written >= max_papers:
                    break

                remaining = max_papers - group_written if max_papers else batch_size
                fetch_n = min(batch_size, remaining) if max_papers else batch_size
                try:
                    papers = _fetch_batch(
                        query=query,
                        start=offset,
                        max_results=fetch_n,
                        delay=delay,
                        start_year=start_year,
                    )
                except Exception as exc:
                    print(f"  [warn] {query} offset {offset}: {exc}")
                    break

                if not papers:
                    break

                for p in papers:
                    fname = _safe_filename(p["id"], group)
                    dest = out / fname
                    if dest.exists():
                        query_skipped += 1
                        continue
                    if len(p["abstract"]) < min_abstract_chars:
                        query_failed += 1
                        continue
                    dest.write_text(_format_paper(p), encoding="utf-8")
                    query_written += 1
                    group_written += 1

                offset += len(papers)
                print(f"    {query}: {offset} fetched, {query_written} written, "
                      f"{query_skipped} skipped, {query_failed} too short")

                if len(papers) < fetch_n:
                    break  # exhausted this subject class

            print(f"  {query} done: {query_written} written.")

        print(f"  Group {group} total: {group_written} papers written.")

    print("\nFinished. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download arXiv abstracts for the Grimoire corpus"
    )
    parser.add_argument("--output", default="data/corpus/saga/")
    parser.add_argument(
        "--group", choices=["math", "datascience", "all"], default="all",
        help="Topic group (default: all)"
    )
    parser.add_argument(
        "--max-papers", type=int, default=0,
        help="Max papers per group (0 = no limit; each subject class fetched fully)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Papers per API call (default 100, max 2000)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.4,
        help="Seconds between API calls (default 0.4; use 1.0 for large runs)"
    )
    parser.add_argument(
        "--start-year", type=int, default=None,
        help="Only fetch papers published from this year onward (e.g. 2010)"
    )
    parser.add_argument(
        "--min-abstract-chars", type=int, default=100,
        help="Drop papers with abstracts shorter than this (default 100)"
    )
    args = parser.parse_args()

    groups = list(ALL_GROUPS.keys()) if args.group == "all" else [args.group]
    scrape(
        output_dir=args.output,
        groups=groups,
        max_papers=args.max_papers,
        batch_size=min(args.batch_size, 2000),
        delay=args.delay,
        start_year=args.start_year,
        min_abstract_chars=args.min_abstract_chars,
    )
