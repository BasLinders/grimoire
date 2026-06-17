"""Download and parse the rpg.stackexchange.com data dump.

The Stack Exchange data dumps are published quarterly on the Internet Archive
under CC BY-SA 4.0.  This script downloads the RPG site's dump, extracts
``Posts.xml``, pairs each question with its highest-scoring answers, strips
HTML markup, and writes chunked ``.txt`` files ready for corpus ingestion.

Output format per Q&A pair
--------------------------
    # Question title
    Tags: [tag1] [tag2]
    Score: N

    <question body as plain text>

    ## Answer  (score: N)

    <answer body as plain text>

    ---

Each output file contains up to ``--chunk`` Q&A pairs (default 200).

Usage
-----
    python scripts/scrape_stackexchange_rpg.py
    python scripts/scrape_stackexchange_rpg.py --out data/corpus/saga --min-score 2
    python scripts/scrape_stackexchange_rpg.py --cache /tmp/se_dump --chunk 500

Requirements
------------
    pip install "grimoire-ai[scraper]"  # includes py7zr, requests, beautifulsoup4
"""

from __future__ import annotations

import argparse
import io
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import py7zr
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUMP_URL   = "https://archive.org/download/stackexchange/rpg.stackexchange.com.7z"
USER_AGENT = (
    "grimoire-corpus-scraper/1.0 (academic/research use; "
    "github.com/BasLinders/grimoire)"
)

# ---------------------------------------------------------------------------
# HTML → plain text
# ---------------------------------------------------------------------------

# Collapse 3+ blank lines to 2.
_BLANK_RUNS = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip HTML tags, decode entities, normalise whitespace."""
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator="\n")
    text = _BLANK_RUNS.sub("\n\n", text)
    return text.strip()


def _parse_tags(raw: str) -> str:
    """Convert '<tag1><tag2>' to 'tag1 tag2'."""
    return " ".join(re.findall(r"<([^>]+)>", raw))


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_dump(url: str, dest: Path, session: requests.Session) -> None:
    """Stream the .7z dump to *dest*, showing progress."""
    print(f"Downloading dump from {url}")
    print(f"  → {dest}")
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded / 1_048_576:.1f} / "
                        f"{total / 1_048_576:.1f} MB",
                        end="",
                        flush=True,
                    )
    print(f"\r  Download complete ({downloaded / 1_048_576:.1f} MB)")


def _extract_posts_xml(archive_path: Path, xml_dest: Path) -> None:
    """Extract only Posts.xml from the .7z archive to *xml_dest*."""
    print(f"Extracting Posts.xml …")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        # py7zr 1.0 dropped read() (in-memory BytesIO); extract() writes
        # straight to disk under xml_dest.parent, named after its archive
        # entry — i.e. exactly xml_dest, since the dump stores it at the root.
        z.extract(path=xml_dest.parent, targets=["Posts.xml"])
    print(f"  {xml_dest.stat().st_size / 1_048_576:.1f} MB uncompressed")


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_posts(xml_path: Path, min_score: int):
    """
    Parse Posts.xml and return (questions, answers) dicts.

    questions : {id_str: {'title', 'body', 'score', 'accepted_id', 'tags'}}
    answers   : {parent_id_str: [{'id', 'body', 'score'}, …]}
    """
    questions: dict[str, dict] = {}
    answers:   dict[str, list] = {}

    total = 0
    print("Parsing Posts.xml …")

    for event, elem in ET.iterparse(str(xml_path), events=["end"]):
        if elem.tag != "row":
            continue

        total += 1
        if total % 50_000 == 0:
            print(f"  … {total:,} rows processed")

        post_type = elem.get("PostTypeId")
        score     = int(elem.get("Score", 0))

        if post_type == "1":  # question
            if score >= min_score:
                questions[elem.get("Id")] = {
                    "title":       elem.get("Title", ""),
                    "body":        elem.get("Body", ""),
                    "score":       score,
                    "accepted_id": elem.get("AcceptedAnswerId"),
                    "tags":        _parse_tags(elem.get("Tags", "")),
                }

        elif post_type == "2":  # answer
            parent_id = elem.get("ParentId")
            if parent_id and parent_id in questions or True:
                # collect all answers; filter to known questions later
                answers.setdefault(parent_id, []).append({
                    "id":    elem.get("Id"),
                    "body":  elem.get("Body", ""),
                    "score": score,
                })

        elem.clear()

    print(f"  Done — {total:,} rows, {len(questions):,} questions kept (score ≥ {min_score})")
    return questions, answers


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_qa(q: dict, ans_list: list[dict], max_answers: int) -> str:
    """Format one question + its top answers as plain text."""
    parts: list[str] = []

    # Question header
    title = q["title"].strip()
    tags  = q["tags"]
    parts.append(f"# {title}")
    if tags:
        parts.append(f"Tags: {tags}")
    parts.append(f"Score: {q['score']}")
    parts.append("")
    parts.append(_html_to_text(q["body"]))

    # Sort answers: accepted first, then by score descending.
    accepted_id = q.get("accepted_id")
    sorted_ans = sorted(
        ans_list,
        key=lambda a: (a["id"] != accepted_id, -a["score"]),
    )

    for ans in sorted_ans[:max_answers]:
        if ans["score"] < 0:
            continue  # skip heavily downvoted answers
        label = "Answer (accepted)" if ans["id"] == accepted_id else "Answer"
        parts.append(f"\n## {label}  (score: {ans['score']})")
        parts.append("")
        parts.append(_html_to_text(ans["body"]))

    parts.append("\n---")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_chunks(
    questions: dict,
    answers: dict,
    out_dir: Path,
    chunk_size: int,
    max_answers: int,
    min_answer_score: int,
) -> None:
    """Render Q&A pairs and write them in chunks of *chunk_size* per file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx  = 0
    pair_count = 0
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal chunk_idx
        if not buffer:
            return
        path = out_dir / f"rpg_se_{chunk_idx:04d}.txt"
        path.write_text("\n\n".join(buffer), encoding="utf-8")
        print(f"  [saved] {path.name}  ({len(buffer)} Q&A pairs)")
        chunk_idx += 1
        buffer.clear()

    for qid, q in questions.items():
        ans_list = [
            a for a in answers.get(qid, [])
            if a["score"] >= min_answer_score
        ]
        if not ans_list:
            continue  # skip unanswered or all-downvoted

        formatted = _format_qa(q, ans_list, max_answers)
        buffer.append(formatted)
        pair_count += 1

        if len(buffer) >= chunk_size:
            _flush()

    _flush()  # remainder
    print(f"\nWrote {pair_count:,} Q&A pairs across {chunk_idx} file(s) → {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an RPG corpus from the Stack Exchange data dump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out",   default="data/corpus/saga",
                        help="Output directory for .txt files")
    parser.add_argument("--cache", default="data/corpus/saga/.cache/se_dump",
                        help="Directory to cache the downloaded .7z and Posts.xml")
    parser.add_argument("--min-score",        type=int, default=1,
                        help="Minimum question score to include")
    parser.add_argument("--min-answer-score", type=int, default=0,
                        help="Minimum answer score to include")
    parser.add_argument("--max-answers",      type=int, default=3,
                        help="Maximum answers to include per question")
    parser.add_argument("--chunk",            type=int, default=200,
                        help="Q&A pairs per output .txt file")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    archive_path = cache_dir / "rpg.stackexchange.com.7z"
    xml_path     = cache_dir / "Posts.xml"
    out_dir      = Path(args.out)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # -- Download ----------------------------------------------------------
    if archive_path.exists():
        print(f"Using cached dump: {archive_path}")
    else:
        _download_dump(DUMP_URL, archive_path, session)

    # -- Extract -----------------------------------------------------------
    if xml_path.exists():
        print(f"Using cached Posts.xml: {xml_path}")
    else:
        _extract_posts_xml(archive_path, xml_path)

    # -- Parse -------------------------------------------------------------
    questions, answers = _parse_posts(xml_path, min_score=args.min_score)

    # -- Write -------------------------------------------------------------
    print(f"\nWriting Q&A pairs to {out_dir} …")
    _write_chunks(
        questions,
        answers,
        out_dir=out_dir,
        chunk_size=args.chunk,
        max_answers=args.max_answers,
        min_answer_score=args.min_answer_score,
    )


if __name__ == "__main__":
    main()
