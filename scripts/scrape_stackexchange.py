"""Download and parse any Stack Exchange site's data dump.

Generalizes ``scripts/scrape_stackexchange_rpg.py`` (which is hardcoded to
rpg.stackexchange.com) to any site in the network, selected by ``--site``.
Same dump format (quarterly, CC BY-SA 4.0, published on the Internet
Archive), same Posts.xml schema, same Q&A-pair extraction and output format
-- only the site slug and output filename prefix differ.

Why this exists
----------------
The rpg.stackexchange.com corpus already prioritizes D&D-specific content for
*pretraining* (see docs/expansion_PLAN.md's source weighting). But teaching
the *generator* to converse coherently -- stay on topic, answer directly
instead of echoing the question, avoid repetition loops -- doesn't require
D&D-topical data. It requires real human question/answer pairs in a variety
of registers. Pointing this script at general-audience Stack Exchange sites
(not rpg) produces exactly that supervision signal at a scale hand-authored
examples (see scripts/finetune_data/general_conversations.jsonl, 64
examples) can't reach, without touching the D&D-specific corpus or its
weighting at all.

Suggested general (non-D&D) sites for conversational variety, all
similarly-sized and available at the same dump URL pattern:
    history    -- history.stackexchange.com   (factual/explanatory register)
    travel     -- travel.stackexchange.com    (practical advice register)
    skeptics   -- skeptics.stackexchange.com  (evidence/reasoning register)
Any site slug from https://stackexchange.com/sites works, as long as the
site is small/medium-sized (Stack Overflow's dump is split across many
files and not handled by this script's single-archive download).

Output format per Q&A pair
--------------------------
    # Question title
    Tags: [tag1] [tag2]
    Score: N

    <question body as plain text>

    ## Answer  (score: N)

    <answer body as plain text>

    ---

Each output file contains up to ``--chunk`` Q&A pairs (default 200), named
``{site}_se_NNNN.txt`` -- the ``{site}_se_`` prefix keeps output from
different sites distinguishable in the same output directory and never
collides with rpg.stackexchange.com's own ``rpg_se_*.txt`` files.

Usage
-----
    python scripts/scrape_stackexchange.py --site history
    python scripts/scrape_stackexchange.py --site travel --out data/corpus/general_qa
    python scripts/scrape_stackexchange.py --site skeptics --min-score 2

Then build fine-tuning data the same way as the rpg pipeline:

    python scripts/build_finetune_data_from_qa.py \\
        --corpus-dir data/corpus/general_qa/ \\
        --pattern    "*_se_*.txt" \\
        --output     data/finetune/general_se_qa.jsonl

Requirements
------------
    pip install "grimoire-ai[scraper]"  # includes py7zr, requests, beautifulsoup4
"""

from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import py7zr
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DUMP_URL_TEMPLATE = "https://archive.org/download/stackexchange/{site}.stackexchange.com.7z"
USER_AGENT = (
    "grimoire-corpus-scraper/1.0 (academic/research use; "
    "github.com/BasLinders/grimoire)"
)

# ---------------------------------------------------------------------------
# HTML -> plain text
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
    print(f"  -> {dest}")
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
    print("Extracting Posts.xml ...")
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extract(path=xml_dest.parent, targets=["Posts.xml"])
    print(f"  {xml_dest.stat().st_size / 1_048_576:.1f} MB uncompressed")


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def _parse_posts(xml_path: Path, min_score: int):
    """
    Parse Posts.xml and return (questions, answers) dicts.

    questions : {id_str: {'title', 'body', 'score', 'accepted_id', 'tags'}}
    answers   : {parent_id_str: [{'id', 'body', 'score'}, ...]}
    """
    questions: dict[str, dict] = {}
    answers: dict[str, list] = {}

    total = 0
    print("Parsing Posts.xml ...")

    for event, elem in ET.iterparse(str(xml_path), events=["end"]):
        if elem.tag != "row":
            continue

        total += 1
        if total % 50_000 == 0:
            print(f"  ... {total:,} rows processed")

        post_type = elem.get("PostTypeId")
        score = int(elem.get("Score", 0))

        if post_type == "1":  # question
            if score >= min_score:
                questions[elem.get("Id")] = {
                    "title": elem.get("Title", ""),
                    "body": elem.get("Body", ""),
                    "score": score,
                    "accepted_id": elem.get("AcceptedAnswerId"),
                    "tags": _parse_tags(elem.get("Tags", "")),
                }

        elif post_type == "2":  # answer
            parent_id = elem.get("ParentId")
            if parent_id:
                answers.setdefault(parent_id, []).append({
                    "id": elem.get("Id"),
                    "body": elem.get("Body", ""),
                    "score": score,
                })

        elem.clear()

    print(f"  Done -- {total:,} rows, {len(questions):,} questions kept (score >= {min_score})")
    return questions, answers


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _format_qa(q: dict, ans_list: list[dict], max_answers: int) -> str:
    """Format one question + its top answers as plain text."""
    parts: list[str] = []

    title = q["title"].strip()
    tags = q["tags"]
    parts.append(f"# {title}")
    if tags:
        parts.append(f"Tags: {tags}")
    parts.append(f"Score: {q['score']}")
    parts.append("")
    parts.append(_html_to_text(q["body"]))

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
    prefix: str,
    chunk_size: int,
    max_answers: int,
    min_answer_score: int,
) -> None:
    """Render Q&A pairs and write them in chunks of *chunk_size* per file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    pair_count = 0
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal chunk_idx
        if not buffer:
            return
        path = out_dir / f"{prefix}{chunk_idx:04d}.txt"
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
    print(f"\nWrote {pair_count:,} Q&A pairs across {chunk_idx} file(s) -> {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fine-tuning corpus from any Stack Exchange site's data dump",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--site", required=True,
                         help="Site slug, e.g. 'history' for history.stackexchange.com. "
                              "See https://stackexchange.com/sites for valid slugs. "
                              "Stack Overflow itself is not supported (multi-file dump).")
    parser.add_argument("--out", default="data/corpus/general_qa",
                         help="Output directory for .txt files")
    parser.add_argument("--cache", default=None,
                         help="Directory to cache the downloaded .7z and Posts.xml "
                              "(default: <out>/.cache/<site>)")
    parser.add_argument("--min-score", type=int, default=1,
                         help="Minimum question score to include")
    parser.add_argument("--min-answer-score", type=int, default=0,
                         help="Minimum answer score to include")
    parser.add_argument("--max-answers", type=int, default=3,
                         help="Maximum answers to include per question")
    parser.add_argument("--chunk", type=int, default=200,
                         help="Q&A pairs per output .txt file")
    args = parser.parse_args()

    out_dir = Path(args.out)
    cache_dir = Path(args.cache) if args.cache else out_dir / ".cache" / args.site
    cache_dir.mkdir(parents=True, exist_ok=True)

    dump_url = DUMP_URL_TEMPLATE.format(site=args.site)
    archive_path = cache_dir / f"{args.site}.stackexchange.com.7z"
    xml_path = cache_dir / "Posts.xml"

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # -- Download ------------------------------------------------------
    if archive_path.exists():
        print(f"Using cached dump: {archive_path}")
    else:
        _download_dump(dump_url, archive_path, session)

    # -- Extract ---------------------------------------------------------
    if xml_path.exists():
        print(f"Using cached Posts.xml: {xml_path}")
    else:
        _extract_posts_xml(archive_path, xml_path)

    # -- Parse -------------------------------------------------------------
    questions, answers = _parse_posts(xml_path, min_score=args.min_score)

    # -- Write -------------------------------------------------------------
    print(f"\nWriting Q&A pairs to {out_dir} ...")
    _write_chunks(
        questions,
        answers,
        out_dir=out_dir,
        prefix=f"{args.site}_se_",
        chunk_size=args.chunk,
        max_answers=args.max_answers,
        min_answer_score=args.min_answer_score,
    )


if __name__ == "__main__":
    main()
