"""Shared Q&A formatting/writing helpers for the Stack Exchange scrapers.

Extracted from ``scrape_stackexchange.py`` so both the archive.org-dump path
and the Hugging Face-mirror path (``scrape_huggingface_stackexchange.py`` --
used when archive.org isn't reachable) produce byte-identical output shape
from the same underlying Posts.xml-equivalent schema (question dict with
``title``/``body``/``score``/``accepted_id``/``tags``, answer dicts with
``id``/``body``/``score``). Neither the download helper nor anything in this
module depends on ``py7zr`` -- that's archive.org-.7z-specific and stays in
``scrape_stackexchange.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "grimoire-corpus-scraper/1.0 (academic/research use; "
    "github.com/BasLinders/grimoire)"
)

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


def _download_file(url: str, dest: Path, session: requests.Session) -> None:
    """Stream a file to *dest*, showing progress. Generic -- works for the
    archive.org .7z dump or a Hugging Face .parquet shard alike."""
    print(f"Downloading from {url}")
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


def _format_qa(q: dict, ans_list: list[dict], max_answers: int) -> str:
    """Format one question + its top answers as plain text.

    Output shape:
        # Question title
        Tags: [tag1] [tag2]
        Score: N

        <question body as plain text>

        ## Answer  (score: N)

        <answer body as plain text>

        ---
    """
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
