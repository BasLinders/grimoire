"""Download and parse a Stack Exchange site's Q&A data from a Hugging Face
mirror, for sites/networks where archive.org (the official dump host) isn't
reachable.

Why this exists
----------------
``scrape_stackexchange.py`` downloads from archive.org, the official host
for Stack Exchange's quarterly data dumps. That host is unreachable from
some networks (confirmed: silently dropped connections, not a DNS or
general-connectivity issue -- consistent with an ISP-level block rather
than an archive.org outage). Stack Exchange also stopped publishing new
dumps to archive.org as of mid-2024, so even where it's reachable the data
is aging.

``HuggingFaceTB/stackexchange_2025_md`` (the default ``--repo``) mirrors the
same underlying per-post data as one Parquet file (or a few size-sharded
files) per site, at ``{site}.stackexchange.com/`` in the repo -- same site
slugs, same ``Body``/``Score``/``Tags``/answer-with-``IsAccepted`` shape as
the archive.org XML dumps (verified directly against the schema: ``Body``
is raw HTML, not pre-rendered text, so the exact same HTML-stripping and
Q&A formatting in ``_stackexchange_common.py`` applies unchanged). Output
is byte-identical in shape to ``scrape_stackexchange.py``'s -- same
``{site}_se_NNNN.txt`` naming, same downstream
``build_finetune_data_from_qa.py``/``--weight-pattern`` commands, no
special-casing needed anywhere else in the pipeline.

Requires the ``pyarrow`` extra (see ``pyproject.toml``'s ``scraper`` group)
to read the Parquet shards -- not needed by the archive.org path, which is
why it isn't a base dependency.

Usage
-----
    python scripts/scrape_huggingface_stackexchange.py --site worldbuilding
    python scripts/scrape_huggingface_stackexchange.py --site gaming --out data/corpus/general_qa

Then build fine-tuning data exactly as with the archive.org path:

    python scripts/build_finetune_data_from_qa.py \\
        --corpus-dir data/corpus/general_qa/ \\
        --pattern    "*_se_*.txt" \\
        --output     data/finetune/general_se_qa.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import requests

from _stackexchange_common import USER_AGENT, _download_file, _parse_tags, _write_chunks

DEFAULT_REPO = "HuggingFaceTB/stackexchange_2025_md"
TREE_API_TEMPLATE = "https://huggingface.co/api/datasets/{repo}/tree/main/{site}.stackexchange.com"
RESOLVE_TEMPLATE = "https://huggingface.co/datasets/{repo}/resolve/main/{site}.stackexchange.com/{filename}"

_COLUMNS = ["Id", "Title", "Body", "Score", "Tags", "AcceptedAnswerId", "Answers"]


def _discover_shards(repo: str, site: str, session: requests.Session) -> list[str]:
    """List Parquet shard filenames for *site* via the HF Hub tree API.

    Large sites are split into multiple ``train-NNNNN-of-NNNNN.parquet``
    shards (e.g. worldbuilding.stackexchange.com is 2); small sites are one
    file. Discovered rather than assumed, since shard count varies by site
    size and isn't predictable from the slug alone.
    """
    url = TREE_API_TEMPLATE.format(repo=repo, site=site)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        raise SystemExit(
            f"'{site}.stackexchange.com' not found in {repo} -- check the "
            f"slug (https://stackexchange.com/sites), or fall back to "
            f"scrape_stackexchange.py if archive.org is reachable for it."
        )
    resp.raise_for_status()
    filenames = sorted(
        entry["path"].rsplit("/", 1)[-1]
        for entry in resp.json()
        if entry["type"] == "file" and entry["path"].endswith(".parquet")
    )
    if not filenames:
        raise SystemExit(f"No Parquet shards found for '{site}' in {repo}.")
    return filenames


def _load_questions_and_answers(
    shard_paths: list[Path], min_score: int,
) -> tuple[dict, dict]:
    """Read Parquet shard(s) into the same (questions, answers) shape
    ``_write_chunks``/``_format_qa`` expect -- one dict keyed by question id.

    Unlike the archive.org XML path (which needs a two-pass correlation of
    separate question/answer rows via ParentId), each Parquet row already
    embeds its own ``Answers`` list, so no correlation step is needed here.
    """
    questions: dict = {}
    answers: dict = {}

    for shard_path in shard_paths:
        print(f"Reading {shard_path.name} ...")
        table = pq.read_table(shard_path, columns=_COLUMNS)
        rows = table.to_pylist()
        kept = 0
        for row in rows:
            if row["Score"] is None or row["Score"] < min_score:
                continue
            qid = row["Id"]
            questions[qid] = {
                "title": row["Title"] or "",
                "body": row["Body"] or "",
                "score": row["Score"],
                "accepted_id": row["AcceptedAnswerId"],
                "tags": _parse_tags(row["Tags"] or ""),
            }
            answers[qid] = [
                {"id": a["Id"], "body": a["Body"] or "", "score": a["Score"]}
                for a in (row["Answers"] or [])
            ]
            kept += 1
        print(f"  {len(rows):,} rows, {kept:,} questions kept (score >= {min_score})")

    return questions, answers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fine-tuning corpus from a Stack Exchange site's "
                     "data, via a Hugging Face Parquet mirror instead of "
                     "archive.org.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--site", required=True,
                         help="Site slug, e.g. 'worldbuilding' for "
                              "worldbuilding.stackexchange.com. Same slugs "
                              "as scrape_stackexchange.py.")
    parser.add_argument("--repo", default=DEFAULT_REPO,
                         help="Hugging Face dataset repo to pull from.")
    parser.add_argument("--out", default="data/corpus/general_qa",
                         help="Output directory for .txt files")
    parser.add_argument("--cache", default=None,
                         help="Directory to cache downloaded .parquet shards "
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

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # -- Discover + download ------------------------------------------------
    shard_names = _discover_shards(args.repo, args.site, session)
    print(f"Found {len(shard_names)} shard(s) for {args.site}.stackexchange.com")

    shard_paths: list[Path] = []
    for filename in shard_names:
        shard_path = cache_dir / filename
        if shard_path.exists():
            print(f"Using cached shard: {shard_path}")
        else:
            url = RESOLVE_TEMPLATE.format(repo=args.repo, site=args.site, filename=filename)
            _download_file(url, shard_path, session)
        shard_paths.append(shard_path)

    # -- Parse ----------------------------------------------------------------
    questions, answers = _load_questions_and_answers(shard_paths, min_score=args.min_score)

    # -- Write ------------------------------------------------------------------
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
