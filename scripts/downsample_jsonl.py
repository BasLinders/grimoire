"""Randomly downsample a JSONL file to a target line count.

Generic, not fine-tune-specific -- built for rebalancing
data/finetune/general_se_qa.jsonl against the D&D-specific fine-tune
sources (open5e_qa.jsonl, saga_se_qa.jsonl) when general content would
otherwise outweigh D&D content in a combined dataset, but works on any
line-delimited JSON file.

Uses a uniform random sample without replacement rather than truncating
to the first N lines: build_finetune_data_from_qa.py's output is grouped
by source file in glob order, so a straight head -n N would keep only
whichever site(s) happen to sort first and drop the rest entirely. A
random sample has no such bias -- each source's share of the output is
in expectation the same as its share of the input, without needing a
per-line source tag to stratify by (the JSONL format here carries none).

Usage
-----
    python scripts/downsample_jsonl.py \\
        --input data/finetune/general_se_qa.jsonl \\
        --output data/finetune/general_se_qa_downsampled.jsonl \\
        --n 40000
"""

from __future__ import annotations

import argparse
import random


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly downsample a JSONL file to a target line count.",
    )
    parser.add_argument("--input", required=True, help="Source .jsonl path.")
    parser.add_argument("--output", required=True, help="Destination .jsonl path.")
    parser.add_argument("--n", type=int, required=True, help="Target number of lines to keep.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0).")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    if args.n >= len(lines):
        print(f"--n {args.n} >= {len(lines)} lines in {args.input}; writing all of it unchanged.")
        sampled = lines
    else:
        rng = random.Random(args.seed)
        sampled = rng.sample(lines, args.n)

    with open(args.output, "w", encoding="utf-8") as f:
        f.writelines(sampled)

    print(f"Wrote {len(sampled):,} of {len(lines):,} line(s) from {args.input} to {args.output}")


if __name__ == "__main__":
    main()
