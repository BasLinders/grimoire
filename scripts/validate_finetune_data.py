"""Validate a Grimoire fine-tuning JSONL dataset.

Checks that every line parses correctly, reports token length statistics,
and flags examples that would be truncated at a given max_seq_len.

Usage
-----
    python scripts/validate_finetune_data.py \\
        --data  data/finetune/saga_v1.jsonl \\
        --vocab data/tokenizer/bpe.json \\
        --max-seq-len 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate a Grimoire fine-tuning JSONL dataset.")
    parser.add_argument("--data",        required=True, help="Path to .jsonl file.")
    parser.add_argument("--vocab",       required=True, help="Path to BPE tokenizer .json file.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Sequence length cap.")
    args = parser.parse_args(argv)

    from grimoire.llm.tokenizer.bpe import BytePairEncoder
    from grimoire.llm.tokenizer.special_tokens import AST_ID, BOS_ID, EOS_ID, SEP_ID, USR_ID

    tokenizer = BytePairEncoder.load(args.vocab)

    lines = Path(args.data).read_text(encoding="utf-8").splitlines()
    lines = [l.strip() for l in lines if l.strip()]

    errors: list[str] = []
    lengths: list[int] = []
    truncated = 0

    for i, line in enumerate(lines, 1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"Line {i}: invalid JSON — {exc}")
            continue

        if "user" not in obj or "assistant" not in obj:
            errors.append(f"Line {i}: missing 'user' or 'assistant' field.")
            continue

        user_ids = tokenizer.encode(obj["user"])
        asst_ids = tokenizer.encode(obj["assistant"])
        ctx_ids  = tokenizer.encode(obj["context"]) if "context" in obj else []

        if ctx_ids:
            total = 1 + 1 + len(ctx_ids) + 1 + 1 + len(user_ids) + 1 + len(asst_ids) + 1
        else:
            total = 1 + 1 + len(user_ids) + 1 + len(asst_ids) + 1

        lengths.append(total)
        if total > args.max_seq_len:
            truncated += 1

    # Report.
    print(f"Dataset:   {args.data}")
    print(f"Examples:  {len(lengths)} valid, {len(errors)} errors")
    if lengths:
        print(f"Lengths:   min={min(lengths)}  max={max(lengths)}  avg={sum(lengths)/len(lengths):.1f}  (max_seq_len={args.max_seq_len})")
        print(f"Truncated: {truncated} ({100*truncated/len(lengths):.1f}%)")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll examples valid.")


if __name__ == "__main__":
    main()
