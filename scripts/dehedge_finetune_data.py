"""Strip common forum-answer hedge openers from fine-tune JSONL "assistant" fields.

Every "same-content-different-wording" Stack Exchange source in this
project's fine-tune mix (see docs/training_PLAN.md's Step 8) inherits the
*format* of a forum answer, not just its topic -- and that format survives
fine-tuning regardless of what the pretrain corpus emphasized (see
docs/known_bugs.md's register-drift entry). A large, recurring share of
that register is a handful of meta-commentary clauses stapled onto the
front of an otherwise fine answer: "I would say that X", "As you can see
in this answer, X", "You are correct that X" -- pure hedging/narration
about the act of answering, not part of the actual content.

This is a deterministic, code-verified pass over the existing data (same
philosophy as generate_open5e_entigraph.py's rejection of LLM-based
rephrasing: no invented content, no model call, every change is a fixed
regex substitution you can read and audit). It is deliberately narrow and
conservative -- only leading clauses with no real semantic content are
stripped (never genuine epistemic hedges like "I'm not sure whether...",
which carry real uncertainty and would change the claim's meaning if
removed) and only at the very start of the response, never mid-paragraph,
to avoid corrupting the rest of the answer. It is an imprecise fix, not a
complete one: it will miss hedge phrasings not in its pattern list, and
it does nothing about other Q&A-shaped tells (direct address to "you",
follow-up-question framing) beyond the opening clause.

Only the "assistant" field is touched. "user" and "context" are left
alone. Examples where stripping would leave nothing (or only a few
characters) are skipped entirely rather than producing a degenerate
answer.

Usage
-----
    python scripts/dehedge_finetune_data.py \\
        --input  data/finetune/general_se_qa_downsampled.jsonl \\
        --output data/finetune/general_se_qa_dehedged.jsonl \\
        --dry-run

    python scripts/dehedge_finetune_data.py \\
        --input  data/finetune/general_se_qa_downsampled.jsonl \\
        --output data/finetune/general_se_qa_dehedged.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Ordered by rough specificity -- longer/more specific phrasings first so a
# generic pattern doesn't fire on a substring of a more specific one before
# it gets a chance to match as a whole.
_LEAD_HEDGE_PATTERNS = [
    r"as you can see in this answer,?\s*",
    r"as you (?:can see|already know),?\s*",
    r"i'?m going to answer this(?: by using| by| using)?(?: the general rule)?(?: that)?\s*",
    r"what i would like to add(?: to the answer)?(?: is)?(?: that)?\s*",
    r"i'?d like to add(?: that)?\s*",
    r"i'?ve been (?:to|in) a similar situation(?: in one of the previous answers)?,?\s*(?:but\s+)?",
    r"i would (?:say|suggest|recommend|argue|note)(?: that)?\s*",
    r"i'?d (?:say|suggest|recommend|argue|note)(?: that)?\s*",
    r"you (?:are|'?re) correct(?: that)?\s*",
    r"i think(?: that)?\s*",
    r"i believe(?: that)?\s*",
    r"i remember(?: that| from experience that)?\s*",
    r"in my experience,?\s*",
    r"from what i understand,?\s*",
    r"to answer your question,?\s*",
]

# Applied at the start of the (remaining) text only, case-insensitive.
_LEAD_HEDGE_RE = re.compile(
    r"^(?:" + "|".join(_LEAD_HEDGE_PATTERNS) + r")",
    re.IGNORECASE,
)

_MIN_REMAINING_CHARS = 20
_MAX_STRIP_ITERATIONS = 3


def dehedge(text: str) -> str:
    """Strip chained leading hedge clauses from *text*, capitalizing what's left.

    Loops up to _MAX_STRIP_ITERATIONS times so a chained hedge ("As you can
    see in this answer, I would suggest that X") gets fully stripped, not
    just its outermost clause. Returns *text* unchanged if stripping would
    leave fewer than _MIN_REMAINING_CHARS characters, to avoid producing a
    degenerate training example.
    """
    stripped = text
    for _ in range(_MAX_STRIP_ITERATIONS):
        new = _LEAD_HEDGE_RE.sub("", stripped, count=1)
        if new == stripped:
            break
        stripped = new.lstrip()
    if stripped == text:
        return text
    if len(stripped) < _MIN_REMAINING_CHARS:
        return text
    return stripped[0].upper() + stripped[1:] if stripped else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, metavar="PATH")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report stats without writing --output.",
    )
    parser.add_argument(
        "--show-examples", type=int, default=5, metavar="N",
        help="Print N before/after examples of changed entries (default: 5).",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Not a file: {in_path}")

    total = 0
    changed = 0
    shown = 0
    out_lines: list[str] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            obj = json.loads(line)
            original = obj.get("assistant", "")
            new = dehedge(original)
            if new != original:
                changed += 1
                if shown < args.show_examples:
                    print(f"--- example {shown + 1} ---")
                    print(f"  before: {original[:120]!r}")
                    print(f"  after:  {new[:120]!r}")
                    shown += 1
                obj["assistant"] = new
            out_lines.append(json.dumps(obj))

    pct = (changed / total * 100) if total else 0.0
    print(f"\n{changed}/{total} example(s) changed ({pct:.1f}%).")

    if args.dry_run:
        print("Dry run -- nothing written. Rerun without --dry-run to write --output.")
        return

    out_path = Path(args.output)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(out_lines)} example(s) -> {out_path}")


if __name__ == "__main__":
    main()
