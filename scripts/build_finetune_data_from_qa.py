"""Convert StackExchange Q&A pairs into a fine-tuning JSONL for the generator.

The retriever side of this corpus (grimoire_ai.llm.data.qa_pairs.load_qa_pairs)
already produces real (question, answer) supervision. This script converts
the same pairs into grimoire_ai.llm.data.conversation.ConversationDataset's
format, so the *generator* can be instruction-tuned on the same real data
instead of the ~30 hand-authored examples it's seen so far (60 fine-tune
steps total on the currently-shipped checkpoint).

--corpus-dir must point at Q&A files that still have their original
StackExchange structural markers (`# title`, `Score: N`, `## Answer
(accepted)? (score: N)`, `---` separators) -- qa_pairs.py's parser keys off
them directly. data/corpus/saga/'s rpg_se_*.txt files no longer qualify:
scripts/clean_stackexchange_markup.py strips exactly those markers (they
were bleeding into pretraining generations verbatim), so pointing this
script at the live pretraining corpus now silently returns zero pairs.
Point it at data/corpus/saga_se_qa_source/ instead -- the pre-cleanup
copy, kept specifically because this script (and embed_tune.py's
--qa-corpus-dir) needs the structured format the pretraining corpus no
longer has.

Usage
-----
python scripts/build_finetune_data_from_qa.py \\
    --corpus-dir data/corpus/saga_se_qa_source/ \\
    --output     data/finetune/saga_se_qa.jsonl \\
    --accepted-only

Then fine-tune with the existing script, pointing --data at the output:

python scripts/finetune_saga.py \\
    --checkpoint checkpoints/finetune/base-294-9/step_0000060.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --data       data/finetune/saga_se_qa.jsonl \\
    --output-dir checkpoints/finetune/saga-se-qa/ \\
    --total-steps 3000

(--total-steps 300, the script's default, was tuned for the original
~30-example dataset; this one is much larger and needs more steps to make a
dent across it -- see scripts/finetune_saga.py's own docstring for guidance.)

Validate the output before training:

python scripts/validate_finetune_data.py \\
    --data  data/finetune/saga_se_qa.jsonl \\
    --vocab data/tokenizer/bpe.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from grimoire_ai.llm.data.qa_pairs import load_qa_pairs, qa_pairs_to_finetune_examples


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert StackExchange Q&A pairs into a ConversationDataset-format "
                     "JSONL for instruction fine-tuning the generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus-dir", required=True, metavar="DIR",
                         help="Directory of StackExchange-format Q&A .txt files "
                              "(rpg_se_*.txt) -- see grimoire_ai.llm.data.qa_pairs.")
    parser.add_argument("--pattern", default="rpg_se_*.txt", metavar="GLOB",
                         help="Glob (within --corpus-dir) selecting which Q&A files "
                              "to load (default: 'rpg_se_*.txt', matching "
                              "scrape_stackexchange_rpg.py's output). Use "
                              "'*_se_*.txt' to pick up every site scraped by "
                              "scrape_stackexchange.py at once, or e.g. "
                              "'history_se_*.txt' for a single site.")
    parser.add_argument("--output", required=True, metavar="PATH",
                         help="Output .jsonl path.")
    parser.add_argument("--min-score", type=int, default=1, metavar="N",
                         help="Minimum answer score to keep (default: 1). Ignored when "
                              "--accepted-only is set.")
    parser.add_argument("--accepted-only", action="store_true",
                         help="Keep only each question's accepted answer, ignoring "
                              "--min-score. Smaller but higher-confidence dataset.")
    parser.add_argument("--context-max-chars", type=int, default=600, metavar="N",
                         help="Max length of the 'context' slice extracted from each "
                              "answer (default: 600).")
    parser.add_argument("--answer-max-chars", type=int, default=350, metavar="N",
                         help="Max length of the 'assistant' slice extracted from each "
                              "answer (default: 350). Should not exceed --context-max-chars.")
    args = parser.parse_args(argv)

    pairs = load_qa_pairs(
        args.corpus_dir, pattern=args.pattern,
        min_score=args.min_score, accepted_only=args.accepted_only,
    )
    print(f"Loaded {len(pairs)} Q&A pair(s) from {args.corpus_dir}")
    if not pairs:
        raise ValueError(
            f"No Q&A pairs found in {args.corpus_dir}. If this directory has been "
            "cleaned by scripts/clean_stackexchange_markup.py, its structural "
            "markers ('# title', 'Score: N', '## Answer (score: N)', '---') are "
            "gone and qa_pairs.py can no longer parse it -- point --corpus-dir at "
            "data/corpus/saga_se_qa_source/ (or another pre-cleanup copy) instead."
        )

    examples = qa_pairs_to_finetune_examples(
        pairs, context_max_chars=args.context_max_chars, answer_max_chars=args.answer_max_chars,
    )
    dropped = len(pairs) - len(examples)
    print(f"Converted {len(examples)} example(s) ({dropped} dropped as too short to split)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.to_json_line() + "\n")
    print(f"Wrote {len(examples)} example(s) to {args.output}")


if __name__ == "__main__":
    main()
