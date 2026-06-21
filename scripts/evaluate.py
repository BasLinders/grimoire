"""Evaluation harness CLI.

Runs perplexity, retrieval hit-rate, and/or quiz evaluations on a
Grimoire checkpoint and prints a summary.

Usage examples
--------------
# All three evaluators (full suite):
python scripts/evaluate.py \\
    --checkpoint checkpoints/finetune/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --corpus-dir data/corpus/saga/ \\
    --corpus-bin data/processed/corpus.bin \\
    --quiz       scripts/eval_data/saga_quiz.jsonl

# Perplexity only:
python scripts/evaluate.py \\
    --checkpoint checkpoints/pretrain/step_0010000.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --corpus-bin data/processed/corpus.bin

# Quiz only (no corpus required):
python scripts/evaluate.py \\
    --checkpoint checkpoints/finetune/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --quiz       scripts/eval_data/saga_quiz.jsonl
"""

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Grimoire evaluation suite.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to a .pt checkpoint.",
    )
    parser.add_argument(
        "--vocab", required=True, metavar="PATH",
        help="Path to the BPE vocabulary JSON.",
    )
    parser.add_argument(
        "--corpus-dir", metavar="DIR", default="",
        help="Directory of .txt corpus files (for retrieval eval).",
    )
    parser.add_argument(
        "--corpus-bin", metavar="PATH", default="",
        help="Path to the tokenised corpus .bin (for perplexity eval).",
    )
    parser.add_argument(
        "--quiz", metavar="PATH", default="",
        help="Path to a JSONL quiz file (for quiz eval). "
             "Defaults to scripts/eval_data/saga_quiz.jsonl when it exists.",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default="data/eval",
        help="Directory for the JSON report (default: data/eval).",
    )
    parser.add_argument(
        "--encoder", choices=["lexical", "model", "minilm", "mpnet"], default="lexical",
        help="Retrieval embedding backend (default: lexical). 'model' uses the "
             "checkpoint's own embeddings; 'minilm'/'mpnet' use a dedicated sentence "
             "encoder independent of the checkpoint (requires pip install -e \".[encoder]\").",
    )
    parser.add_argument(
        "--max-ppl-batches", type=int, default=50, metavar="N",
        help="Cap on perplexity eval batches (0 = all; default: 50).",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Apply int8 quantization when loading the model.",
    )
    parser.add_argument(
        "--quiz-repetition-penalty", type=float, default=1.0, metavar="X",
        help="Penalty (>1.0) applied to already-generated tokens during quiz "
             "generation, to discourage repetition loops. 1.0 (default) disables it.",
    )
    parser.add_argument(
        "--math-tool", action="store_true",
        help="Detect arithmetic/probability in quiz questions and inject the computed "
             "result as context, and resolve <TOOL:python> tags in responses.",
    )
    parser.add_argument(
        "--retrieval-threshold", type=float, default=None, metavar="X",
        help="Minimum top-1 score for retrieved context to be injected during quiz "
             "generation. Default (unset) always injects the top-1 result, even an "
             "irrelevant one. Cosine scores live in [-1, 1].",
    )
    args = parser.parse_args()

    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.eval.harness import run_eval

    corpus_dir = args.corpus_dir.strip()

    math_tool = None
    if args.math_tool:
        from grimoire_ai.tools.math_tool import MathTool
        math_tool = MathTool()

    print("Loading model …")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.vocab,
        quantize=args.quantize,
        math_tool=math_tool,
        retrieval_threshold=args.retrieval_threshold,
    )

    if corpus_dir and Path(corpus_dir).is_dir():
        documents: list[tuple[str, str]] = []
        for txt in sorted(Path(corpus_dir).glob("*.txt")):
            documents.append((txt.read_text(encoding="utf-8"), txt.stem))
        if documents:
            if args.encoder == "model":
                print(f"Building semantic corpus from {len(documents)} file(s) …")
                engine.build_semantic_corpus(documents)
            elif args.encoder in ("minilm", "mpnet"):
                from grimoire_ai.llm.inference.semantic import SemanticRetriever, make_external_embed_fn
                model_id = "all-MiniLM-L6-v2" if args.encoder == "minilm" else "all-mpnet-base-v2"
                print(f"Embedding {len(documents)} file(s) with {model_id} …")
                embed_fn = make_external_embed_fn(model_id)
                retriever = SemanticRetriever(embed_fn=embed_fn)
                for text, source in documents:
                    retriever.add_text(text, source=source)
                retriever.index()
                engine.corpus = retriever
            else:
                from grimoire_ai.corpus.corpus import GrimoireCorpus
                corpus = GrimoireCorpus()
                for text, source in documents:
                    corpus.add_text(text, source=source)
                engine.corpus = corpus
                print(f"Lexical corpus loaded: {len(documents)} file(s).")

    run_eval(
        engine=engine,
        corpus_bin=args.corpus_bin or None,
        quiz_path=args.quiz or None,
        output_dir=args.output_dir,
        max_perplexity_batches=args.max_ppl_batches,
        quiz_repetition_penalty=args.quiz_repetition_penalty,
        on_progress=print,
    )


if __name__ == "__main__":
    main()
