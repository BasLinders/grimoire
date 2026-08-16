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
    # The harness prints Unicode symbols (warning signs, arrows, etc.) throughout
    # its progress messages. Windows consoles often default stdout/stderr to a
    # legacy code page (e.g. cp1252) that can't encode them, crashing a run with
    # a UnicodeEncodeError partway through -- reconfigure to UTF-8 unconditionally
    # so a multi-hour run never dies on a print statement.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        "--encoder", choices=["lexical", "model", "minilm", "mpnet", "lora"], default="lexical",
        help="Retrieval embedding backend (default: lexical). 'model' uses the "
             "checkpoint's own embeddings; 'minilm'/'mpnet' use a dedicated sentence "
             "encoder independent of the checkpoint (requires pip install -e \".[encoder]\"); "
             "'lora' uses the checkpoint's own embeddings through a LoRA adapter trained by "
             "scripts/embed_tune.py (requires --lora). The adapter is applied to a separate "
             "model instance used only for embedding -- the engine that generates chat "
             "responses is unaffected, since the same adapted weights would otherwise change "
             "generation output too.",
    )
    parser.add_argument(
        "--lora", metavar="PATH", default="",
        help="Path to a .lora file from scripts/embed_tune.py. Required when --encoder lora.",
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
        "--quiz-loop-guard-max-repeats", type=int, default=0, metavar="N",
        help="Hard-ban the token that would extend a loop past N consecutive repeats "
             "during quiz generation (0 = off, default). Unlike --quiz-repetition-penalty "
             "(a soft discount), this makes the extending token literally unsampleable -- "
             "see docs/known_bugs.md's repetition-loop entry. Try 3.",
    )
    parser.add_argument(
        "--quiz-loop-guard-max-period", type=int, default=4, metavar="N",
        help="Longest repeating block length (tokens) checked for looping "
             "(default: 4). Ignored if --quiz-loop-guard-max-repeats is 0.",
    )
    parser.add_argument(
        "--math-tool", action="store_true",
        help="Detect arithmetic/probability in quiz questions and inject the computed "
             "result as context, and resolve <TOOL:python> tags in responses.",
    )
    parser.add_argument(
        "--index-batch-size", type=int, default=32, metavar="N",
        help="Passages embedded per forward pass when building a semantic corpus "
             "(model/minilm/mpnet/lora encoders). Larger values amortize fixed "
             "per-call overhead and can meaningfully improve CPU throughput "
             "(default: 32).",
    )
    parser.add_argument(
        "--retrieval-threshold", type=float, default=None, metavar="X",
        help="Minimum top-1 score for retrieved context to be injected during quiz "
             "generation. Default (unset) always injects the top-1 result, even an "
             "irrelevant one. Cosine scores live in [-1, 1].",
    )
    parser.add_argument(
        "--checkpoint-every-files", type=int, default=25, metavar="N",
        help="For --encoder lora: save an index checkpoint every N corpus files "
             "(default: 25). A large corpus can take hours to embed on CPU; this "
             "lets a restart resume from the last checkpoint instead of "
             "re-embedding everything. Checkpoint lives at "
             "<output-dir>/lora_index_checkpoint/.",
    )
    parser.add_argument(
        "--gen-lora", default="", metavar="PATH",
        help="Path to a .lora file from grimoire_ai.llm.training.finetune's "
             "--lora-rank generation fine-tuning (NOT scripts/embed_tune.py's "
             "output -- that one is embedding-only and belongs on --lora "
             "instead). Applied directly to the generation engine, so it "
             "does affect quiz/perplexity output. Unrelated to --encoder "
             "lora/--lora, which adapts a separate embedding-only engine.",
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
    print(f"Device: {engine.device}")

    if args.gen_lora:
        print(f"Loading generation LoRA: {args.gen_lora} …")
        engine.load_lora(args.gen_lora)

    if corpus_dir and Path(corpus_dir).is_dir():
        documents: list[tuple[str, str]] = []
        for txt in sorted(Path(corpus_dir).glob("*.txt")):
            documents.append((txt.read_text(encoding="utf-8"), txt.stem))
        if documents:
            if args.encoder == "model":
                print(f"Building semantic corpus from {len(documents)} file(s) …")
                engine.build_semantic_corpus(documents, batch_size=args.index_batch_size)
            elif args.encoder in ("minilm", "mpnet"):
                from grimoire_ai.llm.inference.semantic import SemanticRetriever, make_external_embed_fn
                model_id = "all-MiniLM-L6-v2" if args.encoder == "minilm" else "all-mpnet-base-v2"
                print(f"Embedding {len(documents)} file(s) with {model_id} …")
                embed_fn = make_external_embed_fn(model_id)
                retriever = SemanticRetriever(embed_fn=embed_fn)
                for text, source in documents:
                    retriever.add_text(text, source=source)
                retriever.index(batch_size=args.index_batch_size)
                engine.corpus = retriever
            elif args.encoder == "lora":
                if not args.lora:
                    raise ValueError("--encoder lora requires --lora <path to .lora file>")
                from grimoire_ai.llm.inference.semantic import SemanticRetriever
                print(f"Loading embedding adapter: {args.lora} …")
                embed_engine = InferenceEngine(
                    checkpoint_path=args.checkpoint,
                    tokenizer_path=args.vocab,
                    quantize=args.quantize,
                )
                print(f"Embedding device: {embed_engine.device}")
                embed_engine.load_lora(args.lora)

                # Embedding a real-sized corpus on CPU can take hours; checkpoint
                # periodically so a restart (e.g. after the machine sleeps) resumes
                # from the last save instead of re-embedding everything. Assumes
                # the corpus hasn't changed between runs -- no staleness check.
                checkpoint_dir = Path(args.output_dir) / "lora_index_checkpoint"
                if checkpoint_dir.exists():
                    print(f"Resuming from index checkpoint: {checkpoint_dir} …")
                    retriever = SemanticRetriever.from_index(checkpoint_dir, embed_fn=embed_engine.embed)
                    print(f"  {retriever.size} passage(s) already indexed from "
                          f"{len(retriever.indexed_sources)} file(s).")
                else:
                    retriever = SemanticRetriever(embed_fn=embed_engine.embed)

                remaining = [(t, s) for t, s in documents if s not in retriever.indexed_sources]
                print(f"Building semantic corpus: {len(remaining)} file(s) remaining "
                      f"of {len(documents)} …")

                files_per_checkpoint = max(args.checkpoint_every_files, 1)
                for i in range(0, len(remaining), files_per_checkpoint):
                    group = remaining[i : i + files_per_checkpoint]
                    for text, source in group:
                        retriever.add_text(text, source=source)
                    retriever.index(on_progress=print, batch_size=args.index_batch_size)
                    retriever.save_index(checkpoint_dir, build_faiss=False)
                    done = min(i + files_per_checkpoint, len(remaining))
                    print(f"  checkpoint saved: {retriever.size} passage(s) indexed "
                          f"({done}/{len(remaining)} file(s) this run) …")

                engine.corpus = retriever
            else:
                from grimoire_ai.corpus.corpus import GrimoireCorpus
                corpus = GrimoireCorpus()
                # Indexing is single-threaded, pure-Python dict/set work with
                # no other output -- on a large corpus (1000+ files) that can
                # take minutes, and silently, which is indistinguishable from
                # a hang without progress output.
                for i, (text, source) in enumerate(documents):
                    corpus.add_text(text, source=source)
                    if i == 0 or (i + 1) % 100 == 0 or i == len(documents) - 1:
                        print(f"  indexing {i+1}/{len(documents)} …")
                engine.corpus = corpus
                print(f"Lexical corpus loaded: {len(documents)} file(s).")

    run_eval(
        engine=engine,
        corpus_bin=args.corpus_bin or None,
        quiz_path=args.quiz or None,
        output_dir=args.output_dir,
        max_perplexity_batches=args.max_ppl_batches,
        quiz_repetition_penalty=args.quiz_repetition_penalty,
        quiz_loop_guard_max_repeats=args.quiz_loop_guard_max_repeats,
        quiz_loop_guard_max_period=args.quiz_loop_guard_max_period,
        on_progress=print,
    )


if __name__ == "__main__":
    main()
