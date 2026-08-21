"""Interactive terminal chat loop for Grimoire.

Starts a stateful multi-turn conversation with a loaded model checkpoint.
Type your query and press Enter; the model responds.  The conversation
history is maintained automatically via ``ConversationState``.

Usage
-----
    python -m grimoire.cli.chat \\
        --checkpoint checkpoints/finetune/step_0000500.pt \\
        --vocab      data/tokenizer/bpe.json

With a corpus for retrieval-augmented responses:

    python -m grimoire.cli.chat \\
        --checkpoint checkpoints/finetune/step_0000500.pt \\
        --vocab      data/tokenizer/bpe.json \\
        --corpus-dir data/corpus/

Special commands (type during chat):
    /clear   — Erase conversation history and start fresh.
    /history — Print all turns in the current session.
    /quit    — Exit the chat loop.
"""

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive multi-turn chat with a Grimoire model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a fine-tuned checkpoint (.pt).",
    )
    p.add_argument(
        "--vocab", default="data/tokenizer/bpe.json",
        help="Path to the BPE vocabulary JSON.",
    )
    p.add_argument(
        "--corpus-dir", default=None,
        help="Directory of .txt corpus files for retrieval-augmented responses.",
    )
    p.add_argument(
        "--top-k-corpus", type=int, default=3,
        help="Number of corpus passages to retrieve per query.",
    )
    p.add_argument(
        "--encoder",
        default="model",
        choices=["model", "minilm", "mpnet", "lexical"],
        help="Embedding backend for corpus retrieval. "
             "model: the trained transformer's own embeddings (default). "
             "minilm: all-MiniLM-L6-v2 via sentence-transformers. "
             "mpnet: all-mpnet-base-v2 via sentence-transformers. "
             "lexical: Jaccard word-overlap (no neural embedding).",
    )
    p.add_argument(
        "--retrieval-threshold", type=float, default=None,
        help="Minimum retrieval score to inject corpus context. Queries whose "
             "best match scores below this value are answered without grounding "
             "(pure-chat). For semantic retrieval cosine scores are in [-1, 1]; "
             "0.0 is a reasonable starting point. Omit to always inject context "
             "when a corpus is attached.",
    )
    p.add_argument(
        "--max-turns", type=int, default=20,
        help="Maximum number of turns to keep in rolling history.",
    )
    p.add_argument(
        "--temperature", type=float, default=0.8,
    )
    p.add_argument(
        "--top-k", type=int, default=50,
    )
    p.add_argument(
        "--top-p", type=float, default=0.9,
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=256,
    )
    p.add_argument(
        "--math-tool", action="store_true",
        help="Enable the math tool: detect arithmetic in queries, evaluate it safely, "
             "and inject the result as context before generation.  Also resolves "
             "<TOOL:python>...</TOOL> tags in model responses (for fine-tuned models).",
    )
    p.add_argument(
        "--stat-block-constraint", action="store_true",
        help="Restrict Challenge Rating / XP / AC / HP values to well-formed "
             "continuations at decode time — structurally blocks a hallucinated "
             "value instead of hoping the model got it right. Decode-time only; "
             "does not affect prose generation elsewhere in the response.",
    )
    p.add_argument(
        "--loop-guard", action="store_true",
        help="Hard-ban a token that would extend an already-established repeating "
             "loop ('does does does...' or short-phrase loops), a structural "
             "backstop stronger than the soft repetition_penalty discount. Uses "
             "RepetitionLoopGuard's defaults (max_repeats=3, max_period=4).",
    )
    p.add_argument(
        "--loop-guard-max-period", type=int, default=4, metavar="N",
        help="Longest repeating block length (tokens) the loop guard checks "
             "(default: 4). Raise this to catch whole repeating sentence "
             "templates, not just short phrases. Ignored unless --loop-guard "
             "is set.",
    )
    p.add_argument(
        "--loop-guard-template-match-ratio", type=float, default=1.0, metavar="R",
        help="Fraction of positions within a repeating block that must match "
             "across cycles before the loop guard fires (default: 1.0, exact "
             "repeats only). Lower it (e.g. 0.6) to also catch templated loops "
             "where a value is substituted each cycle ('CR = 10 + Dex bonus. "
             "CR = 14 + Str bonus...'). Ignored unless --loop-guard is set.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the interactive chat loop."""
    args = _parse_args(argv)

    from grimoire_ai.corpus.corpus import GrimoireCorpus
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.llm.inference.semantic import EXTERNAL_ENCODERS, SemanticRetriever, make_external_embed_fn
    from grimoire_ai.state.conversation import ConversationState

    _ENCODER_MAP = {
        "minilm": "MiniLM (all-MiniLM-L6-v2)",
        "mpnet":  "MPNet (all-mpnet-base-v2)",
    }
    use_lexical  = args.encoder == "lexical"
    use_external = args.encoder in _ENCODER_MAP

    # --- Load corpus (optional) -----------------------------------------
    corpus = None
    documents: list[tuple[str, str]] = []
    if args.corpus_dir is not None:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            print(f"Warning: corpus directory not found: {corpus_dir}", file=sys.stderr)
        else:
            for txt_file in sorted(corpus_dir.glob("*.txt")):
                text = txt_file.read_text(encoding="utf-8")
                documents.append((text, txt_file.stem))
                if use_lexical:
                    if corpus is None:
                        corpus = GrimoireCorpus()
                    corpus.add_text(text, source=txt_file.stem)
            print(f"Corpus: {len(documents)} file(s) loaded from {corpus_dir} [{args.encoder}]")

    # --- Load engine ----------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    math_tool = None
    if args.math_tool:
        from grimoire_ai.tools.math_tool import MathTool
        math_tool = MathTool()
        print("Math tool enabled — arithmetic in queries will be pre-computed.")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.vocab,
        corpus=corpus,
        retrieval_threshold=args.retrieval_threshold,
        math_tool=math_tool,
    )
    if args.stat_block_constraint:
        from grimoire_ai.llm.inference.constrained_decoding import StatBlockConstraint
        engine.stat_block_constraint = StatBlockConstraint(engine.tokenizer)
        print("Stat-block constraint enabled — CR/XP/AC/HP values are restricted to well-formed continuations.")
    print(f"Model ready on {engine.device.upper()}.")

    # --- Build semantic index -------------------------------------------
    if documents and not use_lexical:
        if use_external:
            encoder_id = EXTERNAL_ENCODERS[_ENCODER_MAP[args.encoder]]
            print(f"Loading external encoder: {encoder_id} ...")
            embed_fn = make_external_embed_fn(encoder_id)
            retriever = SemanticRetriever(embed_fn=embed_fn)
            for text, source in documents:
                retriever.add_text(text, source=source)
            print("Embedding corpus passages...")
            retriever.index()
            engine.corpus = retriever
            engine.retrieval_threshold = args.retrieval_threshold
        else:
            print("Embedding corpus passages with model embeddings...")
            retriever = engine.build_semantic_corpus(documents)
        print(f"Index ready: {retriever.size} passage(s).")

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        loop_guard_max_repeats=3 if args.loop_guard else 0,
        loop_guard_max_period=args.loop_guard_max_period,
        loop_guard_template_match_ratio=args.loop_guard_template_match_ratio,
    )
    if args.loop_guard:
        print("Loop guard enabled — repeating token/phrase loops are hard-blocked.")

    state = ConversationState(max_turns=args.max_turns)

    print("\nGrimoire Chat — type /quit to exit, /clear to reset, /history to review.\n")

    # --- Chat loop ------------------------------------------------------
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        if query == "/quit":
            print("Goodbye.")
            break

        if query == "/clear":
            state.clear()
            print("  [Conversation history cleared.]\n")
            continue

        if query == "/history":
            if state.turn_count == 0:
                print("  [No history yet.]\n")
            else:
                for i, turn in enumerate(state.history, 1):
                    print(f"  [{i}] You: {turn.user}")
                    print(f"       Grimoire: {turn.assistant}")
                print()
            continue

        response = engine.chat(
            query=query,
            state=state,
            top_k_corpus=args.top_k_corpus,
            gen_config=gen_config,
        )
        print(f"Grimoire: {response}\n")


if __name__ == "__main__":
    main()
