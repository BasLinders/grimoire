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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the interactive chat loop."""
    args = _parse_args(argv)

    from grimoire_ai.corpus.corpus import GrimoireCorpus
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.state.conversation import ConversationState

    # --- Load corpus (optional) -----------------------------------------
    corpus = None
    if args.corpus_dir is not None:
        corpus_dir = Path(args.corpus_dir)
        if not corpus_dir.exists():
            print(f"Warning: corpus directory not found: {corpus_dir}", file=sys.stderr)
        else:
            corpus = GrimoireCorpus()
            loaded = 0
            for txt_file in sorted(corpus_dir.glob("*.txt")):
                corpus.add_text(
                    txt_file.read_text(encoding="utf-8"),
                    source=txt_file.stem,
                )
                loaded += 1
            print(f"Corpus: {loaded} file(s) loaded from {corpus_dir}")

    # --- Load engine ----------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    engine = InferenceEngine(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.vocab,
        corpus=corpus,
    )
    print(f"Model ready on {engine.device.upper()}.")

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

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
