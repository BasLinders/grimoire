"""Side-by-side chat comparison of two fine-tuned checkpoints on fixed prompts.

This project has needed this exact comparison ad hoc several times
already (baseline vs. weighted, weighted vs. weighted_clean, a 7-prompt
check of a new checkpoint against production -- see docs/expansion_PLAN.md)
without ever becoming a reusable script. Unlike scripts/qualitative_check.py
(raw text completion for a checkpoint that hasn't been fine-tuned yet),
this uses InferenceEngine.chat() so both checkpoints see the same
instruction-formatted prompt they were actually fine-tuned to expect --
appropriate for comparing two *fine-tuned* checkpoints, not a raw
pretrain one.

Each prompt gets a fresh ConversationState per checkpoint (no shared
history), so this is a single-turn Q&A comparison, matching the quiz
eval's own shape and the qualitative checks documented in
docs/expansion_PLAN.md -- not a multi-turn conversation test.

Usage
-----
    python scripts/compare_checkpoints.py \\
        --checkpoint-a checkpoints/finetune/saga-combined-v1/step_0011339.pt --label-a combined-v1 \\
        --checkpoint-b checkpoints/finetune/saga-se-qa-weighted-clean-v2/step_0007288.pt --label-b production \\
        --vocab data/tokenizer/bpe.json

    # Custom prompt set (one per line) instead of the built-in 12:
    python scripts/compare_checkpoints.py \\
        --checkpoint-a A.pt --checkpoint-b B.pt --prompts-file my_prompts.txt
"""

from __future__ import annotations

import argparse

_PROMPTS = [
    "What happens when a creature is grappled?",
    "What does the Fireball spell do?",
    "What is a cantrip?",
    "How is a monster's Challenge Rating determined?",
    "Describe what the party finds when they enter a dragon's lair.",
    "A rogue is sneaking through a crypt. What should they watch out for?",
    "What is the difference between a short rest and a long rest?",
    "What is an opportunity attack?",
    "How does advantage work in 5e?",
    "What is the armor class of a Goblin?",
    "What is a saving throw DC?",
    "What is the difference between mean and median?",
]


def _load_prompts(path: str | None) -> list[str]:
    """One prompt per line from *path*, or the built-in default list."""
    if not path:
        return _PROMPTS
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}.")
    return prompts


def _run_checkpoint(
    checkpoint: str, vocab: str, repetition_penalty: float, max_new_tokens: int, prompts: list[str],
) -> list[str]:
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.state.conversation import ConversationState

    engine = InferenceEngine(checkpoint_path=checkpoint, tokenizer_path=vocab)
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=repetition_penalty,
    )

    responses = []
    for prompt in prompts:
        state = ConversationState()  # fresh per prompt -- no cross-prompt history
        responses.append(engine.chat(prompt, state, gen_config=gen_config))
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two fine-tuned checkpoints on the same fixed prompts.",
    )
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--vocab", default="data/tokenizer/bpe.json")
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                         help="Matches agents.json's saga gen_config (default: 1.3).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompts-file", default=None, metavar="PATH",
                         help="One prompt per line. Defaults to a built-in 12-prompt list "
                              "covering D&D mechanics, narrative, stat-block facts, and a "
                              "general (non-D&D) question.")
    args = parser.parse_args()
    prompts = _load_prompts(args.prompts_file)

    print(f"Loading {args.label_a}: {args.checkpoint_a}")
    responses_a = _run_checkpoint(args.checkpoint_a, args.vocab, args.repetition_penalty, args.max_new_tokens, prompts)

    print(f"Loading {args.label_b}: {args.checkpoint_b}")
    responses_b = _run_checkpoint(args.checkpoint_b, args.vocab, args.repetition_penalty, args.max_new_tokens, prompts)

    for prompt, a, b in zip(prompts, responses_a, responses_b):
        print(f"\n=== {prompt} ===")
        print(f"[{args.label_a}] {a}")
        print(f"[{args.label_b}] {b}")


if __name__ == "__main__":
    main()
