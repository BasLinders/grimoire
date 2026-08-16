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
]


def _run_checkpoint(checkpoint: str, vocab: str, repetition_penalty: float, max_new_tokens: int) -> list[str]:
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
    for prompt in _PROMPTS:
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
    args = parser.parse_args()

    print(f"Loading {args.label_a}: {args.checkpoint_a}")
    responses_a = _run_checkpoint(args.checkpoint_a, args.vocab, args.repetition_penalty, args.max_new_tokens)

    print(f"Loading {args.label_b}: {args.checkpoint_b}")
    responses_b = _run_checkpoint(args.checkpoint_b, args.vocab, args.repetition_penalty, args.max_new_tokens)

    for prompt, a, b in zip(_PROMPTS, responses_a, responses_b):
        print(f"\n=== {prompt} ===")
        print(f"[{args.label_a}] {a}")
        print(f"[{args.label_b}] {b}")


if __name__ == "__main__":
    main()
