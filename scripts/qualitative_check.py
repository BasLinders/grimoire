"""Qualitative generation check for a raw (not yet fine-tuned) pretrain checkpoint.

Neither of this project's two chat interfaces fit this job:
- grimoire-chat (grimoire_ai/cli/chat.py) has no --repetition-penalty flag
  at all (its GenerationConfig only sets max_new_tokens/temperature/top_k/
  top_p), so it can't reproduce the repetition_penalty=1.3 condition this
  project's prior qualitative checks (docs/expansion_PLAN.md) used.
- Both chat.py and the Chat tab wrap queries in a conversational prompt
  template meant for a *fine-tuned* model; a raw pretrain checkpoint like
  checkpoints/pretrain/weighted_clean_v2 was never trained to follow that
  format; a raw text-completion prompt is the fair test.

Uses grimoire_ai.llm.inference.sampler.generate() directly -- no chat
templating, no conversation state, just a prompt string in and a
continuation out, exactly what a completion-only quality check needs.

The prompt list below is written in the same spirit as the topics named
in docs/expansion_PLAN.md's qualitative checks (a spell, a dungeon/lair
scene, a narrative hook, a rules-mechanics question, a condition, a
cantrip) -- the original literal prompt text was never committed
anywhere, so these are equivalent-shaped stand-ins, not a byte-for-byte
replay of the historical ones.

Usage
-----
    python scripts/qualitative_check.py \\
        --checkpoint checkpoints/pretrain/weighted_clean_v2/step_0015259.pt \\
        --vocab data/tokenizer/bpe.json
"""

from __future__ import annotations

import argparse

import torch

from grimoire_ai.llm.inference.sampler import GenerationConfig, generate
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.checkpoint import load_checkpoint

_PROMPTS = [
    ("spell",      "Fireball is a"),
    ("dungeon",    "The party descends into the dragon's lair, and"),
    ("narrative",  "The rogue creeps through the crypt, listening for"),
    ("mechanics",  "A creature's Challenge Rating determines"),
    ("condition",  "When a creature is grappled, it"),
    ("cantrip",    "Fire Bolt is a cantrip that"),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed-prompt completions from a checkpoint to eyeball coherence.",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt file.")
    parser.add_argument("--vocab", default="data/tokenizer/bpe.json",
                         help="Path to the BPE vocabulary JSON.")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3,
                         help="Matches this project's documented qualitative-check "
                              "convention (default: 1.3).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = BytePairEncoder.load(args.vocab)

    ckpt = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    print(f"Loaded {args.checkpoint} (step {ckpt.get('step', '?')})\n")

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    for label, prompt in _PROMPTS:
        prompt_ids = tokenizer.encode(prompt)
        generated_ids = generate(model, prompt_ids, config=gen_config, device=device)
        completion = tokenizer.decode(generated_ids)
        print(f"=== {label} ===")
        print(f"Prompt: {prompt}")
        print(f"Completion: {completion}")
        print()


if __name__ == "__main__":
    main()
