"""Instruction fine-tuning entry point.

Loads a pre-trained checkpoint and continues training on a small JSONL
dataset of structured conversation examples, teaching the model to respond
in the ``<USR>…<AST>…<EOS>`` prompt format.

Usage
-----
    python -m grimoire.llm.training.finetune \\
        --resume  checkpoints/pretrain_step_5000.pt \\
        --data    data/finetune/examples.jsonl \\
        --output  checkpoints/finetune/

All flags are optional; sensible defaults are provided for a small dataset
on consumer hardware (RTX 3050 4 GB or CPU).

Fine-tuning vs pre-training
----------------------------
The only differences from pre-training are:

1. **Dataset**: ``ConversationDataset`` instead of ``TokenizedDataset``.
   Examples are JSONL ``{user, assistant[, context]}`` objects.
2. **Loss masking**: ``ConversationDataset`` masks prompt tokens in the
   target so only response tokens contribute to the loss.
3. **Learning rate**: much lower (``5e-5`` default vs ``3e-4`` for
   pre-training) to nudge the weights rather than overwrite what
   pre-training learned.
4. **Scale**: typically hundreds of steps on a few hundred examples,
   not thousands of steps on a large corpus.

Everything else — the Trainer, checkpointing, AMP, gradient accumulation,
the model itself — is reused unchanged.
"""

import argparse
import sys

import torch

from grimoire_ai.llm.data.conversation import ConversationDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.trainer import Trainer


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instruction fine-tune a GrimoireTransformer.")
    p.add_argument("--resume",  required=True,
                   help="Path to the pre-trained checkpoint (.pt).")
    p.add_argument("--data",    required=True,
                   help="Path to the fine-tuning JSONL file.")
    p.add_argument("--vocab",   default="data/tokenizer/bpe.json",
                   help="Path to the BPE vocabulary JSON (default: data/tokenizer/bpe.json).")
    p.add_argument("--output",  default="checkpoints/finetune/",
                   help="Directory to write fine-tuned checkpoints.")
    p.add_argument("--total-steps",    type=int,   default=500)
    p.add_argument("--warmup-steps",   type=int,   default=10)
    p.add_argument("--peak-lr",        type=float, default=5e-5,
                   help="Peak learning rate (default: 5e-5, much lower than pre-training).")
    p.add_argument("--batch-size",     type=int,   default=4)
    p.add_argument("--accumulate",     type=int,   default=4,
                   help="Gradient accumulation steps.")
    p.add_argument("--log-every",      type=int,   default=25)
    p.add_argument("--save-every",     type=int,   default=100)
    p.add_argument("--max-seq-len",    type=int,   default=512,
                   help="Maximum sequence length for fine-tuning examples.")
    p.add_argument("--device",         default=None,
                   help="Device override (cpu / cuda). Auto-detected if omitted.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the fine-tuning loop."""
    args = _parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fine-tuning on device: {device}")

    # Load pre-trained checkpoint and reconstruct model.
    print(f"Loading checkpoint: {args.resume}")
    ckpt = load_checkpoint(args.resume)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])

    # Load fine-tuning dataset.
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
    tokenizer = BytePairEncoder.load(args.vocab)

    print(f"Loading fine-tuning data: {args.data}")
    dataset = ConversationDataset(
        path=args.data,
        tokenizer=tokenizer,
        max_seq_len=min(args.max_seq_len, config.max_seq_len),
    )
    print(f"  {len(dataset)} examples loaded.")

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.output,
        device=device,
    )

    print("Starting fine-tuning…")
    trainer.train(resume_from=None)  # always start fresh from the provided checkpoint
    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
