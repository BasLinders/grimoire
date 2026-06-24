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
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, random_split

from grimoire_ai.llm.data.conversation import ConversationDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import save_lora
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.trainer import Trainer


def split_dataset(
    dataset: Dataset,
    val_split: float,
    seed: int = 42,
) -> tuple[Dataset, Optional[Dataset]]:
    """Randomly split a dataset into train and validation subsets.

    Unlike the pre-training corpus (a continuous token stream where windows
    overlap), fine-tuning examples are independent, so a plain random
    partition introduces no leakage.  The split is seeded for reproducibility,
    so the same held-out examples are used across resumed runs.

    Args:
        dataset: Any indexable ``torch.utils.data.Dataset`` (e.g.
            ``ConversationDataset``).
        val_split: Fraction of examples to hold out for validation.  Values
            ``<= 0`` disable the split and return ``(dataset, None)``.
        seed: RNG seed for the random partition.

    Returns:
        ``(train_dataset, val_dataset)``.  ``val_dataset`` is ``None`` when no
        split is requested or the dataset is too small to spare an example.
    """
    if val_split <= 0.0 or len(dataset) < 2:
        return dataset, None

    n_val = int(round(len(dataset) * val_split))
    # Always keep at least one example on each side.
    n_val = max(1, min(n_val, len(dataset) - 1))
    n_train = len(dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    return train_subset, val_subset


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
    p.add_argument("--val-split",      type=float, default=0.0,
                   help="Fraction of examples held out for validation loss (0 = disabled).")
    p.add_argument("--eval-every",     type=int,   default=0,
                   help="Compute validation loss every N steps (0 = use --save-every).")
    p.add_argument("--eval-batches",   type=int,   default=0,
                   help="Max validation batches per eval pass (0 = whole val set).")
    p.add_argument("--max-seq-len",    type=int,   default=512,
                   help="Maximum sequence length for fine-tuning examples.")
    p.add_argument("--device",         default=None,
                   help="Device override (cpu / cuda). Auto-detected if omitted.")
    p.add_argument("--lora-rank",    type=int,   default=0,
                   help="LoRA rank r. 0 (default) = full fine-tuning. Typical: 8 or 16.")
    p.add_argument("--lora-alpha",   type=float, default=16.0,
                   help="LoRA scaling alpha (default: 16.0; effective scale = alpha / rank).")
    p.add_argument("--lora-targets", type=str,   default="q_proj,v_proj",
                   help="Comma-separated Linear layer names to adapt (default: q_proj,v_proj).")
    p.add_argument("--resume-lora",  default=None,
                   help="Path to a .lora file to resume LoRA training from a previous run.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the fine-tuning loop."""
    # Trainer prints Unicode symbols (e.g. the checkpoint-saved arrow) that
    # crash with UnicodeEncodeError on Windows' default cp1252 console
    # encoding -- reconfigure unconditionally so a multi-hour run never dies
    # on a print statement, of all things.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

    train_dataset, val_dataset = split_dataset(dataset, args.val_split)
    if val_dataset is not None:
        print(f"  {len(train_dataset)} train / {len(val_dataset)} validation examples.")

    # Optional LoRA: freeze base weights, wrap target layers.
    if args.resume_lora and args.lora_rank == 0:
        print("Warning: --resume-lora is ignored when --lora-rank is 0.")

    lora_targets: list[str] = []
    on_save_lora = None
    if args.lora_rank > 0:
        lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        model.add_lora_adapters(rank=args.lora_rank, alpha=args.lora_alpha, targets=lora_targets)
        print(
            f"LoRA enabled: rank={args.lora_rank}, alpha={args.lora_alpha}, "
            f"targets={lora_targets}"
        )
        if args.resume_lora:
            from grimoire_ai.llm.model.lora import load_lora
            load_lora(model, args.resume_lora)
            print(f"  Resumed from LoRA adapter: {args.resume_lora}")
        print(f"  Trainable parameters: {model.num_parameters():,}")

        def on_save_lora(step: int, _elapsed: float) -> None:
            lora_path = Path(args.output) / f"step_{step:07d}.lora"
            save_lora(model, args.lora_rank, args.lora_alpha, lora_targets, str(lora_path))
            print(f"  → LoRA adapter saved: {lora_path}")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        accumulate_steps=args.accumulate,
        log_every=args.log_every,
        save_every=args.save_every,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        checkpoint_dir=args.output,
        device=device,
        on_save=on_save_lora,
        model_state_dict_fn=model.merged_state_dict if args.lora_rank > 0 else None,
    )

    print("Starting fine-tuning…")
    trainer.train(resume_from=None)  # always start fresh from the provided checkpoint

    if args.lora_rank > 0:
        final_lora = Path(args.output) / "lora_final.lora"
        save_lora(model, args.lora_rank, args.lora_alpha, lora_targets, str(final_lora))
        print(f"Final LoRA adapter saved: {final_lora}")

    print("Fine-tuning complete.")


if __name__ == "__main__":
    main()
