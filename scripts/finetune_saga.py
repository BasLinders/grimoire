"""Fine-tune a pre-trained Grimoire checkpoint on the Saga instruction dataset.

Loads a pre-trained checkpoint, wraps the Saga JSONL dataset in a
ConversationDataset, and runs a short fine-tuning pass using the
Trainer with response-only loss masking.

Usage
-----
    python scripts/finetune_saga.py \\
        --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \\
        --vocab      data/tokenizer/bpe.json \\
        --data       data/finetune/saga_v1.jsonl \\
        --output-dir checkpoints/saga/

Output: checkpoints/saga/step_XXXXXXX.pt

The script prints per-step loss so you can watch convergence.
Typical fine-tune run: 200-500 steps on this dataset, though a large
auto-derived dataset (e.g. scripts/build_finetune_data_from_qa.py's output)
needs proportionally more -- see that script's docstring.

Resuming an interrupted run
----------------------------
--save-every already writes a full Trainer checkpoint (model + optimizer +
LR-scheduler state) every N steps to --output-dir. If the run dies partway
through (interruption, crash, sleep), rerun with --resume pointing at the
latest step_XXXXXXX.pt instead of restarting from --checkpoint:

    python scripts/finetune_saga.py \\
        --checkpoint checkpoints/pretrain/step_XXXXXXX.pt \\
        --vocab      data/tokenizer/bpe.json \\
        --data       data/finetune/saga_v1.jsonl \\
        --output-dir checkpoints/saga/ \\
        --resume     checkpoints/saga/step_0000200.pt \\
        --total-steps 300

--total-steps is the absolute target (resuming at step 200 toward
total-steps=300 runs 100 more steps, not 300 more) -- --checkpoint is still
required to determine the model architecture, but its weights are
immediately overwritten by --resume's, so it doesn't need to be the exact
same file you started from.
"""

from __future__ import annotations

import argparse
import sys

from grimoire_ai.llm.data.conversation import ConversationDataset
from grimoire_ai.llm.device import select_device
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.trainer import Trainer


def main(argv: list[str] | None = None) -> None:
    # Trainer prints Unicode symbols (e.g. the checkpoint-saved arrow) that
    # crash with UnicodeEncodeError on Windows' default cp1252 console
    # encoding -- reconfigure unconditionally so a multi-hour run never dies
    # on a print statement, of all things.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Fine-tune a Grimoire checkpoint on the Saga instruction dataset."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the pre-trained .pt checkpoint.",
    )
    parser.add_argument(
        "--vocab", required=True,
        help="Path to the BPE tokenizer .json file.",
    )
    parser.add_argument(
        "--data", default="scripts/finetune_data/saga_v1.jsonl",
        help="Path to the JSONL fine-tuning dataset (default: scripts/finetune_data/saga_v1.jsonl).",
    )
    parser.add_argument(
        "--output-dir", default="checkpoints/saga/",
        help="Directory for fine-tuned checkpoints (default: checkpoints/saga/).",
    )
    parser.add_argument("--total-steps",    type=int,   default=300)
    parser.add_argument("--warmup-steps",   type=int,   default=20)
    parser.add_argument("--peak-lr",        type=float, default=5e-5)
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument("--accum-steps",    type=int,   default=4)
    parser.add_argument("--log-every",      type=int,   default=25)
    parser.add_argument("--save-every",     type=int,   default=100)
    parser.add_argument("--max-seq-len",    type=int,   default=512)
    parser.add_argument(
        "--resume", default=None, metavar="PATH",
        help="Resume an interrupted run from a step_XXXXXXX.pt checkpoint written by a "
             "previous --save-every (restores model, optimizer, and LR-scheduler state). "
             "--total-steps is the absolute target, not an additional count.",
    )
    args = parser.parse_args(argv)

    device = select_device()
    print(f"Device: {device}")

    # Load checkpoint and reconstruct model.
    ckpt = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Model config: vocab={config.vocab_size}, d_model={config.d_model}, layers={config.n_layers}")

    # Tokenizer.
    tokenizer = BytePairEncoder.load(args.vocab)

    # Dataset.
    seq_len = min(args.max_seq_len, config.max_seq_len)
    dataset = ConversationDataset(
        path=args.data,
        tokenizer=tokenizer,
        max_seq_len=seq_len,
    )
    print(f"Dataset: {len(dataset)} examples from {args.data}")

    # Fine-tune.
    def on_log(step: int, loss: float, lr: float, elapsed: float) -> None:
        print(f"  step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")

    Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        peak_lr=args.peak_lr,
        batch_size=args.batch_size,
        accumulate_steps=args.accum_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.output_dir,
        device=device,
        on_log=on_log,
    ).train(resume_from=args.resume)

    print(f"\nFine-tuning complete. Checkpoints saved to: {args.output_dir}")
    print("Update agents.json to point 'saga' checkpoint to the latest .pt file.")


if __name__ == "__main__":
    main()
