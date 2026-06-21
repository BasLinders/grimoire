"""Train a LoRA adapter that specialises GrimoireTransformer's own pooled
embeddings for semantic retrieval, via self-supervised contrastive learning.

See grimoire_ai/llm/training/embed_tune.py for the loss and training loop.
Corpus passages are chunked the same way build_semantic_corpus() chunks them
(grimoire_ai.llm.inference.semantic.chunk_text), so this works on any corpus
of plain-text files with no domain-specific setup.

Usage
-----
python scripts/embed_tune.py \\
    --checkpoint checkpoints/saga/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --corpus-dir data/corpus/saga/ \\
    --output     checkpoints/saga/embed.lora

Output: a .lora file (grimoire_ai.llm.model.lora.save_lora format).

Consuming the result
---------------------
Load it with grimoire_ai.llm.model.lora.load_lora() into its OWN model
instance — separate from whatever InferenceEngine is used for chat — then
pass that model's .embed method as SemanticRetriever's embed_fn. The
adapter reroutes the same q_proj/v_proj weights forward() uses, so applying
it to the model that also generates chat responses would change generation
output too, defeating the point of keeping this off the generation path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from grimoire_ai.llm.inference.semantic import chunk_text
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import save_lora
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.embed_tune import (
    EmbedTuner,
    PassageDataset,
    collate_passages,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for semantic-retrieval embeddings via "
                     "self-supervised contrastive learning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                         help="Path to a .pt checkpoint to tune (base weights stay frozen).")
    parser.add_argument("--vocab", required=True, metavar="PATH",
                         help="Path to the BPE tokenizer vocabulary JSON.")
    parser.add_argument("--corpus-dir", required=True, metavar="DIR",
                         help="Directory of .txt files to train on. No domain-specific setup needed.")
    parser.add_argument("--output", required=True, metavar="PATH",
                         help="Output .lora file path.")
    parser.add_argument("--chunk-chars", type=int, default=400, metavar="N",
                         help="Target passage size in characters (default: 400).")
    parser.add_argument("--rank", type=int, default=8, metavar="R",
                         help="LoRA rank (default: 8).")
    parser.add_argument("--alpha", type=float, default=16.0, metavar="A",
                         help="LoRA alpha; effective scale = alpha / rank (default: 16.0).")
    parser.add_argument("--targets", nargs="+", default=["q_proj", "v_proj"], metavar="NAME",
                         help="Linear layers to wrap with LoRA (default: q_proj v_proj).")
    parser.add_argument("--lr", type=float, default=1e-4, metavar="X",
                         help="AdamW learning rate (default: 1e-4).")
    parser.add_argument("--temperature", type=float, default=0.05, metavar="X",
                         help="InfoNCE softmax temperature (default: 0.05).")
    parser.add_argument("--batch-size", type=int, default=8, metavar="N",
                         help="Passages per training batch (default: 8); must be >= 2.")
    parser.add_argument("--total-steps", type=int, default=500, metavar="N",
                         help="Number of training steps (default: 500).")
    parser.add_argument("--log-every", type=int, default=25, metavar="N",
                         help="Print mean loss every N steps (default: 25).")
    args = parser.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    tokenizer = BytePairEncoder.load(args.vocab)

    txt_files = sorted(Path(args.corpus_dir).glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {args.corpus_dir}")
    passages: list[str] = []
    for txt in txt_files:
        passages.extend(chunk_text(txt.read_text(encoding="utf-8"), args.chunk_chars))
    print(f"Corpus: {len(passages)} passage(s) from {len(txt_files)} file(s)")
    if len(passages) < args.batch_size:
        raise ValueError(
            f"Only {len(passages)} passage(s) but --batch-size={args.batch_size}; "
            f"need at least batch_size passages for in-batch negatives to exist."
        )

    dataset = PassageDataset(passages, tokenizer, max_seq_len=config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_passages,
        drop_last=True,
    )

    model.add_lora_adapters(rank=args.rank, alpha=args.alpha, targets=args.targets)
    trainable = model.num_parameters(trainable_only=True)
    print(f"LoRA adapters added: rank={args.rank}, alpha={args.alpha}, "
          f"targets={args.targets}, trainable params={trainable:,}")

    tuner = EmbedTuner(model, lr=args.lr, temperature=args.temperature, device=device)
    tuner.train(loader, total_steps=args.total_steps, log_every=args.log_every)

    save_lora(model, rank=args.rank, alpha=args.alpha, targets=args.targets, path=args.output)
    print(f"\nSaved LoRA adapter to: {args.output}")


if __name__ == "__main__":
    main()
