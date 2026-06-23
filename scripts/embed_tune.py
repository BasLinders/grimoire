"""Train a LoRA adapter that specialises GrimoireTransformer's own pooled
embeddings for semantic retrieval, via contrastive learning.

See grimoire_ai/llm/training/embed_tune.py for the losses and training loops.

Two modes
---------
Self-supervised (default): corpus passages are chunked the same way
build_semantic_corpus() chunks them (grimoire_ai.llm.inference.semantic.
chunk_text), so this works on any corpus of plain-text files with no
domain-specific setup. The positive pair is two dropout views of the SAME
passage (SimCSE) -- this only ever teaches "is this the same content", never
"does this passage answer this query".

Supervised (--qa-corpus-dir): trains on real (question, answer) pairs from a
StackExchange-style Q&A corpus (grimoire_ai.llm.data.qa_pairs.load_qa_pairs),
directly optimising the actual retrieval task instead of a same-passage
proxy. Use this when real query/relevant-passage pairs are available --
self-supervised training on the SAME corpus measurably underperforms it.

Usage
-----
# Self-supervised, any corpus of .txt files:
python scripts/embed_tune.py \\
    --checkpoint checkpoints/saga/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --corpus-dir data/corpus/saga/ \\
    --output     checkpoints/saga/embed.lora

# Supervised, real Q&A pairs:
python scripts/embed_tune.py \\
    --checkpoint checkpoints/saga/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --qa-corpus-dir data/corpus/saga/ \\
    --output     checkpoints/saga/embed-qa.lora

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

from grimoire_ai.llm.data.qa_pairs import load_qa_pairs
from grimoire_ai.llm.inference.semantic import chunk_text
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import save_lora
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.checkpoint import load_checkpoint
from grimoire_ai.llm.training.embed_tune import (
    DocumentGroupedBatchSampler,
    EmbedTuner,
    PassageDataset,
    QAPairDataset,
    collate_passages,
    collate_qa_pairs,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train a LoRA adapter for semantic-retrieval embeddings via contrastive "
                     "learning (self-supervised --corpus-dir or supervised --qa-corpus-dir).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                         help="Path to a .pt checkpoint to tune (base weights stay frozen).")
    parser.add_argument("--vocab", required=True, metavar="PATH",
                         help="Path to the BPE tokenizer vocabulary JSON.")
    parser.add_argument("--corpus-dir", default="", metavar="DIR",
                         help="Self-supervised mode: directory of .txt files to train on "
                              "(same-passage dropout pairs). No domain-specific setup needed. "
                              "Mutually exclusive with --qa-corpus-dir.")
    parser.add_argument("--qa-corpus-dir", default="", metavar="DIR",
                         help="Supervised mode: directory of StackExchange-format Q&A .txt "
                              "files (rpg_se_*.txt) -- see grimoire_ai.llm.data.qa_pairs. "
                              "Trains on real (question, answer) pairs instead of same-passage "
                              "dropout pairs, directly optimising the retrieval task itself. "
                              "Mutually exclusive with --corpus-dir.")
    parser.add_argument("--qa-min-score", type=int, default=1, metavar="N",
                         help="Supervised mode: minimum answer score to keep (default: 1). "
                              "Ignored when --qa-accepted-only is set.")
    parser.add_argument("--qa-accepted-only", action="store_true",
                         help="Supervised mode: keep only each question's accepted answer, "
                              "ignoring --qa-min-score.")
    parser.add_argument("--output", required=True, metavar="PATH",
                         help="Output .lora file path.")
    parser.add_argument("--chunk-chars", type=int, default=400, metavar="N",
                         help="Self-supervised mode: target passage size in characters "
                              "(default: 400).")
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
                         help="Items per training batch (default: 8); must be >= 2. In "
                              "self-supervised mode, must also be a multiple of "
                              "--passages-per-doc.")
    parser.add_argument("--passages-per-doc", type=int, default=4, metavar="N",
                         help="Self-supervised mode only: passages sampled from each document "
                              "per batch (default: 4). Each batch draws "
                              "batch-size/passages-per-doc documents, so in-batch negatives "
                              "include same-document near-misses (hard) alongside "
                              "cross-document negatives (easy) -- pure random batching only "
                              "ever produces easy negatives, which is why an earlier run of "
                              "this script confused things like 'advantage' with 'disadvantage' "
                              "despite a low training loss.")
    parser.add_argument("--total-steps", type=int, default=500, metavar="N",
                         help="Number of training steps (default: 500).")
    parser.add_argument("--log-every", type=int, default=25, metavar="N",
                         help="Print mean loss every N steps (default: 25).")
    args = parser.parse_args(argv)

    if bool(args.corpus_dir) == bool(args.qa_corpus_dir):
        raise ValueError(
            "Pass exactly one of --corpus-dir (self-supervised) or --qa-corpus-dir "
            "(supervised), not both and not neither."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ckpt = load_checkpoint(args.checkpoint)
    config = TransformerConfig.from_dict(ckpt["config"])
    model = GrimoireTransformer(config)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    tokenizer = BytePairEncoder.load(args.vocab)

    if args.qa_corpus_dir:
        qa_pairs = load_qa_pairs(
            args.qa_corpus_dir, min_score=args.qa_min_score, accepted_only=args.qa_accepted_only,
        )
        if not qa_pairs:
            raise ValueError(f"No Q&A pairs found in {args.qa_corpus_dir}")
        print(f"Q&A corpus: {len(qa_pairs)} pair(s) from {args.qa_corpus_dir}")
        if len(qa_pairs) < args.batch_size:
            raise ValueError(
                f"Only {len(qa_pairs)} pair(s) but --batch-size={args.batch_size}; "
                f"need at least batch_size pairs for in-batch negatives to exist."
            )

        pairs = [(p.question, p.answer) for p in qa_pairs]
        dataset = QAPairDataset(pairs, tokenizer, max_seq_len=config.max_seq_len)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_qa_pairs, drop_last=True,
        )
    else:
        txt_files = sorted(Path(args.corpus_dir).glob("*.txt"))
        if not txt_files:
            raise ValueError(f"No .txt files found in {args.corpus_dir}")
        passages: list[str] = []
        doc_ids: list[int] = []
        for doc_id, txt in enumerate(txt_files):
            chunks = chunk_text(txt.read_text(encoding="utf-8"), args.chunk_chars)
            passages.extend(chunks)
            doc_ids.extend([doc_id] * len(chunks))
        print(f"Corpus: {len(passages)} passage(s) from {len(txt_files)} file(s)")
        if len(passages) < args.batch_size:
            raise ValueError(
                f"Only {len(passages)} passage(s) but --batch-size={args.batch_size}; "
                f"need at least batch_size passages for in-batch negatives to exist."
            )

        dataset = PassageDataset(passages, tokenizer, max_seq_len=config.max_seq_len)
        sampler = DocumentGroupedBatchSampler(
            doc_ids,
            batch_size=args.batch_size,
            passages_per_doc=args.passages_per_doc,
            num_batches=args.total_steps,
        )
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_passages)

    model.add_lora_adapters(rank=args.rank, alpha=args.alpha, targets=args.targets)
    trainable = model.num_parameters(trainable_only=True)
    print(f"LoRA adapters added: rank={args.rank}, alpha={args.alpha}, "
          f"targets={args.targets}, trainable params={trainable:,}")

    tuner = EmbedTuner(model, lr=args.lr, temperature=args.temperature, device=device)
    if args.qa_corpus_dir:
        tuner.train_pairs(loader, total_steps=args.total_steps, log_every=args.log_every)
    else:
        tuner.train(loader, total_steps=args.total_steps, log_every=args.log_every)

    save_lora(model, rank=args.rank, alpha=args.alpha, targets=args.targets, path=args.output)
    print(f"\nSaved LoRA adapter to: {args.output}")


if __name__ == "__main__":
    main()
