"""Fit the scoped kNN-LM gate and report the go/no-go check.

PR 3 in docs/KNN_LM.md. Builds inference-shaped examples
(``<BOS><SEP>{context}<SEP><USR>{question}<AST>{answer}<EOS>``), caches
per-answer-token kNN features with one forward pass per example, fits the
gate on a training split, and compares held-out NLL with and without kNN.

Context source
--------------
- ``--corpus-dir`` given (recommended): context is *retrieved* for each
  question through ``InferenceEngine._retrieve`` with the model's own
  embeddings, exactly as at inference. The gate then also learns to back off
  when retrieval returns something irrelevant.
- Otherwise: each record's own ``"context"`` field is used. Records without
  one are skipped, and so are records whose answer appears verbatim in their
  context (some Stack Exchange extractions set ``context`` to the answer
  itself, which would turn the fit into a copying test). Gold context is
  cleaner than real retrieval, so a gate fit this way tends to trust the
  store more than it should.

Data hygiene
------------
Fit on records the model was NOT fine-tuned on (pass the SFT files via
``--exclude`` and any overlapping questions are dropped). The quiz file is
always excluded: it's the test set.

Usage
-----
python scripts/eval/fit_knn_gate.py \\
    --checkpoint checkpoints/finetune/step_0000500.pt \\
    --vocab      data/tokenizer/bpe.json \\
    --data       data/finetune/heldout_qa.jsonl \\
    --exclude    scripts/finetune/data/saga_v1.jsonl scripts/finetune/data/saga_v2.jsonl \\
    --corpus-dir data/corpus/saga/ --corpus-limit 300

Writes ``<checkpoint>.knn_gate.pt`` (override with ``--out``) unless
``--no-save``. Exit code 0 on a PASS verdict, 2 on FAIL.
"""

import argparse
import json
import random
import sys
from pathlib import Path

_DEFAULT_QUIZ = Path(__file__).resolve().parent / "data" / "saga_quiz.jsonl"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _question(rec: dict) -> str:
    return rec.get("user") or rec.get("prompt") or ""


def _answer(rec: dict) -> str:
    return rec.get("assistant") or rec.get("response") or ""


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(
        description="Fit the scoped kNN-LM gate (docs/KNN_LM.md, PR 3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH")
    p.add_argument("--vocab", required=True, metavar="PATH")
    p.add_argument("--data", required=True, nargs="+", metavar="JSONL",
                   help="Held-out {user, assistant[, context]} records.")
    p.add_argument("--exclude", nargs="*", default=[], metavar="JSONL",
                   help="Files whose questions must not be fitted on (SFT data). "
                        "The quiz file is always excluded.")
    p.add_argument("--quiz", default=str(_DEFAULT_QUIZ), metavar="JSONL",
                   help="Quiz/test file to exclude (default: scripts/eval/data/saga_quiz.jsonl).")
    p.add_argument("--corpus-dir", default="", metavar="DIR",
                   help="Retrieve context per question from this corpus (model embeddings).")
    p.add_argument("--corpus-limit", type=int, default=0, metavar="N",
                   help="Sample N corpus files with a fixed seed (0 = all). Same as evaluate.py.")
    p.add_argument("--top-k-corpus", type=int, default=5, metavar="N",
                   help="Passages retrieved per question (default 5, same as the engine).")
    p.add_argument("--index-batch-size", type=int, default=32, metavar="N")
    p.add_argument("--gen-lora", default="", metavar="PATH",
                   help="Generation LoRA to apply. Fit with the adapter you'll generate with.")
    p.add_argument("--k", type=int, default=16, help="Neighbors per position (default 16).")
    p.add_argument("--lam-max", type=float, default=0.8, help="Cap on λ (default 0.8).")
    p.add_argument("--val-frac", type=float, default=0.2, help="Held-out example share (default 0.2).")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-examples", type=int, default=0, metavar="N",
                   help="Use at most N records, sampled with --seed (0 = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-delta", type=float, default=0.0, metavar="NATS",
                   help="PASS requires fused NLL below LM NLL by more than this (default 0).")
    p.add_argument("--out", default="", metavar="PATH",
                   help="Gate output path (default: <checkpoint>.knn_gate.pt).")
    p.add_argument("--no-save", action="store_true", help="Report only; don't write the gate.")
    p.add_argument("--allow-answer-in-context", action="store_true",
                   help="Record-context mode only: keep records whose answer appears "
                        "verbatim in their context (dropped by default).")
    args = p.parse_args()

    from grimoire_ai.corpus.corpus import QueryResult
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.knn_lm import KNNConfig, checkpoint_tag, save_gate
    from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID
    from grimoire_ai.llm.training.knn_gate import (
        GateExample,
        collect_features,
        fit_gate,
        gate_report,
        split_by_example,
    )

    # ---- records -----------------------------------------------------------
    records = [r for path in args.data for r in _load_jsonl(path)]
    excluded: set[str] = set()
    for path in [*args.exclude, *([args.quiz] if args.quiz and Path(args.quiz).exists() else [])]:
        excluded |= {_norm(_question(r)) for r in _load_jsonl(path)}
    before = len(records)
    records = [r for r in records
               if _question(r) and _answer(r) and _norm(_question(r)) not in excluded]
    print(f"Records: {before} loaded, {before - len(records)} dropped "
          f"(excluded question or empty), {len(records)} kept.")
    if args.max_examples and len(records) > args.max_examples:
        records = random.Random(args.seed).sample(records, args.max_examples)
        print(f"Sampled {len(records)} record(s) (--max-examples).")

    # ---- engine + optional retrieval --------------------------------------
    print("Loading model …")
    engine = InferenceEngine(checkpoint_path=args.checkpoint, tokenizer_path=args.vocab)
    print(f"Device: {engine.device}")
    if args.gen_lora:
        print(f"Loading generation LoRA: {args.gen_lora} …")
        engine.load_lora(args.gen_lora)

    use_retrieval = bool(args.corpus_dir.strip())
    if use_retrieval:
        txt_paths = sorted(Path(args.corpus_dir).glob("*.txt"))
        if args.corpus_limit and len(txt_paths) > args.corpus_limit:
            txt_paths = random.Random(0).sample(txt_paths, args.corpus_limit)
        documents = [(t.read_text(encoding="utf-8"), t.stem) for t in txt_paths]
        if not documents:
            sys.exit(f"No .txt files in --corpus-dir {args.corpus_dir}")
        print(f"Building semantic corpus from {len(documents)} file(s) (model embeddings) …")
        engine.build_semantic_corpus(documents, batch_size=args.index_batch_size)
        print("Context source: retrieved (top-%d)." % args.top_k_corpus)
    else:
        print("Context source: record 'context' field (no --corpus-dir).")
        if not args.allow_answer_in_context:
            n = len(records)
            records = [r for r in records
                       if _norm(_answer(r)) not in _norm(r.get("context", ""))]
            if n - len(records):
                print(f"Dropped {n - len(records)} record(s) whose answer appears verbatim "
                      "in their context (pure copying would flatter kNN; "
                      "--allow-answer-in-context to keep).")

    # ---- examples ---------------------------------------------------------
    examples: list[GateExample] = []
    for rec in records:
        q = _question(rec)
        if use_retrieval:
            results = engine._retrieve(q, top_k=args.top_k_corpus)
        elif rec.get("context"):
            results = [QueryResult(multi_token=(), next_token=None, score=1.0,
                                   source=None, excerpt=rec["context"])]
        else:
            results = []
        prompt_ids = engine.prompt_builder.build(q, results)
        answer_ids = engine.tokenizer.encode(_answer(rec)) + [EOS_ID]
        examples.append(GateExample(prompt_ids, answer_ids))

    cfg = KNNConfig(k=args.k, lam_max=args.lam_max)
    print(f"Collecting features for {len(examples)} example(s) (k={cfg.k}) …")
    feats = collect_features(engine.model, examples, cfg, device=engine.device, on_progress=print)
    print(f"  {feats.n_examples} usable, {feats.n_skipped} skipped (no context block / no room), "
          f"{feats.n_positions} answer tokens.")

    # ---- fit + report -----------------------------------------------------
    train, val = split_by_example(feats, args.val_frac, args.seed)
    print(f"Fitting gate on {int(train.sum())} tokens, holding out {int(val.sum())} …")
    gate = fit_gate(feats, train, cfg, epochs=args.epochs, lr=args.lr, on_progress=print)

    tr, va = gate_report(gate, feats, train, cfg.eps), gate_report(gate, feats, val, cfg.eps)
    print("\n                 train        val")
    for key in ("n_tokens", "lm_nll", "fused_nll", "delta_nll", "mean_lambda", "coverage"):
        fmt = "{:>12d}" if key == "n_tokens" else "{:>12.4f}"
        print(f"{key:<12}" + fmt.format(tr[key]) + fmt.format(va[key]))
    print(f"temperature {gate.temperature.item():.3f}   "
          f"weights {[round(w, 4) for w in gate.lin.weight[0].tolist()]}   "
          f"bias {gate.lin.bias.item():.4f}")

    passed = va["delta_nll"] < -args.min_delta
    print("\nVERDICT: " + (
        f"PASS — kNN lowers held-out NLL by {-va['delta_nll']:.4f} nats/token."
        if passed else
        f"FAIL — kNN does not lower held-out NLL (Δ = {va['delta_nll']:+.4f}). "
        "Check retrieval quality and coverage before tuning further (docs/KNN_LM.md)."
    ))

    if not args.no_save:
        out = Path(args.out) if args.out else Path(args.checkpoint + ".knn_gate.pt")
        save_gate(out, gate, cfg, checkpoint_tag(args.checkpoint))
        print(f"Gate saved: {out}")
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
