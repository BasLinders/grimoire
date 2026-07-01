"""Top-level evaluation harness.

Orchestrates all three evaluators (perplexity, retrieval, quiz) and
writes a timestamped JSON report to ``data/eval/``.

Typical usage (CLI)
-------------------
    python scripts/evaluate.py \\
        --checkpoint checkpoints/finetune/step_0000500.pt \\
        --vocab      data/tokenizer/bpe.json \\
        --corpus-dir data/corpus/saga/ \\
        --quiz       scripts/eval_data/saga_quiz.jsonl \\
        --corpus-bin data/processed/corpus.bin

Typical usage (Python)
----------------------
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.eval.harness import run_eval

    engine = InferenceEngine(checkpoint_path=..., tokenizer_path=...)
    engine.build_semantic_corpus([...])
    results = run_eval(engine=engine, quiz_path="scripts/eval_data/saga_quiz.jsonl")
    print(results["summary"])
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from grimoire_ai.llm.inference.engine import InferenceEngine


def run_eval(
    engine: Optional["InferenceEngine"] = None,
    corpus_bin: Optional[str] = None,
    quiz_path: Optional[str] = None,
    output_dir: str = "data/eval",
    retrieval_queries: Optional[list[dict]] = None,
    max_perplexity_batches: int = 50,
    perplexity_batch_size: int = 4,
    quiz_repetition_penalty: float = 1.0,
    on_progress: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Run enabled evaluators and write a JSON report.

    At least one of ``engine`` (for quiz + retrieval) or ``corpus_bin``
    (for perplexity) must be provided.

    Args:
        engine: Loaded ``InferenceEngine``.  Required for quiz and retrieval
            evals.  Perplexity uses ``engine.model`` when ``corpus_bin`` is
            also provided.
        corpus_bin: Path to a ``.bin`` corpus file for perplexity eval.
        quiz_path: Path to a JSONL quiz file.  If ``None`` and a default
            exists at ``scripts/eval_data/saga_quiz.jsonl``, it is used.
        output_dir: Directory for the JSON report.  Created if absent.
        retrieval_queries: Custom query set for retrieval eval.  Falls back
            to ``SAGA_QUERIES`` when ``None``.
        max_perplexity_batches: Cap on batches for perplexity eval.
        perplexity_batch_size: Batch size for perplexity eval.
        quiz_repetition_penalty: Multiplicative penalty on previously-generated
            tokens during quiz generation. ``1.0`` (default) disables it,
            matching the quiz's standalone default. Values > 1.0 discourage
            the model from repeating itself — useful for isolating how much
            of a repetition-loop pattern is a decoding-time effect versus
            something that needs more training to fix.
        on_progress: Optional progress callback.
        stop_event: When set, each evaluator stops as soon as it notices
            (between batches/questions/queries) and any evaluator not yet
            started is skipped entirely. The report is still written with
            whatever evaluators completed or partially completed.

    Returns:
        Full results dict including ``summary`` and ``timestamp``.
    """
    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def _stopped() -> bool:
        return stop_event is not None and stop_event.is_set()

    if engine is None and corpus_bin is None:
        raise ValueError("Provide at least one of 'engine' or 'corpus_bin'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: dict = {"timestamp": timestamp, "evals": {}}

    # -----------------------------------------------------------------------
    # 1. Perplexity
    # -----------------------------------------------------------------------
    if _stopped():
        _log("⏹  Stopped before perplexity eval.")
    elif corpus_bin and Path(corpus_bin).is_file():
        _log("─── Perplexity eval ───────────────────────────────────")
        from grimoire_ai.llm.eval.perplexity import eval_perplexity
        import torch

        model = engine.model if engine is not None else None
        if model is None:
            _log("  ⚠  No model available — skipping perplexity eval.")
        else:
            device = engine.device if engine is not None else ("cuda" if torch.cuda.is_available() else "cpu")
            seq_len = engine.model.config.max_seq_len if engine is not None else 1024
            ppl_result = eval_perplexity(
                model=model,
                corpus_path=corpus_bin,
                seq_len=seq_len,
                batch_size=perplexity_batch_size,
                max_batches=max_perplexity_batches,
                device=device,
                on_progress=on_progress,
                stop_event=stop_event,
            )
            results["evals"]["perplexity"] = ppl_result
    else:
        if corpus_bin:
            _log(f"  ⚠  Corpus binary not found: {corpus_bin} — skipping perplexity.")
        else:
            _log("  ⚠  No corpus binary provided — skipping perplexity eval.")

    # -----------------------------------------------------------------------
    # 2. Retrieval hit-rate
    # -----------------------------------------------------------------------
    if _stopped():
        _log("⏹  Stopped before retrieval eval.")
    elif engine is not None and engine.corpus is not None:
        _log("─── Retrieval eval ────────────────────────────────────")
        from grimoire_ai.llm.eval.retrieval import eval_retrieval
        ret_result = eval_retrieval(
            engine=engine,
            queries=retrieval_queries,
            on_progress=on_progress,
            stop_event=stop_event,
        )
        results["evals"]["retrieval"] = ret_result
    elif engine is not None:
        _log("  ⚠  No corpus attached to engine — skipping retrieval eval.")

    # -----------------------------------------------------------------------
    # 3. Quiz
    # -----------------------------------------------------------------------
    if _stopped():
        _log("⏹  Stopped before quiz eval.")
    elif engine is not None:
        # Resolve quiz path: explicit → default saga quiz → skip.
        _quiz_path = quiz_path
        if not _quiz_path:
            default = Path("scripts/eval_data/saga_quiz.jsonl")
            if default.is_file():
                _quiz_path = str(default)

        if _quiz_path and Path(_quiz_path).is_file():
            _log("─── Quiz eval ─────────────────────────────────────────")
            from grimoire_ai.llm.eval.quiz import eval_quiz, load_quiz
            examples = load_quiz(_quiz_path)
            quiz_gen_config = None
            if quiz_repetition_penalty != 1.0:
                from grimoire_ai.llm.inference.sampler import GenerationConfig
                quiz_gen_config = GenerationConfig(
                    max_new_tokens=128, temperature=0.0, top_k=1, top_p=1.0,
                    repetition_penalty=quiz_repetition_penalty,
                )
            quiz_result = eval_quiz(
                engine=engine, examples=examples, gen_config=quiz_gen_config,
                on_progress=on_progress, stop_event=stop_event,
            )
            results["evals"]["quiz"] = quiz_result
        else:
            _log("  ⚠  No quiz file found — skipping quiz eval.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    summary_lines = []
    for name, res in results["evals"].items():
        if name == "perplexity":
            summary_lines.append(
                f"perplexity={res['perplexity']:.2f}  BPC={res['bpc']:.4f}"
            )
        elif name == "retrieval":
            summary_lines.append(
                f"retrieval hit-rate={res['hit_rate']:.1%} ({res['hits']}/{res['total']})"
            )
        elif name == "quiz":
            summary_lines.append(
                f"quiz pass-rate={res['pass_rate']:.1%}  kw-recall={res['mean_keyword_recall']:.2%}"
            )
    results["summary"] = "  |  ".join(summary_lines) if summary_lines else "No evals ran."
    results["stopped_early"] = _stopped()

    # -----------------------------------------------------------------------
    # Write report
    # -----------------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"\nReport saved → {out_path}")
    _log(f"Summary: {results['summary']}")

    return results
