"""Factual Q&A quiz evaluator.

Runs the model on a set of domain questions and scores the responses
using two complementary metrics:

    keyword_recall  — fraction of expected keywords present in the response
                      (case-insensitive substring match).  A "pass" requires
                      recall ≥ ``pass_threshold`` (default 0.5).

    token_f1        — token-level overlap between the reference answer and
                      the model response (macro-averaged F1 over unigrams).
                      Robust to paraphrasing; directly comparable to the
                      standard SQuAD evaluation metric.

Quiz file format (JSONL)
------------------------
Each line must be a JSON object with at least:

    {
        "user": "What is the proficiency bonus at level 7?",
        "assistant": "At level 7 the proficiency bonus is +3.",
        "keywords": ["+3", "proficiency"]
    }

``keywords`` is required for keyword_recall.  ``assistant`` is used for
token_f1 (optional — omit the field to skip token_f1 for that item).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between *prediction* and *reference* (SQuAD standard)."""
    from collections import Counter
    pred_tokens = _tokenize(prediction)
    ref_tokens  = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    # Use multiset (Counter) intersection so repeated tokens are counted
    # correctly — a set intersection would over-count recall for repeated words.
    common_counts = Counter(pred_tokens) & Counter(ref_tokens)
    n_common = sum(common_counts.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _keyword_recall(prediction: str, keywords: list[str]) -> float:
    """Fraction of keywords found in *prediction* (case-insensitive)."""
    if not keywords:
        return 1.0
    pred_lower = prediction.lower()
    return sum(kw.lower() in pred_lower for kw in keywords) / len(keywords)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_quiz(path: str) -> list[dict]:
    """Load a JSONL quiz file and return a list of example dicts."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Quiz file line {lineno}: {exc}") from exc
            if "user" not in obj:
                raise ValueError(f"Quiz file line {lineno}: missing 'user' field")
            examples.append(obj)
    return examples


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def eval_quiz(
    engine: "InferenceEngine",
    examples: list[dict],
    gen_config: Optional["GenerationConfig"] = None,
    pass_threshold: float = 0.5,
    on_progress: Optional[Callable[[str], None]] = None,
    stop_event: Optional[threading.Event] = None,
) -> dict:
    """Run the model on each quiz question and score the responses.

    Args:
        engine: A loaded ``InferenceEngine``.
        examples: List of quiz dicts with ``user``, optionally ``assistant``
            (reference answer) and ``keywords``.
        gen_config: Generation config override.  Falls back to a capped
            default (128 new tokens) to keep evaluation fast.
        pass_threshold: Minimum keyword_recall to count as a pass.
        on_progress: Optional callback for log lines.
        stop_event: When set, the question loop exits early and returns the
            results accumulated so far rather than running to completion.
            Checked between questions — generation for the current question
            still runs to completion since it's a single blocking call.

    Returns:
        Dict with aggregate metrics and per-question ``results`` list.
    """
    from grimoire_ai.llm.inference.sampler import GenerationConfig

    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    if not examples:
        _log("  ⚠  No quiz examples — skipping quiz eval.")
        return {
            "pass_rate": float("nan"),
            "mean_keyword_recall": float("nan"),
            "mean_token_f1": float("nan"),
            "passes": 0,
            "total": 0,
            "results": [],
        }

    # Use a capped config so the eval doesn't run forever.
    if gen_config is None:
        gen_config = GenerationConfig(max_new_tokens=128, temperature=0.0, top_k=1, top_p=1.0)

    _log(f"Quiz eval: {len(examples)} questions …")

    results: list[dict] = []
    passes = 0
    total_kw_recall = 0.0
    total_f1 = 0.0
    n_f1 = 0

    for i, ex in enumerate(examples):
        if stop_event is not None and stop_event.is_set():
            _log(f"  ⏹  Stopped early at question {i}/{len(examples)}.")
            break

        question   = ex["user"]
        reference  = ex.get("assistant", "")
        keywords   = ex.get("keywords", [])

        response = engine.respond(question, gen_config=gen_config)

        kw_recall = _keyword_recall(response, keywords)
        passed    = kw_recall >= pass_threshold
        if passed:
            passes += 1
        total_kw_recall += kw_recall

        f1 = float("nan")
        if reference:
            f1 = _token_f1(response, reference)
            total_f1 += f1
            n_f1 += 1

        results.append({
            "question": question,
            "response": response,
            "reference": reference,
            "keywords": keywords,
            "keyword_recall": round(kw_recall, 4),
            "token_f1": round(f1, 4) if not (f1 != f1) else None,  # nan → None
            "pass": passed,
        })

        if i == 0 or (i + 1) % 5 == 0 or i == len(examples) - 1:
            _log(
                f"  {i+1}/{len(examples)}  passes: {passes}  "
                f"kw_recall: {total_kw_recall/(i+1):.2%}"
            )

    # Use len(results) rather than len(examples) so a stop_event that cuts
    # the loop short doesn't divide by the original (larger) question count.
    total = len(results)
    mean_kw = total_kw_recall / total if total else 0.0
    mean_f1 = total_f1 / n_f1 if n_f1 else float("nan")
    pass_rate = passes / total if total else 0.0

    _log(
        f"  pass_rate {pass_rate:.1%}  mean_kw_recall {mean_kw:.2%}  "
        f"mean_token_f1 {mean_f1:.4f}"
    )

    return {
        "pass_rate": round(pass_rate, 4),
        "mean_keyword_recall": round(mean_kw, 4),
        "mean_token_f1": round(mean_f1, 4) if not (mean_f1 != mean_f1) else None,
        "passes": passes,
        "total": total,
        "results": results,
    }
