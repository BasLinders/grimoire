"""Factual Q&A quiz evaluator.

Runs the model on a set of domain questions and scores the responses
using two complementary metrics:

    keyword_recall  — fraction of expected keywords present in the response
                      (case-insensitive substring match).  A "pass" requires
                      recall ≥ ``pass_threshold`` (default 0.5).

    token_f1        — best-matching-window token F1 between the reference
                      answer and the model response: rather than scoring
                      the whole (possibly long, free-form) response, this
                      finds and scores the response's best-matching
                      contiguous span, so a correct short answer embedded
                      in a longer response isn't penalized purely for the
                      response's length. Question vocabulary is excluded
                      from the overlap count on both sides, and the
                      tokenizer preserves signs/fractions/decimals
                      (e.g. "+3" vs "-3", "1/4" vs "1/8") instead of
                      stripping them like plain punctuation.

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
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[+-]?\d+(?:[./]\d+)?|[a-z]+(?:'[a-z]+)*")

# Cap how long a window we'll search, so an unusually long generation (a
# quiz gen_config override with a high max_new_tokens) can't blow up the
# O(P^2) window search. The default quiz cap is 128 tokens, well under this.
_MAX_WINDOW_SEARCH_TOKENS = 512


def _tokenize(text: str) -> list[str]:
    """Lowercase; extract number-like and word-like tokens.

    Numbers keep their sign/decimal/fraction, so "+3"/"-3" and "1/4"/"1/8"
    don't collapse to the same token like they would under plain
    punctuation-stripping, and possessives like "gundren's" stay whole
    instead of leaking a bare "s" token.
    """
    return _TOKEN_RE.findall(text.lower())


def _content_tokens(tokens: list[str], exclude: set[str]) -> list[str]:
    """Drop tokens that also appear in the question, so restating the
    question back doesn't buy free token-overlap credit."""
    return [t for t in tokens if t not in exclude]


def _apply_delta(window_counts: dict, ref_counts: Counter, tok: str, delta: int) -> int:
    """Adjust window_counts[tok] by delta; return the resulting change in
    min(window_count, ref_count) for tok -- i.e. the multiset-intersection
    size contributed by this token."""
    before = min(window_counts.get(tok, 0), ref_counts.get(tok, 0))
    window_counts[tok] = window_counts.get(tok, 0) + delta
    after = min(window_counts[tok], ref_counts.get(tok, 0))
    return after - before


def _best_window_f1(pred_tokens: list[str], ref_tokens: list[str]) -> float:
    """Max token-F1 over all contiguous windows of *pred_tokens* against the
    fixed multiset *ref_tokens*.

    Scoring the best-matching span (instead of the whole prediction) is
    what makes precision length-insensitive: a correct short answer
    embedded in a longer response scores on its own merits instead of
    being diluted by everything around it. Recall is unaffected -- it was
    never the broken part.
    """
    if not pred_tokens or not ref_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    n_ref = len(ref_tokens)
    n_pred = len(pred_tokens)
    max_len = min(n_pred, _MAX_WINDOW_SEARCH_TOKENS)

    best_f1 = 0.0
    for length in range(1, max_len + 1):
        window_counts: dict = {}
        n_common = 0
        for tok in pred_tokens[:length]:
            n_common += _apply_delta(window_counts, ref_counts, tok, 1)

        for start in range(0, n_pred - length + 1):
            if start > 0:
                n_common += _apply_delta(window_counts, ref_counts, pred_tokens[start - 1], -1)
                n_common += _apply_delta(window_counts, ref_counts, pred_tokens[start + length - 1], 1)
            if n_common:
                precision = n_common / length
                recall = n_common / n_ref
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > best_f1:
                    best_f1 = f1

    return best_f1


def _token_f1(prediction: str, reference: str, question: str) -> float:
    """Best-matching-window token F1 between *prediction* and *reference*.

    A length-insensitive, question-vocabulary-excluded variant of unigram
    F1. Plain SQuAD-style F1 assumes *prediction* is already a short
    extracted span; this quiz scores full free-form generations instead,
    so this searches for and scores the response's best-matching
    contiguous span rather than the whole response, and excludes any
    vocabulary shared with the question from the overlap count on both
    sides (so echoing the question doesn't inflate the score).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    question_tokens = set(_tokenize(question))
    filtered_pred = _content_tokens(pred_tokens, question_tokens)
    filtered_ref = _content_tokens(ref_tokens, question_tokens)
    # A short reference can overlap almost entirely with the question's own
    # wording -- don't degrade to a spurious 0.0 if exclusion empties either
    # side; fall back to the unfiltered tokens for this example instead.
    if not filtered_pred or not filtered_ref:
        filtered_pred, filtered_ref = pred_tokens, ref_tokens

    return _best_window_f1(filtered_pred, filtered_ref)


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
            f1 = _token_f1(response, reference, question)
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
