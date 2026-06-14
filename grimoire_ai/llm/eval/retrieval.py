"""Retrieval hit-rate evaluation.

For each query in a fixed query set, the top-1 retrieved passage is
checked for the presence of a set of expected keywords.  A hit is
counted when *all* expected keywords appear in the passage text
(case-insensitive substring match).

This metric measures whether the RAG index actually surfaces the right
passage for domain-specific questions — a prerequisite for grounded
generation quality.

Default query set
-----------------
``SAGA_QUERIES`` is a built-in 20-query set for the Saga D&D corpus.
Pass your own list to override it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from grimoire_ai.llm.inference.engine import InferenceEngine

# ---------------------------------------------------------------------------
# Default Saga query set
# ---------------------------------------------------------------------------

SAGA_QUERIES: list[dict] = [
    {"query": "grapple speed movement", "keywords": ["grappled", "speed"]},
    {"query": "advantage roll d20 twice", "keywords": ["advantage", "d20"]},
    {"query": "critical hit natural 20 damage dice", "keywords": ["critical", "damage"]},
    {"query": "proficiency bonus level scaling", "keywords": ["proficiency"]},
    {"query": "spell save DC calculation formula", "keywords": ["save dc", "proficiency"]},
    {"query": "concentration spell casting rules", "keywords": ["concentration"]},
    {"query": "rogue sneak attack conditions", "keywords": ["sneak attack"]},
    {"query": "bounded accuracy attack bonus AC", "keywords": ["bounded accuracy"]},
    {"query": "frightened condition disadvantage", "keywords": ["frightened", "disadvantage"]},
    {"query": "incapacitated condition actions reactions", "keywords": ["incapacitated", "actions"]},
    {"query": "spellcasting wizard spell slots level 5", "keywords": ["spell slots"]},
    {"query": "encounter XP budget deadly threshold", "keywords": ["xp", "deadly"]},
    {"query": "monster multiplier encounter difficulty", "keywords": ["multiplier"]},
    {"query": "action bonus action reaction turn", "keywords": ["action", "bonus action"]},
    {"query": "hit chance armor class attack bonus formula", "keywords": ["attack", "ac"]},
    {"query": "4d6 drop lowest ability score", "keywords": ["4d6", "drop"]},
    {"query": "saving throw ability check difference", "keywords": ["saving throw"]},
    {"query": "greatsword 2d6 average damage", "keywords": ["2d6", "damage"]},
    {"query": "disadvantage natural 20 critical hit", "keywords": ["disadvantage", "natural 20"]},
    {"query": "long rest spell slot recovery", "keywords": ["long rest"]},
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def eval_retrieval(
    engine: "InferenceEngine",
    queries: Optional[list[dict]] = None,
    top_k: int = 1,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Measure retrieval hit-rate over a fixed query set.

    A *hit* is counted when the top-``top_k`` retrieved passages together
    contain all expected keywords (case-insensitive).

    Args:
        engine: A loaded ``InferenceEngine`` with a corpus attached.
            If no corpus is set, returns ``nan`` for all metrics.
        queries: List of ``{"query": str, "keywords": list[str]}`` dicts.
            Defaults to ``SAGA_QUERIES``.
        top_k: Number of passages to retrieve per query.  Hit is counted
            when any passage in the top-k contains all keywords.
        on_progress: Optional callback for log lines.

    Returns:
        Dict with keys ``hit_rate``, ``hits``, ``total``, ``per_query``.
    """
    def _log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    if engine.corpus is None:
        _log("  ⚠  No corpus attached — skipping retrieval eval.")
        return {
            "hit_rate": float("nan"),
            "hits": 0,
            "total": 0,
            "per_query": [],
        }

    if queries is None:
        queries = SAGA_QUERIES

    _log(f"Retrieval eval: {len(queries)} queries, top-{top_k} …")

    hits = 0
    per_query: list[dict] = []

    for i, item in enumerate(queries):
        query = item["query"]
        keywords = [kw.lower() for kw in item["keywords"]]

        results = engine.corpus.query(query, top_k=top_k)
        passages = " ".join(
            (r.excerpt or "").lower()
            for r in results
        )

        matched = all(kw in passages for kw in keywords)
        if matched:
            hits += 1

        top_score = results[0].score if results else 0.0
        per_query.append({
            "query": query,
            "keywords": item["keywords"],
            "hit": matched,
            "top_score": round(float(top_score), 4),
            "top_passage_snippet": passages[:120] if passages else "",
        })

        if (i + 1) % 5 == 0:
            _log(f"  {i+1}/{len(queries)} queries  hits so far: {hits}")

    total = len(queries)
    hit_rate = hits / total if total else 0.0
    _log(f"  hit-rate {hit_rate:.1%}  ({hits}/{total})")

    return {
        "hit_rate": round(hit_rate, 4),
        "hits": hits,
        "total": total,
        "per_query": per_query,
    }
