"""EntiGraph-style synthetic corpus text from Open5e's structured data.

Unlike generate_open5e_qa.py (which produces Q&A instruction-tuning pairs),
this produces plain corpus prose -- new *pretraining* text connecting
facts about PAIRS of related entities pulled from Open5e's structured API
fields (a monster immune to a condition, a class proficient with a weapon,
a spell available to a class). No LLM call anywhere in this script and no
invented content: every sentence is template-assembled from real fields,
and the *relevance* linking each pair is itself read from a real field
(the monster's own condition_immunities string, the class's own
prof_weapons string, the spell's own spell_lists) rather than sampled at
random -- unlike a naive full cross-product (every monster x every item),
which would mostly produce thematically meaningless pairings with no
grounded reason to connect them.

This is the concrete, code-verified alternative to LLM-based rephrasing/
augmentation -- docs/architecture_optimization.md item #9's "synthetic
augmentation via rephrasing" sub-bullet is explicitly out of scope (see
docs/expansion_PLAN.md's reasoning against bulk LLM-generated content:
impractical volume, hallucination risk, model-collapse risk). Matches the
existing derived-adventure pilot's practice of only ever including
mechanically-verified facts (docs/expansion_PLAN.md).

Reuses scrape_open5e.py's pagination helper (_fetch_all) the same way
generate_open5e_qa.py does, via the sys.path.insert(0, ...) trick since
scripts/ isn't a package.

Categories
----------
- monster_condition: a monster paired with a condition it's immune to
  (via the monster's own condition_immunities field).
- class_weapon: a class paired with a weapon it's proficient with (via
  the class's own prof_weapons field matching the weapon's category or name).
- class_spell: a class paired with a spell on that class's spell list
  (via the spell's own spell_lists field).

Output
------
Plain .txt files under data/corpus/saga_derived/ (NOT saga/ -- same
directory-separation rationale as the existing derived-adventure pilot,
per docs/expansion_PLAN.md: "so future ingestion scripts can filter/weight
this content independently"), one file per category, filename-prefixed
entigraph_* per the corpus's existing source-tagging convention (srd_*,
open5e_*, gutenberg_*, synth_*).

Usage
-----
    python scripts/generate_open5e_entigraph.py
    python scripts/generate_open5e_entigraph.py --categories monster_condition class_spell
    python scripts/generate_open5e_entigraph.py --max-pairs-per-category 200 --seed 1

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from scrape_open5e import _fetch_all  # noqa: E402


# ---------------------------------------------------------------------------
# Per-category passage templates
#
# Each function returns a passage string, or None when a required field is
# missing or the pair isn't actually relevant to each other -- skipped
# rather than emitting a hollow or ungrounded passage, matching
# generate_open5e_qa.py's existing rule.
# ---------------------------------------------------------------------------

def _monster_condition_passage(monster: dict, condition: dict) -> Optional[str]:
    m_name = (monster.get("name") or "").strip()
    c_name = (condition.get("name") or "").strip()
    if not m_name or not c_name:
        return None
    immunities = (monster.get("condition_immunities") or "").lower()
    if c_name.lower() not in immunities:
        return None
    desc = (condition.get("desc") or "").strip()
    if not desc:
        return None
    first_line = desc.lstrip("*").split("\n")[0].strip().rstrip(".") + "."
    return (
        f"A {m_name} is immune to the {c_name.lower()} condition.\n"
        f"{c_name}: {first_line}"
    )


def _class_weapon_passage(cls: dict, weapon: dict) -> Optional[str]:
    cls_name = (cls.get("name") or "").strip()
    weapon_name = (weapon.get("name") or "").strip()
    damage_dice = weapon.get("damage_dice")
    damage_type = weapon.get("damage_type")
    if not cls_name or not weapon_name or not damage_dice or not damage_type:
        return None

    prof = (cls.get("prof_weapons") or "").lower()
    category = (weapon.get("category") or "").lower()
    if not prof or not category:
        return None
    # Relevance check: either the weapon's broad category (simple/martial,
    # from Open5e's own "Simple Melee Weapons"-style category string) is
    # listed among the class's weapon proficiencies, or the weapon's own
    # name is called out there specifically (e.g. Rogue: "...rapiers...").
    category_words = [w for w in ("simple", "martial") if w in category]
    is_proficient = any(w in prof for w in category_words) or weapon_name.lower().rstrip("s") in prof
    if not is_proficient:
        return None

    hit_dice = cls.get("hit_dice")
    hit_dice_part = f" (hit dice {hit_dice})" if hit_dice else ""
    lines = [
        f"A {cls_name}{hit_dice_part} is proficient with {category}, "
        f"which includes the {weapon_name}.",
        f"The {weapon_name} deals {damage_dice} {damage_type} damage.",
    ]
    properties = weapon.get("properties")
    if properties:
        lines.append(f"It has the following properties: {', '.join(properties)}.")
    return "\n".join(lines)


def _class_spell_passage(cls: dict, spell: dict) -> Optional[str]:
    cls_name = (cls.get("name") or "").strip()
    cls_slug = (cls.get("slug") or "").strip().lower()
    spell_name = (spell.get("name") or "").strip()
    if not cls_name or not cls_slug or not spell_name:
        return None

    spell_lists = spell.get("spell_lists") or []
    if cls_slug not in [s.lower() for s in spell_lists]:
        return None

    level = spell.get("level_int", spell.get("level"))
    school = spell.get("school")
    if level is None or not school:
        return None
    level_str = "cantrip" if str(level) == "0" else f"level {level}"

    lines = [f"{spell_name} is a {level_str} {school} spell available to {cls_name}s."]
    desc = (spell.get("desc") or "").strip()
    if desc:
        first_sentence = desc.split(". ")[0].rstrip(".") + "."
        if len(first_sentence) > 15:
            lines.append(first_sentence)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category registry: (fetch endpoints, pairing function, output filename)
# ---------------------------------------------------------------------------

_CATEGORIES = {
    "monster_condition": (("monsters", "conditions"), _monster_condition_passage, "entigraph_monster_condition.txt"),
    "class_weapon": (("classes", "weapons"), _class_weapon_passage, "entigraph_class_weapon.txt"),
    "class_spell": (("classes", "spells"), _class_spell_passage, "entigraph_class_spell.txt"),
}


def _fetch_endpoint_cached(endpoint: str, delay: float, cache: dict[str, list[dict]]) -> list[dict]:
    """Fetch an Open5e endpoint's full item list, once per endpoint per run.

    Several categories share an endpoint -- both class_weapon and
    class_spell fetch "classes" -- so caching avoids re-fetching (and
    re-paginating through) the same few thousand items twice.
    """
    if endpoint not in cache:
        print(f"Fetching {endpoint}...")
        cache[endpoint] = _fetch_all(endpoint, delay=delay)
    return cache[endpoint]


def _generate_category(
    category: str,
    delay: float,
    max_pairs: Optional[int],
    rng: random.Random,
    fetch_cache: dict[str, list[dict]],
) -> list[str]:
    (endpoint_a, endpoint_b), passage_fn, _ = _CATEGORIES[category]
    items_a = _fetch_endpoint_cached(endpoint_a, delay, fetch_cache)
    items_b = _fetch_endpoint_cached(endpoint_b, delay, fetch_cache)

    passages: list[str] = []
    for a in items_a:
        for b in items_b:
            passage = passage_fn(a, b)
            if passage:
                passages.append(passage)

    print(f"  {len(passages)} relevant {category} pair(s) found "
          f"out of {len(items_a) * len(items_b):,} possible.")

    if max_pairs is not None and len(passages) > max_pairs:
        passages = rng.sample(passages, max_pairs)
        print(f"  Sampled down to {max_pairs} pair(s) (--max-pairs-per-category).")

    return passages


def generate(
    output_dir: str,
    categories: list[str],
    delay: float,
    max_pairs_per_category: Optional[int],
    seed: int,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    fetch_cache: dict[str, list[dict]] = {}

    for category in categories:
        _, _, filename = _CATEGORIES[category]
        passages = _generate_category(category, delay, max_pairs_per_category, rng, fetch_cache)
        if not passages:
            print(f"  No passages generated for {category}, skipping.")
            continue
        out_path = out / filename
        out_path.write_text("\n\n---\n\n".join(passages), encoding="utf-8")
        print(f"  ✔ {len(passages)} passage(s) -> {out_path}\n")

    print("Done. These are pretraining passages, not fine-tune Q&A -- run the "
          "Preprocess step (or grimoire-preprocess) against data/corpus/saga_derived/ "
          "to include them, keeping this directory separate from data/corpus/saga/ "
          "so it can be filtered/weighted independently.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate EntiGraph-style synthetic pretraining passages "
                     "from pairs of related Open5e entities.",
    )
    parser.add_argument(
        "--output-dir", default="data/corpus/saga_derived/",
        help="Output directory for .txt files (default: data/corpus/saga_derived/).",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(_CATEGORIES.keys()),
        choices=list(_CATEGORIES.keys()),
        help="Which entity-pair categories to generate (default: all).",
    )
    parser.add_argument(
        "--max-pairs-per-category", type=int, default=500, metavar="N",
        help="Cap on passages per category; 0 = no cap (default: 500).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for the max-pairs-per-category sample (default: 0).",
    )
    parser.add_argument(
        "--delay", type=float, default=0.25,
        help="Seconds between paginated API requests (default: 0.25).",
    )
    args = parser.parse_args()

    max_pairs = None if args.max_pairs_per_category == 0 else args.max_pairs_per_category
    generate(args.output_dir, args.categories, args.delay, max_pairs, args.seed)
