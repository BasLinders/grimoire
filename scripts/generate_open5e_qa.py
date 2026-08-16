"""Generate template-based fine-tuning Q&A from the Open5e API.

Complements scripts/scrape_stackexchange.py's general-conversation data with
the opposite property: every fact here is code-derived from Open5e's
structured fields (CR, HP, AC, spell level/school/range/...), so there is no
hallucination risk -- the question templates below can only ever produce a
*correct* answer, because the answer is read straight off the same field the
question asks about. Fixes the "correctly" half of "converse coherently and
correctly"; scrape_stackexchange.py's general sites fix the "coherently"
half, and don't need to be D&D-specific to do it (see docs/expansion_PLAN.md
and agents.json's description of Saga as a D&D *and* data-science assistant
-- conversational fluency and D&D knowledge are separable, and only the
knowledge half needs D&D-topical source data).

"No hallucination risk" holds per-example (each answer is read from a real
field) but not in aggregate without --document-slug: Open5e mixes the
official wotc-srd document with unrelated third-party rulesets under the
same endpoints, so the same monster/spell name can have several entries
with different field values across documents (e.g. two different CRs for
"Aboleth") -- without filtering, that produces the exact same question
with directly contradictory "correct" answers in the output, which is
worse for fine-tuning than an individually-wrong answer would be. Defaults
to --document-slug wotc-srd for this reason (see
generate_open5e_entigraph.py's module docstring for the same issue found
and fixed there first).

Reuses scrape_open5e.py's pagination (_fetch_all) and formatters
(_fmt_monster, _fmt_spell) so the ``context`` field here is byte-identical to
what that script writes into data/corpus/saga/open5e_*.txt -- i.e. the same
passage a retriever would actually hand the model at inference.

Output format
-------------
JSONL, one example per line, in grimoire_ai.llm.data.conversation.
ConversationDataset's format:

    {"user": "What is the challenge rating of the Troll?",
     "assistant": "The Troll is challenge rating 5 (1,800 XP).",
     "context": "# Troll\\nType: giant | Size: Large | ..."}

Several question templates are generated per monster/spell (CR/XP, HP, AC,
type/size/alignment, speed for monsters; level/school, casting time, range,
duration, effect for spells), so one API entry yields several examples.
A template is skipped when its underlying field is empty, rather than
emitting a question with no real answer.

Usage
-----
    python scripts/generate_open5e_qa.py
    python scripts/generate_open5e_qa.py --types monsters --output data/finetune/open5e_monster_qa.jsonl
    python scripts/generate_open5e_qa.py --delay 0.3
    python scripts/generate_open5e_qa.py --document-slug ""  # every Open5e document, unfiltered

Then fine-tune the same way as the StackExchange-derived data (see
scripts/finetune_saga.py) -- concatenate this file with any other .jsonl
fine-tune source first if training on more than one:

    cat data/finetune/open5e_qa.jsonl data/finetune/general_se_qa.jsonl \\
        > data/finetune/combined.jsonl

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scrape_open5e import _fetch_all, _fmt_monster, _fmt_spell  # noqa: E402

# Challenge ratings below 1 are fractional in the API (e.g. 0.125) -- map
# them back to the conventional "1/8" etc. display form used in sourcebooks
# and by the rest of this corpus (see docs/expansion_PLAN.md's CR/XP table).
_FRACTIONAL_CR = {0.125: "1/8", 0.25: "1/4", 0.5: "1/2"}


def _fmt_cr(cr) -> str:
    if cr in _FRACTIONAL_CR:
        return _FRACTIONAL_CR[cr]
    if isinstance(cr, float) and cr == int(cr):
        return str(int(cr))
    return str(cr)


def _example(user: str, assistant: str, context: str) -> str:
    return json.dumps({"user": user, "assistant": assistant, "context": context})


# ---------------------------------------------------------------------------
# Monster templates
# ---------------------------------------------------------------------------

def _monster_examples(m: dict) -> list[str]:
    name = m.get("name", "").strip()
    if not name:
        return []
    context = _fmt_monster(m)
    out: list[str] = []

    cr = m.get("challenge_rating")
    if cr not in (None, ""):
        xp = m.get("xp")
        xp_part = f" ({xp:,} XP)" if isinstance(xp, (int, float)) and xp else ""
        out.append(_example(
            f"What is the challenge rating of the {name}?",
            f"The {name} is challenge rating {_fmt_cr(cr)}{xp_part}.",
            context,
        ))

    hp = m.get("hit_points")
    if hp not in (None, ""):
        dice_part = f" ({m['hit_dice']})" if m.get("hit_dice") else ""
        out.append(_example(
            f"How many hit points does a {name} have?",
            f"A {name} has {hp} hit points{dice_part}.",
            context,
        ))

    ac = m.get("armor_class")
    if ac not in (None, ""):
        desc_part = f" ({m['armor_desc']})" if m.get("armor_desc") else ""
        out.append(_example(
            f"What is the armor class of a {name}?",
            f"A {name} has an armor class of {ac}{desc_part}.",
            context,
        ))

    size, ctype, alignment = m.get("size"), m.get("type"), m.get("alignment")
    if size and ctype:
        align_part = f", {alignment}" if alignment else ""
        out.append(_example(
            f"What type of creature is a {name}?",
            f"A {name} is a {size.lower()} {ctype}{align_part}.",
            context,
        ))

    speed = m.get("speed")
    if speed:
        speed_str = speed if isinstance(speed, str) else ", ".join(
            f"{k} {v} ft." for k, v in speed.items()
        ) if isinstance(speed, dict) else str(speed)
        if speed_str:
            out.append(_example(
                f"How fast can a {name} move?",
                f"A {name}'s speed is {speed_str.rstrip('.')}.",
                context,
            ))

    return out


# ---------------------------------------------------------------------------
# Spell templates
# ---------------------------------------------------------------------------

def _spell_examples(s: dict) -> list[str]:
    name = s.get("name", "").strip()
    if not name:
        return []
    context = _fmt_spell(s)
    out: list[str] = []

    level = s.get("level_int", s.get("level"))
    school = s.get("school")
    if level is not None and school:
        level_str = "cantrip" if str(level) == "0" else f"level {level}"
        out.append(_example(
            f"What level and school is the spell {name}?",
            f"{name} is a {level_str} {school} spell.",
            context,
        ))

    casting_time = s.get("casting_time")
    if casting_time:
        out.append(_example(
            f"How long does it take to cast {name}?",
            f"{name} has a casting time of {casting_time}.",
            context,
        ))

    rng = s.get("range")
    if rng:
        out.append(_example(
            f"What is the range of the spell {name}?",
            f"{name} has a range of {rng}.",
            context,
        ))

    duration = s.get("duration")
    if duration:
        conc = s.get("concentration")
        conc_part = " It requires concentration." if str(conc).lower() in ("true", "yes", "1") else ""
        out.append(_example(
            f"How long does {name} last?",
            f"{name}'s duration is {duration}.{conc_part}",
            context,
        ))

    desc = (s.get("desc") or "").strip()
    if desc:
        first_sentence = desc.split(". ")[0].rstrip(".") + "."
        if len(first_sentence) > 15:
            out.append(_example(
                f"What does the spell {name} do?",
                first_sentence,
                context,
            ))

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(output: str, types: list[str], delay: float, document_slug: str | None) -> None:
    lines: list[str] = []

    if "monsters" in types:
        print("Fetching monsters...")
        for m in _fetch_all("monsters", delay=delay, document_slug=document_slug):
            lines.extend(_monster_examples(m))
        print(f"  {len(lines)} example(s) so far")

    if "spells" in types:
        before = len(lines)
        print("Fetching spells...")
        for s in _fetch_all("spells", delay=delay, document_slug=document_slug):
            lines.extend(_spell_examples(s))
        print(f"  {len(lines) - before} spell example(s)")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {len(lines):,} example(s) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate template-based, guaranteed-correct fine-tuning "
                     "Q&A from the Open5e API.",
    )
    parser.add_argument("--output", default="data/finetune/open5e_qa.jsonl",
                         help="Output .jsonl path (default: data/finetune/open5e_qa.jsonl)")
    parser.add_argument("--types", nargs="+", default=["monsters", "spells"],
                         choices=["monsters", "spells"],
                         help="Which Open5e resource types to generate from (default: both)")
    parser.add_argument("--delay", type=float, default=0.25,
                         help="Seconds between paginated API requests (default: 0.25)")
    parser.add_argument(
        "--document-slug", default="wotc-srd", metavar="SLUG",
        help="Restrict to one Open5e document (default: wotc-srd, the "
             "official 5e SRD). Open5e mixes unrelated third-party rulesets "
             "into the same endpoints -- without this filter, the same "
             "monster/spell name can appear multiple times from different "
             "documents with contradictory field values (e.g. two different "
             "CRs for 'Aboleth'), producing the same question with directly "
             "conflicting answers in the output. Pass an empty string to "
             "disable filtering and pull from every document Open5e has.",
    )
    args = parser.parse_args()
    generate(args.output, args.types, args.delay, args.document_slug or None)
