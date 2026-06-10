"""Scrape the Open5e public API and write corpus .txt files.

Downloads monsters, spells, classes, races, magic items, conditions,
weapons, armor, and feats from https://api.open5e.com and writes one
.txt file per resource type into the output directory.

Usage
-----
    python scripts/scrape_open5e.py
    python scripts/scrape_open5e.py --output data/corpus/saga/ --delay 0.3

Requirements
------------
    pip install requests  (already in grimoire-ai[scraper])
"""

import argparse
import json
import time
from pathlib import Path

import requests

BASE_URL = "https://api.open5e.com/v1"

# Endpoints to scrape and the output filename for each.
ENDPOINTS = [
    ("monsters",     "open5e_monsters.txt"),
    ("spells",       "open5e_spells.txt"),
    ("classes",      "open5e_classes.txt"),
    ("races",        "open5e_races.txt"),
    ("magicitems",   "open5e_magic_items.txt"),
    ("conditions",   "open5e_conditions.txt"),
    ("weapons",      "open5e_weapons.txt"),
    ("armor",        "open5e_armor.txt"),
    ("feats",        "open5e_feats.txt"),
    ("backgrounds",  "open5e_backgrounds.txt"),
    ("sections",     "open5e_sections.txt"),  # rules text — very useful
]


# ---------------------------------------------------------------------------
# Formatters — convert a JSON object to readable plain text
# ---------------------------------------------------------------------------

def _fmt_monster(m: dict) -> str:
    lines = [f"# {m.get('name', 'Unknown')}"]
    lines.append(
        f"Type: {m.get('type', '')} | Size: {m.get('size', '')} | "
        f"Alignment: {m.get('alignment', '')} | CR: {m.get('challenge_rating', '')}"
    )
    lines.append(
        f"HP: {m.get('hit_points', '')} ({m.get('hit_dice', '')}) | "
        f"AC: {m.get('armor_class', '')} ({m.get('armor_desc', '')}) | "
        f"Speed: {m.get('speed', '')}"
    )
    stats = ["strength", "dexterity", "constitution", "intelligence", "wisdom", "charisma"]
    stat_line = " | ".join(f"{s.capitalize()[:3]}: {m.get(s, '')}" for s in stats)
    lines.append(stat_line)
    if m.get("senses"):
        lines.append(f"Senses: {m['senses']}")
    if m.get("languages"):
        lines.append(f"Languages: {m['languages']}")
    if m.get("damage_immunities"):
        lines.append(f"Damage immunities: {m['damage_immunities']}")
    if m.get("damage_resistances"):
        lines.append(f"Damage resistances: {m['damage_resistances']}")
    if m.get("condition_immunities"):
        lines.append(f"Condition immunities: {m['condition_immunities']}")
    if m.get("saving_throws"):
        lines.append(f"Saving throws: {m['saving_throws']}")
    if m.get("skills"):
        lines.append(f"Skills: {m['skills']}")
    for trait in m.get("special_abilities") or []:
        lines.append(f"\n{trait.get('name', '')}: {trait.get('desc', '')}")
    for action in m.get("actions") or []:
        lines.append(f"\nAction — {action.get('name', '')}: {action.get('desc', '')}")
    for reaction in m.get("reactions") or []:
        lines.append(f"\nReaction — {reaction.get('name', '')}: {reaction.get('desc', '')}")
    for la in m.get("legendary_actions") or []:
        lines.append(f"\nLegendary action — {la.get('name', '')}: {la.get('desc', '')}")
    if m.get("desc"):
        lines.append(f"\n{m['desc']}")
    return "\n".join(lines)


def _fmt_spell(s: dict) -> str:
    lines = [f"# {s.get('name', 'Unknown')}"]
    lines.append(
        f"Level: {s.get('level_int', s.get('level', ''))} | "
        f"School: {s.get('school', '')} | "
        f"Casting time: {s.get('casting_time', '')} | "
        f"Range: {s.get('range', '')}"
    )
    lines.append(
        f"Components: {s.get('components', '')} | "
        f"Duration: {s.get('duration', '')} | "
        f"Concentration: {s.get('concentration', '')} | "
        f"Ritual: {s.get('ritual', '')}"
    )
    if s.get("classes"):
        lines.append(f"Classes: {s['classes']}")
    if s.get("desc"):
        lines.append(f"\n{s['desc']}")
    if s.get("higher_level"):
        lines.append(f"\nAt higher levels: {s['higher_level']}")
    return "\n".join(lines)


def _fmt_generic(obj: dict) -> str:
    """Fallback formatter: write name + all text fields."""
    name = obj.get("name", obj.get("slug", "Unknown"))
    lines = [f"# {name}"]
    for key, val in obj.items():
        if key in ("name", "slug", "document__slug", "document__title"):
            continue
        if isinstance(val, str) and val.strip():
            lines.append(f"{key.replace('_', ' ').capitalize()}: {val.strip()}")
        elif isinstance(val, list) and val:
            try:
                lines.append(f"{key.replace('_', ' ').capitalize()}: {', '.join(str(v) for v in val)}")
            except Exception:
                pass
    return "\n".join(lines)


_FORMATTERS = {
    "monsters": _fmt_monster,
    "spells":   _fmt_spell,
}


def _format(endpoint: str, obj: dict) -> str:
    fmt = _FORMATTERS.get(endpoint, _fmt_generic)
    try:
        return fmt(obj)
    except Exception:
        return _fmt_generic(obj)


# ---------------------------------------------------------------------------
# Pagination helper
# ---------------------------------------------------------------------------

def _fetch_all(endpoint: str, delay: float, page_size: int = 100) -> list[dict]:
    """Fetch all pages from an Open5e list endpoint."""
    url = f"{BASE_URL}/{endpoint}/?limit={page_size}&format=json"
    results = []
    page = 0
    while url:
        page += 1
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"  ✘ {endpoint} page {page}: {exc}")
            break
        data = resp.json()
        batch = data.get("results", [])
        results.extend(batch)
        print(f"  {endpoint}: page {page} — {len(results)} items so far")
        url = data.get("next")
        if url:
            time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def scrape(output_dir: str, delay: float) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for endpoint, filename in ENDPOINTS:
        print(f"\nFetching {endpoint}...")
        items = _fetch_all(endpoint, delay=delay)
        if not items:
            print(f"  No data returned for {endpoint}, skipping.")
            continue

        texts = []
        for item in items:
            text = _format(endpoint, item)
            if text.strip():
                texts.append(text)

        out_path = out / filename
        out_path.write_text("\n\n---\n\n".join(texts), encoding="utf-8")
        print(f"  ✔ {len(texts)} records → {out_path}")

    print("\nDone. Run the Preprocess tab to tokenize the new files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Open5e API to corpus .txt files")
    parser.add_argument(
        "--output", default="data/corpus/saga/",
        help="Output directory for .txt files (default: data/corpus/saga/)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.25,
        help="Seconds between paginated requests (default: 0.25)",
    )
    args = parser.parse_args()
    scrape(args.output, args.delay)
