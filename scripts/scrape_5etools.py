"""Download and parse SRD-licensed D&D 5e data from the 5etools-src repo.

5etools (https://5etools.com) maintains a structured JSON dataset covering
spells, monsters, items, feats, and more.  The repo's *code* is MIT-licensed,
but the rules text itself is WotC IP — only entries explicitly flagged as
part of the free SRD / Basic Rules (``srd``, ``srd52``, ``basicRules``, or
``basicRules2024`` truthy) are safe to use here.  This script downloads a
pinned release of the repo, filters every supported category to SRD-only
entries, renders 5etools' nested entry/inline-tag format to plain text, and
writes chunked ``.txt`` files ready for corpus ingestion.

Output format per entry
------------------------
    # Entry name
    Category: spell
    <category-specific metadata line(s)>

    <rendered body as plain text>

    ---

Each output file contains up to ``--chunk`` entries (default 200).

Usage
-----
    python scripts/scrape_5etools.py
    python scripts/scrape_5etools.py --out data/corpus/saga --categories spells bestiary
    python scripts/scrape_5etools.py --cache /tmp/5etools_src --chunk 500

Requirements
------------
    pip install "grimoire-ai[scraper]"  # includes requests
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO        = "5etools-mirror-3/5etools-src"
DEFAULT_REF = "v2.30.1"
ZIP_URL     = "https://github.com/{repo}/archive/refs/tags/{ref}.zip"
USER_AGENT  = (
    "grimoire-corpus-scraper/1.0 (academic/research use; "
    "github.com/BasLinders/grimoire)"
)

_SCHOOL_NAMES = {
    "A": "Abjuration", "C": "Conjuration", "D": "Divination",
    "E": "Enchantment", "V": "Evocation", "I": "Illusion",
    "N": "Necromancy", "T": "Transmutation",
}
_SIZE_NAMES = {
    "T": "Tiny", "S": "Small", "M": "Medium", "L": "Large",
    "H": "Huge", "G": "Gargantuan",
}
_ABILS = ["str", "dex", "con", "int", "wis", "cha"]

_ATK_LABELS = {
    "mw": "Melee Weapon Attack:",
    "rw": "Ranged Weapon Attack:",
    "ms": "Melee Spell Attack:",
    "rs": "Ranged Spell Attack:",
}


def _ordinal(n: int) -> str:
    """Render an integer as an ordinal string, e.g. 1 -> '1st'."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ---------------------------------------------------------------------------
# License filter
# ---------------------------------------------------------------------------

def _is_srd(entry: dict) -> bool:
    """Return True when *entry* is flagged as free SRD / Basic Rules content."""
    return bool(
        entry.get("srd")
        or entry.get("srd52")
        or entry.get("basicRules")
        or entry.get("basicRules2024")
    )


# ---------------------------------------------------------------------------
# Inline tag resolution ({@tag arg1|arg2|...})
# ---------------------------------------------------------------------------

_TAG_RE = re.compile(r"\{@(\w+)\s*([^}]*)\}")


def _resolve_tag(match: "re.Match") -> str:
    tag = match.group(1)
    raw = match.group(2)
    parts = [p.strip() for p in raw.split("|")] if raw else []
    first = parts[0] if parts else ""
    last = parts[-1] if parts else ""

    if tag == "h":
        return "Hit: "
    if tag == "hit":
        return f"+{first}" if first else ""
    if tag == "dc":
        return f"DC {first}" if first else "DC"
    if tag == "recharge":
        return f"(Recharge {first or '6'}-6)"
    if tag == "chance":
        return f"{first}%" if first else ""
    if tag == "atk":
        codes = [c.strip() for c in first.split(",")] if first else []
        labels = [_ATK_LABELS.get(c, c) for c in codes]
        return " or ".join(labels)
    if tag in ("scaledamage", "scaledice"):
        return last
    if tag in ("dice", "damage"):
        return first or raw
    return first or raw


def _resolve_tags(text: str) -> str:
    return _TAG_RE.sub(_resolve_tag, text)


# ---------------------------------------------------------------------------
# Recursive entry-block rendering
# ---------------------------------------------------------------------------

def _render_block(block, depth: int = 0) -> list[str]:
    """Render one entries-format block (string or typed dict) to text lines."""
    if isinstance(block, str):
        return [_resolve_tags(block)]
    if not isinstance(block, dict):
        return []

    btype = block.get("type")
    lines: list[str] = []

    if btype == "list":
        for item in block.get("items", []):
            if isinstance(item, str):
                lines.append(f"- {_resolve_tags(item)}")
            elif isinstance(item, dict):
                name = item.get("name")
                if "entry" in item:
                    text = _resolve_tags(str(item["entry"]))
                    lines.append(f"- {name}: {text}" if name else f"- {text}")
                elif "entries" in item:
                    if name:
                        lines.append(f"- {name}")
                    for sub in item.get("entries", []):
                        lines.extend(_render_block(sub, depth + 1))
                else:
                    lines.extend(_render_block(item, depth + 1))
        return lines

    if btype == "table":
        if block.get("caption"):
            lines.append(_resolve_tags(block["caption"]))
        col_labels = block.get("colLabels", [])
        if col_labels:
            lines.append(" | ".join(_resolve_tags(str(c)) for c in col_labels))
        for row in block.get("rows", []):
            cells = row if isinstance(row, list) else [row]
            lines.append(" | ".join(_resolve_tags(str(c)) for c in cells))
        return lines

    # Generic container: entries / inset / section / entry (string) variants.
    name = block.get("name")
    if name:
        lines.append(str(name))
    if "entries" in block:
        for sub in block["entries"]:
            lines.extend(_render_block(sub, depth + 1))
    elif "entry" in block:
        lines.append(_resolve_tags(str(block["entry"])))
    return lines


def _render_entries(entries: list) -> str:
    lines: list[str] = []
    for block in entries:
        lines.extend(_render_block(block))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Category-specific metadata + formatters
# ---------------------------------------------------------------------------

def _format_spell(sp: dict) -> str:
    level = sp.get("level", 0)
    level_str = "Cantrip" if level == 0 else f"{_ordinal(level)}-level"
    school = _SCHOOL_NAMES.get(sp.get("school", ""), sp.get("school", ""))
    meta = [f"Category: spell", f"Level: {level_str}", f"School: {school}"]
    body = _render_entries(sp.get("entries", []))
    return "\n".join(meta) + "\n\n" + body


def _format_monster(m: dict) -> str:
    size = " ".join(_SIZE_NAMES.get(s, s) for s in m.get("size", []))
    mtype = m.get("type")
    type_str = mtype if isinstance(mtype, str) else (mtype or {}).get("type", "")
    cr = m.get("cr")
    cr_str = cr if isinstance(cr, str) else (cr or {}).get("cr", "") if cr else ""
    meta = [f"Category: monster", f"Size: {size}", f"Type: {type_str}", f"CR: {cr_str}"]
    body_parts: list[str] = []
    for trait_key, label in (("trait", "Traits"), ("action", "Actions"), ("legendary", "Legendary Actions")):
        traits = m.get(trait_key)
        if traits:
            body_parts.append(f"## {label}")
            for t in traits:
                name = t.get("name", "")
                text = _render_entries(t.get("entries", []))
                body_parts.append(f"{name}: {text}" if name else text)
    return "\n".join(meta) + "\n\n" + "\n\n".join(body_parts)


def _format_item(it: dict) -> str:
    rarity = it.get("rarity", "")
    item_type = it.get("type", "")
    meta = [f"Category: item", f"Type: {item_type}", f"Rarity: {rarity}"]
    body = _render_entries(it.get("entries", []))
    return "\n".join(meta) + "\n\n" + body


def _format_generic_factory(kind: str) -> Callable[[dict], str]:
    """Build a formatter for simple categories: name + entries, nothing more."""

    def _format(entry: dict) -> str:
        meta = f"Category: {kind}"
        body = _render_entries(entry.get("entries", []))
        return f"{meta}\n\n{body}"

    return _format


def _format_language(entry: dict) -> str:
    meta = [f"Category: language", f"Type: {entry.get('type', '')}"]
    script = entry.get("script")
    if script:
        meta.append(f"Script: {script}")
    body_parts = []
    if "entries" in entry:
        body_parts.append(_render_entries(entry["entries"]))
    speakers = entry.get("typicalSpeakers")
    if speakers:
        body_parts.append("Typical speakers: " + ", ".join(speakers))
    return "\n".join(meta) + "\n\n" + "\n".join(body_parts)


# ---------------------------------------------------------------------------
# Category registry
# ---------------------------------------------------------------------------

@dataclass
class CategorySpec:
    glob: str
    key: str
    formatter: Callable[[dict], str]
    name_field: str = "name"


CATEGORY_SPECS: "dict[str, CategorySpec]" = {
    "spells": CategorySpec("data/spells/spells-*.json", "spell", _format_spell),
    "bestiary": CategorySpec("data/bestiary/bestiary-*.json", "monster", _format_monster),
    "items-base": CategorySpec("data/items-base.json", "baseitem", _format_item),
    "items": CategorySpec("data/items.json", "item", _format_item),
    "feats": CategorySpec("data/feats.json", "feat", _format_generic_factory("feat")),
    "backgrounds": CategorySpec("data/backgrounds.json", "background", _format_generic_factory("background")),
    "races": CategorySpec("data/races.json", "race", _format_generic_factory("race")),
    "conditions": CategorySpec("data/conditionsdiseases.json", "condition", _format_generic_factory("condition")),
    "variantrules": CategorySpec("data/variantrules.json", "variantrule", _format_generic_factory("variantrule")),
    "optionalfeatures": CategorySpec("data/optionalfeatures.json", "optionalfeature", _format_generic_factory("optionalfeature")),
    "actions": CategorySpec("data/actions.json", "action", _format_generic_factory("action")),
    "skills": CategorySpec("data/skills.json", "skill", _format_generic_factory("skill")),
    "senses": CategorySpec("data/senses.json", "sense", _format_generic_factory("sense")),
    "languages": CategorySpec("data/languages.json", "language", _format_language),
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_zip(ref: str, dest: Path, session: requests.Session) -> None:
    """Stream the repo's release zip to *dest*, showing progress."""
    url = ZIP_URL.format(repo=REPO, ref=ref)
    print(f"Downloading {REPO}@{ref}")
    print(f"  → {dest}")
    with session.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65_536):
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {downloaded / 1_048_576:.1f} / "
                        f"{total / 1_048_576:.1f} MB",
                        end="",
                        flush=True,
                    )
    print(f"\r  Download complete ({downloaded / 1_048_576:.1f} MB)")


def _extract_zip(zip_path: Path, extract_dir: Path, ref: str) -> Path:
    """Extract the downloaded zip and return the path to its root directory."""
    print("Extracting archive …")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    version = ref.lstrip("v")
    repo_name = REPO.split("/")[1]
    root = extract_dir / f"{repo_name}-{version}"
    if not root.is_dir():
        # Fall back to whatever single directory got extracted.
        candidates = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(candidates) == 1:
            root = candidates[0]
        else:
            raise FileNotFoundError(f"Could not locate extracted root under {extract_dir}")
    return root


# ---------------------------------------------------------------------------
# Parsing + writing
# ---------------------------------------------------------------------------

def _load_entries(repo_root: Path, spec: CategorySpec) -> list[dict]:
    """Load and concatenate every JSON array entry matching *spec*, SRD-filtered."""
    entries: list[dict] = []
    for path in sorted(repo_root.glob(spec.glob)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for entry in data.get(spec.key, []):
            if _is_srd(entry):
                entries.append(entry)
    return entries


def _write_chunks(
    category: str,
    entries: list[dict],
    spec: CategorySpec,
    out_dir: Path,
    chunk_size: int,
) -> int:
    """Render entries and write them in chunks of *chunk_size* per file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    buffer: list[str] = []

    def _flush() -> None:
        nonlocal chunk_idx
        if not buffer:
            return
        path = out_dir / f"5etools_{category}_{chunk_idx:04d}.txt"
        path.write_text("\n\n".join(buffer), encoding="utf-8")
        print(f"  [saved] {path.name}  ({len(buffer)} entries)")
        chunk_idx += 1
        buffer.clear()

    for entry in entries:
        name = entry.get(spec.name_field, "Untitled")
        try:
            body = spec.formatter(entry)
        except Exception as exc:  # malformed entry; skip rather than abort the run
            print(f"  [skip] {name!r}: {exc}")
            continue
        buffer.append(f"# {name}\n{body}\n\n---")
        if len(buffer) >= chunk_size:
            _flush()

    _flush()
    return len(entries)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a D&D 5e SRD corpus from the 5etools-src dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", default="data/corpus/saga",
                        help="Output directory for .txt files")
    parser.add_argument("--cache", default="data/corpus/saga/.cache/5etools_src",
                        help="Directory to cache the downloaded zip and extracted repo")
    parser.add_argument("--ref", default=DEFAULT_REF,
                        help="5etools-src git tag to download")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORY_SPECS),
                        choices=list(CATEGORY_SPECS),
                        help="Which categories to scrape")
    parser.add_argument("--chunk", type=int, default=200,
                        help="Entries per output .txt file")
    args = parser.parse_args()

    cache_dir = Path(args.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / f"5etools-src-{args.ref}.zip"
    extract_dir = cache_dir / "extracted"
    out_dir = Path(args.out)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    # -- Download ------------------------------------------------------
    if zip_path.exists():
        print(f"Using cached archive: {zip_path}")
    else:
        _download_zip(args.ref, zip_path, session)

    # -- Extract ---------------------------------------------------------
    repo_root = None
    if extract_dir.exists():
        version = args.ref.lstrip("v")
        repo_name = REPO.split("/")[1]
        candidate = extract_dir / f"{repo_name}-{version}"
        if candidate.is_dir():
            repo_root = candidate
            print(f"Using cached extraction: {repo_root}")
    if repo_root is None:
        repo_root = _extract_zip(zip_path, extract_dir, args.ref)

    # -- Parse + write -----------------------------------------------------
    total = 0
    for category in args.categories:
        spec = CATEGORY_SPECS[category]
        print(f"\n[{category}] loading SRD entries …")
        entries = _load_entries(repo_root, spec)
        print(f"  {len(entries):,} SRD-eligible entries")
        if not entries:
            continue
        count = _write_chunks(category, entries, spec, out_dir, args.chunk)
        total += count

    print(f"\nWrote {total:,} entries across {len(args.categories)} categories → {out_dir}")


if __name__ == "__main__":
    main()
