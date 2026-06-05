"""Smoke tests for the Saga corpus build script and reference files.

These tests do NOT download the SRD (to keep the suite fast and offline-safe).
They verify:
  - The reference .txt files exist and contain expected keywords.
  - The build script runs with --skip-download and copies references correctly.
  - GrimoireCorpus can load the reference files without error and returns
    results for a sample query.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_REFERENCES = Path(__file__).parent.parent / "scripts" / "saga_references"
_REF_FILES = [
    "dice_probability.txt",
    "encounter_building.txt",
    "encounter_scaling.txt",
    "dnd_math_and_statistics.txt",
]


# ---------------------------------------------------------------------------
# Reference file content
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("filename", _REF_FILES)
def test_reference_file_exists(filename):
    assert (_REFERENCES / filename).exists(), f"Missing: {filename}"


@pytest.mark.parametrize("filename", _REF_FILES)
def test_reference_file_not_empty(filename):
    text = (_REFERENCES / filename).read_text(encoding="utf-8")
    assert len(text.strip()) > 100


def test_dice_probability_contains_expected_terms():
    text = (_REFERENCES / "dice_probability.txt").read_text(encoding="utf-8")
    for term in ("advantage", "disadvantage", "d20", "critical", "saving throw"):
        assert term in text.lower(), f"Expected term '{term}' not found"


def test_encounter_building_contains_xp_table():
    text = (_REFERENCES / "encounter_building.txt").read_text(encoding="utf-8")
    for term in ("easy", "medium", "hard", "deadly", "challenge rating"):
        assert term in text.lower(), f"Expected term '{term}' not found"


def test_encounter_scaling_contains_multiplier_table():
    text = (_REFERENCES / "encounter_scaling.txt").read_text(encoding="utf-8")
    assert "multiplier" in text.lower()
    assert "minion" in text.lower()


def test_dnd_math_contains_dpr_formula():
    text = (_REFERENCES / "dnd_math_and_statistics.txt").read_text(encoding="utf-8")
    for term in ("dpr", "bounded accuracy", "proficiency", "ability score"):
        assert term in text.lower(), f"Expected term '{term}' not found"


# ---------------------------------------------------------------------------
# Build script --skip-download
# ---------------------------------------------------------------------------

def test_build_script_skip_download_copies_references(tmp_path):
    """Running build_saga_corpus.py --skip-download should copy reference files."""
    script = Path(__file__).parent.parent / "scripts" / "build_saga_corpus.py"
    result = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path), "--skip-download"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    produced = {f.name for f in tmp_path.glob("*.txt")}
    for ref in _REF_FILES:
        assert ref in produced, f"Reference file '{ref}' not copied to output."


def test_build_script_output_matches_source(tmp_path):
    """Copied reference files should have identical content to originals."""
    script = Path(__file__).parent.parent / "scripts" / "build_saga_corpus.py"
    subprocess.run(
        [sys.executable, str(script), "--output-dir", str(tmp_path), "--skip-download"],
        capture_output=True,
    )
    for ref in _REF_FILES:
        original = (_REFERENCES / ref).read_text(encoding="utf-8")
        copied = (tmp_path / ref).read_text(encoding="utf-8")
        assert original == copied, f"Content mismatch for {ref}"


# ---------------------------------------------------------------------------
# GrimoireCorpus integration
# ---------------------------------------------------------------------------

def test_corpus_loads_reference_files_without_error(tmp_path):
    """GrimoireCorpus should index the reference files without raising."""
    from grimoire.corpus.corpus import GrimoireCorpus

    corpus = GrimoireCorpus()
    for ref in _REF_FILES:
        text = (_REFERENCES / ref).read_text(encoding="utf-8")
        corpus.add_text(text, source=ref)
    assert corpus is not None


def test_corpus_query_returns_results(tmp_path):
    """Querying the corpus for 'encounter difficulty' should return results."""
    from grimoire.corpus.corpus import GrimoireCorpus

    corpus = GrimoireCorpus()
    for ref in _REF_FILES:
        text = (_REFERENCES / ref).read_text(encoding="utf-8")
        corpus.add_text(text, source=ref)

    results = corpus.query("encounter difficulty xp threshold", top_k=3)
    assert len(results) > 0


def test_corpus_query_dice_returns_results():
    """Querying for dice probability should return relevant results."""
    from grimoire.corpus.corpus import GrimoireCorpus

    corpus = GrimoireCorpus()
    text = (_REFERENCES / "dice_probability.txt").read_text(encoding="utf-8")
    corpus.add_text(text, source="dice_probability")

    results = corpus.query("advantage disadvantage d20 probability", top_k=3)
    assert len(results) > 0
