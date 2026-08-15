"""Tests for heuristic document-quality filtering
(docs/architecture_optimization.md item #9, curation half).

Gate criteria:
- Normal, coherent prose passes every rule.
- Each rule independently catches the document shape it's meant to.
- filter_low_quality returns correct kept indices and a report per document.
- Degenerate inputs (empty list) are handled.
"""

from grimoire_ai.llm.data.quality_filter import (
    QualityThresholds,
    filter_low_quality,
    score_document,
)

_GOOD_PROSE = (
    "A grappled creature has its speed reduced to zero, and it cannot "
    "benefit from any bonus to its speed. The condition ends if the "
    "grappler is incapacitated, or if an effect removes the grappled "
    "creature from the reach of the grappler. A creature can escape by "
    "succeeding on a Strength (Athletics) or Dexterity (Acrobatics) check "
    "contested by the grappler's Strength (Athletics) check, taking its "
    "action to break free from the hold entirely on its own turn."
)


def test_keeps_normal_prose() -> None:
    report = score_document(_GOOD_PROSE)
    assert report.keep, report.reasons


def test_drops_too_short() -> None:
    report = score_document("Too short.")
    assert not report.keep
    assert any("chars" in r for r in report.reasons)


def test_drops_low_alpha_ratio() -> None:
    text = "1234567890 " * 50  # long enough, but almost no alphabetic content
    report = score_document(text)
    assert not report.keep
    assert any("alpha ratio" in r for r in report.reasons)


def test_drops_high_symbol_ratio() -> None:
    text = ("#$%^&*@~`|\\<>{}[]=+ " * 30) + ("word " * 30)
    report = score_document(text)
    assert not report.keep
    assert any("symbol ratio" in r for r in report.reasons)


def test_drops_garbled_word_length() -> None:
    # Concatenated/garbled "words" with no real spaces -- classic OCR failure shape.
    garbled_word = "a" * 40
    text = " ".join([garbled_word] * 50)
    report = score_document(text)
    assert not report.keep
    assert any("mean word length" in r for r in report.reasons)


def test_drops_nav_menu_shape() -> None:
    # Many short lines -- navigation menu / table-of-contents shape.
    text = "\n".join(f"Item {i}" for i in range(60))
    report = score_document(text)
    assert not report.keep
    assert any("short-line ratio" in r for r in report.reasons)


def test_keeps_structured_entries_with_many_short_header_lines() -> None:
    """A document shaped like the real corpus's ``5etools_*``/``wp_math_*``
    files: most *lines* are short structural ones ("# Header", "Category:
    skill", rendered-equation tokens), but most *characters* are real
    prose. short_line_ratio must be measured by character volume, not raw
    line count, or this legitimate content gets dropped -- caught by
    empirically running the quality filter against the real corpus
    (docs/architecture_optimization.md item #9 PR review)."""
    entry = (
        "# Acrobatics\n"
        "Category: skill\n\n"
        + _GOOD_PROSE
        + "\n\n---\n\n"
    )
    text = entry * 20
    report = score_document(text)
    assert report.keep, report.reasons


def test_drops_degenerate_repetition() -> None:
    text = "the quick fox jumps " * 200
    report = score_document(text, thresholds=QualityThresholds(max_top_ngram_repetition_ratio=0.05))
    assert not report.keep
    assert any("repetition ratio" in r for r in report.reasons)


def test_custom_thresholds_are_respected() -> None:
    """A document that fails the default min_chars must pass with a lower one."""
    text = "Short but real prose about a topic worth reading."
    assert not score_document(text).keep
    lenient = QualityThresholds(min_chars=10, min_words=5)
    assert score_document(text, thresholds=lenient).keep


def test_filter_low_quality_returns_correct_indices() -> None:
    texts = [_GOOD_PROSE, "too short", _GOOD_PROSE]
    kept, reports = filter_low_quality(texts)

    assert kept == [0, 2]
    assert len(reports) == 3
    assert reports[0].keep and reports[2].keep
    assert not reports[1].keep


def test_filter_low_quality_empty_input() -> None:
    kept, reports = filter_low_quality([])
    assert kept == []
    assert reports == []


def test_report_contains_reasons_and_stats_for_dropped_docs() -> None:
    _, reports = filter_low_quality(["x"])
    report = reports[0]
    assert not report.keep
    assert len(report.reasons) > 0
    assert "char_count" in report.stats
    assert "word_count" in report.stats


def test_report_index_matches_input_position() -> None:
    texts = ["a" * 500, "b" * 500, "c" * 500]
    _, reports = filter_low_quality(texts)
    assert [r.index for r in reports] == [0, 1, 2]
