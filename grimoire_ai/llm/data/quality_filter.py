"""Heuristic document-quality filtering (Gopher/C4-style rules).

Drops individual documents that are unlikely to be coherent, useful
training text -- independent of whether they duplicate other documents
(``dedup.py``'s concern) or how well the current model already predicts
them (``score_difficulty.py``'s concern). Neither existing mechanism
catches a single garbled OCR page, a truncated download, or a
nav-menu-only scrape that has no near-duplicate anywhere else in the
corpus, and ``score_difficulty.py``/``build_source_weights.py`` only ever
*downweight* documents (a weight floor still gets sampled occasionally),
never drop them outright.

Pure stdlib, no NER/spacy/ML dependency -- consistent with ``dedup.py``'s
own "no external MinHash/LSH dependency" approach, and this project's
decision not to add general-purpose NER tooling (see
``scripts/generate_open5e_entigraph.py``, which sources entities from
Open5e's already-structured API data instead of general NER).

Item #9 from docs/architecture_optimization.md (curation half; the
"synthetic augmentation via rephrasing" sub-bullet of that item is
explicitly out of scope -- see docs/expansion_PLAN.md's reasoning against
bulk LLM-generated content).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Punctuation that's expected in ordinary prose and should never count
# toward the "junk symbol" ratio.
_COMMON_PUNCT = set(".,;:!?'\"()-–—/\n\t ")

_WORD_RE = re.compile(r"\S+")


@dataclass
class QualityThresholds:
    """Tunable cutoffs for each heuristic rule.

    Defaults are deliberately permissive starting points meant to catch
    clearly-broken documents (empty scrapes, OCR garbage, nav-menu-only
    pages, degenerate repetition), not to aggressively cull marginal ones.

    Attributes:
        min_chars: Documents shorter than this (after stripping) are dropped.
        min_words: Documents with fewer whitespace-separated words are dropped.
        min_alpha_ratio: Minimum fraction of non-whitespace characters that
            must be alphabetic. Catches documents that are mostly digits,
            punctuation, or control characters.
        max_symbol_ratio: Maximum fraction of characters that are neither
            alphanumeric, whitespace, nor common prose punctuation. Catches
            encoding garbage and markup that survived scraping.
        max_mean_word_length: Maximum average word length. Catches
            concatenated/garbled OCR "words" (real prose rarely averages
            much above 8-10 characters per word).
        max_short_line_ratio: Maximum fraction of a document's characters
            (not lines) that fall on a "short" line -- one with fewer than
            3 words. Catches navigation menus, tables of contents, and
            broken table scrapes, which are short-line-dominated by both
            line count *and* character volume. A character-volume ratio
            (rather than a raw count of short lines) is what keeps this
            rule from misfiring on documents that are legitimately prose
            but happen to contain many one-token lines -- e.g. rendered
            MathML/LaTeX ("n", "(", ")" each on their own line) or
            structured entries (a "# Header" / "Category: skill" line per
            short D&D rules entry) -- since those short lines are
            individually tiny and so contribute little to the document's
            actual character volume even when they dominate the line
            count. Only applied when there are enough lines to be
            meaningful.
        max_top_ngram_repetition_ratio: Maximum share of all word 3-grams
            that are the single most common 3-gram. Catches degenerate
            repetition ("does does does...", a boilerplate line repeated
            hundreds of times).
    """

    min_chars: int = 200
    min_words: int = 40
    min_alpha_ratio: float = 0.6
    max_symbol_ratio: float = 0.15
    max_mean_word_length: float = 12.0
    max_short_line_ratio: float = 0.6
    max_top_ngram_repetition_ratio: float = 0.3


@dataclass
class QualityReport:
    """Per-document verdict.

    Dropping data outright is a stronger, less reversible action than
    downweighting it (a document has to be re-scraped to get it back), so
    the audit trail carries *why* a document was dropped and the raw metric
    values, not just a keep/drop bit.

    Attributes:
        index: Position of this document in the input sequence.
        keep: Whether the document passed every rule.
        reasons: Human-readable reason per failed rule. Empty when ``keep``
            is ``True``.
        stats: Raw metric value per rule, regardless of pass/fail --
            useful for reviewing where the thresholds should sit before
            trusting the filter on a new corpus source.
    """

    index: int
    keep: bool
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def score_document(
    text: str,
    thresholds: "QualityThresholds | None" = None,
    index: int = 0,
) -> QualityReport:
    """Evaluate one document against every quality rule.

    Args:
        text: Raw document text.
        thresholds: Cutoffs to apply. Defaults to ``QualityThresholds()``.
        index: Value to stamp on the returned report's ``index`` field --
            callers iterating a list of documents pass the document's
            position here; a single ad-hoc call can leave it at 0.

    Returns:
        A ``QualityReport``. ``keep`` is ``True`` only when every rule passes.
    """
    thresholds = thresholds if thresholds is not None else QualityThresholds()
    reasons: list[str] = []
    stats: dict[str, float] = {}

    stripped = text.strip()
    char_count = len(stripped)
    stats["char_count"] = float(char_count)
    if char_count < thresholds.min_chars:
        reasons.append(f"only {char_count} chars (min {thresholds.min_chars})")

    words = _WORD_RE.findall(stripped)
    word_count = len(words)
    stats["word_count"] = float(word_count)
    if word_count < thresholds.min_words:
        reasons.append(f"only {word_count} words (min {thresholds.min_words})")

    non_space_count = sum(1 for c in stripped if not c.isspace())
    alpha_count = sum(1 for c in stripped if c.isalpha())
    alpha_ratio = alpha_count / non_space_count if non_space_count else 0.0
    stats["alpha_ratio"] = alpha_ratio
    if non_space_count > 0 and alpha_ratio < thresholds.min_alpha_ratio:
        reasons.append(f"alpha ratio {alpha_ratio:.2f} below min {thresholds.min_alpha_ratio}")

    symbol_count = sum(1 for c in stripped if c not in _COMMON_PUNCT and not c.isalnum())
    symbol_ratio = symbol_count / char_count if char_count else 0.0
    stats["symbol_ratio"] = symbol_ratio
    if char_count > 0 and symbol_ratio > thresholds.max_symbol_ratio:
        reasons.append(f"symbol ratio {symbol_ratio:.2f} above max {thresholds.max_symbol_ratio}")

    mean_word_length = (sum(len(w) for w in words) / word_count) if word_count else 0.0
    stats["mean_word_length"] = mean_word_length
    if mean_word_length > thresholds.max_mean_word_length:
        reasons.append(
            f"mean word length {mean_word_length:.1f} above max {thresholds.max_mean_word_length}"
        )

    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 5:
        line_chars = [len(line) for line in lines]
        total_line_chars = sum(line_chars)
        short_line_chars = sum(
            n for line, n in zip(lines, line_chars) if len(_WORD_RE.findall(line)) < 3
        )
        short_line_ratio = short_line_chars / total_line_chars if total_line_chars else 0.0
        stats["short_line_ratio"] = short_line_ratio
        if short_line_ratio > thresholds.max_short_line_ratio:
            reasons.append(
                f"short-line ratio {short_line_ratio:.2f} above max {thresholds.max_short_line_ratio}"
            )
    else:
        stats["short_line_ratio"] = 0.0

    lower_words = [w.lower() for w in words]
    if len(lower_words) >= 3:
        trigrams = [tuple(lower_words[i : i + 3]) for i in range(len(lower_words) - 2)]
        top_count = Counter(trigrams).most_common(1)[0][1]
        top_ngram_ratio = top_count / len(trigrams)
    else:
        top_ngram_ratio = 0.0
    stats["top_ngram_repetition_ratio"] = top_ngram_ratio
    if top_ngram_ratio > thresholds.max_top_ngram_repetition_ratio:
        reasons.append(
            f"top 3-gram repetition ratio {top_ngram_ratio:.2f} "
            f"above max {thresholds.max_top_ngram_repetition_ratio}"
        )

    return QualityReport(index=index, keep=not reasons, reasons=reasons, stats=stats)


def filter_low_quality(
    texts: list[str],
    thresholds: "QualityThresholds | None" = None,
) -> tuple[list[int], list[QualityReport]]:
    """Score every document and return which ones survive.

    Args:
        texts: Documents to evaluate.
        thresholds: Cutoffs to apply. Defaults to ``QualityThresholds()``.

    Returns:
        ``(kept_indices, reports)`` -- ``kept_indices`` is the sorted list
        of surviving document indices (same two-tuple shape as
        ``dedup.deduplicate_indices``'s ``(kept_indices, clusters)``, so
        callers wire both the same way) and ``reports`` is one
        ``QualityReport`` per input document, in input order, regardless
        of verdict.
    """
    reports = [score_document(text, thresholds, index=i) for i, text in enumerate(texts)]
    kept = [r.index for r in reports if r.keep]
    return kept, reports
