"""Integration test for grimoire_ai.llm.data.preprocessing.preprocess().

First test file for preprocess() itself (previously only its constituent
pieces -- dedup.py, the BPE encoder -- had direct tests). Covers the new
--quality-filter wiring (docs/architecture_optimization.md item #9).
"""

import json

import numpy as np

from grimoire_ai.llm.data.preprocessing import preprocess
from grimoire_ai.llm.tokenizer.special_tokens import EOS_ID

_CLEAN_TEXT = (
    "A grappled creature has its speed reduced to zero, and it cannot "
    "benefit from any bonus to its speed until the condition ends. "
    "The condition ends if the grappler is incapacitated, or if an "
    "effect removes the grappled creature from the grappler's reach."
)


def test_quality_filter_removes_junk_file_before_tokenising(tmp_path) -> None:
    """A document that fails the quality filter must never contribute
    tokens to the output .bin -- only the clean document should survive."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "clean.txt").write_text(_CLEAN_TEXT, encoding="utf-8")
    (corpus_dir / "junk.txt").write_text("x", encoding="utf-8")  # fails min_chars trivially

    output_path = tmp_path / "corpus.bin"
    vocab_path = tmp_path / "bpe.json"
    report_path = tmp_path / "quality_report.jsonl"

    total_tokens = preprocess(
        input_path=str(corpus_dir),
        output_path=str(output_path),
        vocab_path=str(vocab_path),
        vocab_size=300,
        quality_filter=True,
        quality_report_path=str(report_path),
    )

    assert total_tokens > 0
    ids = np.fromfile(str(output_path), dtype=np.int32)
    n_docs = int((ids == EOS_ID).sum())
    assert n_docs == 1, (
        "Only the clean document should have been tokenised -- "
        "junk.txt must have been dropped by the quality filter."
    )

    report_lines = report_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(report_lines) == 1
    dropped = json.loads(report_lines[0])
    assert dropped["reasons"]


def test_quality_filter_off_by_default_keeps_all_documents(tmp_path) -> None:
    """Without --quality-filter, even a document that would fail every rule
    must still be tokenised -- the flag must be strictly opt-in."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "clean.txt").write_text(_CLEAN_TEXT, encoding="utf-8")
    (corpus_dir / "junk.txt").write_text("x", encoding="utf-8")

    output_path = tmp_path / "corpus.bin"
    vocab_path = tmp_path / "bpe.json"

    preprocess(
        input_path=str(corpus_dir),
        output_path=str(output_path),
        vocab_path=str(vocab_path),
        vocab_size=300,
    )

    ids = np.fromfile(str(output_path), dtype=np.int32)
    n_docs = int((ids == EOS_ID).sum())
    assert n_docs == 2, "Both documents must be tokenised when quality_filter is off."


def test_quality_report_not_written_when_nothing_dropped(tmp_path) -> None:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "clean.txt").write_text(_CLEAN_TEXT, encoding="utf-8")

    output_path = tmp_path / "corpus.bin"
    vocab_path = tmp_path / "bpe.json"
    report_path = tmp_path / "quality_report.jsonl"

    preprocess(
        input_path=str(corpus_dir),
        output_path=str(output_path),
        vocab_path=str(vocab_path),
        vocab_size=300,
        quality_filter=True,
        quality_report_path=str(report_path),
    )

    assert not report_path.exists()


def test_input_path_accepts_a_list_of_directories(tmp_path) -> None:
    """input_path may be a list of directories -- lets a caller combine
    sibling corpus directories (e.g. a base scrape plus a derived/synthetic
    directory kept separate on disk) into one build without merging them."""
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "a.txt").write_text(_CLEAN_TEXT, encoding="utf-8")

    derived_dir = tmp_path / "derived"
    derived_dir.mkdir()
    (derived_dir / "b.txt").write_text(_CLEAN_TEXT, encoding="utf-8")

    output_path = tmp_path / "corpus.bin"
    vocab_path = tmp_path / "bpe.json"

    preprocess(
        input_path=[str(base_dir), str(derived_dir)],
        output_path=str(output_path),
        vocab_path=str(vocab_path),
        vocab_size=300,
    )

    ids = np.fromfile(str(output_path), dtype=np.int32)
    n_docs = int((ids == EOS_ID).sum())
    assert n_docs == 2, "Both directories' documents must be tokenised."
