"""Unit tests for StackExchange Q&A pair parsing.

Coverage:
    parse_qa_text  — title/answer extraction, multiple answers per question,
                     accepted flag, score parsing (incl. negative), malformed
                     and empty blocks, trailing answer with no separator.
    load_qa_pairs  — directory globbing, min_score filtering, accepted_only.
"""

from pathlib import Path

import pytest

from grimoire_ai.llm.data.qa_pairs import QAPair, load_qa_pairs, parse_qa_text


# A faithful mock of the scraper's output format (two questions, the first
# with an accepted answer plus a second answer, the second with one answer).
SAMPLE = """\
# What happens to a creature's speed when grappled?
Score: 42

I'm confused about how grappling affects movement.

## Answer (accepted)  (score: 215)

A grappled creature has its speed reduced to zero until the grapple ends.

## Answer  (score: 13)

It cannot move, full stop.

---
# How does advantage work?
Score: 18

When do I roll twice?

## Answer  (score: 7)

You roll two d20s and take the higher result.

---
"""


class TestParseQAText:
    def test_extracts_all_pairs(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert len(pairs) == 3

    def test_question_title_is_the_query(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert pairs[0].question == "What happens to a creature's speed when grappled?"
        assert pairs[2].question == "How does advantage work?"

    def test_answer_body_is_captured_and_stripped(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert pairs[0].answer == (
            "A grappled creature has its speed reduced to zero until the grapple ends."
        )

    def test_multiple_answers_share_question(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert pairs[0].question == pairs[1].question
        assert pairs[0].answer != pairs[1].answer

    def test_accepted_flag(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert pairs[0].accepted is True
        assert pairs[1].accepted is False
        assert pairs[2].accepted is False

    def test_scores_parsed(self):
        pairs = list(parse_qa_text(SAMPLE))
        assert pairs[0].answer_score == 215
        assert pairs[1].answer_score == 13
        assert pairs[2].answer_score == 7

    def test_question_score_line_not_treated_as_answer_body(self):
        """The 'Score: N' line is question metadata, never answer text."""
        pairs = list(parse_qa_text(SAMPLE))
        for p in pairs:
            assert "Score:" not in p.answer

    def test_negative_answer_score(self):
        text = (
            "# A bad question\n"
            "Score: -5\n\n"
            "## Answer  (score: -3)\n\n"
            "A downvoted answer.\n\n"
            "---\n"
        )
        pairs = list(parse_qa_text(text))
        assert len(pairs) == 1
        assert pairs[0].answer_score == -3

    def test_question_with_no_answers_yields_nothing(self):
        text = "# Lonely question\nScore: 4\n\nNo answers here.\n\n---\n"
        assert list(parse_qa_text(text)) == []

    def test_empty_answer_body_skipped(self):
        text = (
            "# A question\nScore: 1\n\n"
            "## Answer  (score: 5)\n\n"
            "\n\n"
            "---\n"
        )
        assert list(parse_qa_text(text)) == []

    def test_trailing_answer_without_separator(self):
        """An answer at EOF with no closing '---' is still captured."""
        text = (
            "# Final question\nScore: 1\n\n"
            "## Answer  (score: 9)\n\n"
            "The last answer, no trailing separator."
        )
        pairs = list(parse_qa_text(text))
        assert len(pairs) == 1
        assert pairs[0].answer == "The last answer, no trailing separator."

    def test_answer_header_not_mistaken_for_question(self):
        """'## Answer' must never be parsed as a '# ' question title."""
        pairs = list(parse_qa_text(SAMPLE))
        assert all(not p.question.startswith("Answer") for p in pairs)

    def test_multiline_answer_body_preserved(self):
        text = (
            "# Multi\nScore: 1\n\n"
            "## Answer  (score: 5)\n\n"
            "First line.\n\nSecond line.\n\n"
            "---\n"
        )
        pairs = list(parse_qa_text(text))
        assert pairs[0].answer == "First line.\n\nSecond line."

    def test_empty_document(self):
        assert list(parse_qa_text("")) == []


class TestLoadQAPairs:
    def _write(self, tmp_path: Path, name: str, text: str) -> None:
        (tmp_path / name).write_text(text, encoding="utf-8")

    def test_loads_from_directory(self, tmp_path):
        self._write(tmp_path, "rpg_se_0000.txt", SAMPLE)
        pairs = load_qa_pairs(tmp_path, min_score=1)
        assert len(pairs) == 3
        assert all(isinstance(p, QAPair) for p in pairs)

    def test_min_score_filters_low_answers(self, tmp_path):
        self._write(tmp_path, "rpg_se_0000.txt", SAMPLE)
        # Scores present: 215, 13, 7. min_score=10 keeps the first two.
        pairs = load_qa_pairs(tmp_path, min_score=10)
        assert len(pairs) == 2
        assert {p.answer_score for p in pairs} == {215, 13}

    def test_accepted_only(self, tmp_path):
        self._write(tmp_path, "rpg_se_0000.txt", SAMPLE)
        pairs = load_qa_pairs(tmp_path, accepted_only=True)
        assert len(pairs) == 1
        assert pairs[0].accepted is True

    def test_accepted_only_ignores_min_score(self, tmp_path):
        """accepted_only keeps the accepted answer regardless of its score."""
        text = (
            "# Q\nScore: 1\n\n"
            "## Answer (accepted)  (score: -2)\n\n"
            "Accepted but downvoted.\n\n"
            "---\n"
        )
        self._write(tmp_path, "rpg_se_0000.txt", text)
        pairs = load_qa_pairs(tmp_path, min_score=5, accepted_only=True)
        assert len(pairs) == 1

    def test_pattern_selects_only_matching_files(self, tmp_path):
        self._write(tmp_path, "rpg_se_0000.txt", SAMPLE)
        self._write(tmp_path, "5etools_actions_0000.txt", "# Not a Q&A file\nirrelevant\n")
        pairs = load_qa_pairs(tmp_path, min_score=1)
        assert len(pairs) == 3  # the 5etools file is excluded by the glob

    def test_multiple_files_concatenated(self, tmp_path):
        self._write(tmp_path, "rpg_se_0000.txt", SAMPLE)
        self._write(tmp_path, "rpg_se_0001.txt", SAMPLE)
        pairs = load_qa_pairs(tmp_path, min_score=1)
        assert len(pairs) == 6

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_qa_pairs(tmp_path / "does_not_exist")

    def test_no_matching_files_returns_empty(self, tmp_path):
        pairs = load_qa_pairs(tmp_path, min_score=1)
        assert pairs == []
