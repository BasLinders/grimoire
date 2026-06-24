"""Unit tests for StackExchange Q&A pair parsing and finetune-data conversion.

Coverage:
    parse_qa_text                — title/answer extraction, multiple answers
                                    per question, accepted flag, score parsing
                                    (incl. negative), malformed and empty
                                    blocks, trailing answer with no separator.
    load_qa_pairs                 — directory globbing, min_score filtering,
                                    accepted_only.
    qa_pair_to_finetune_example  — context/assistant extraction budgets,
                                    short-answer rejection, JSONL round-trip.
"""

import json
from pathlib import Path

import pytest

from grimoire_ai.llm.data.qa_pairs import (
    FinetuneExample,
    QAPair,
    load_qa_pairs,
    parse_qa_text,
    qa_pair_to_finetune_example,
    qa_pairs_to_finetune_examples,
)


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


# ---------------------------------------------------------------------------
# Instruction-tuning conversion
# ---------------------------------------------------------------------------

def _pair(answer: str, question: str = "How does grappling work?") -> QAPair:
    return QAPair(question=question, answer=answer, answer_score=10, accepted=True)


class TestQaPairToFinetuneExample:
    def test_user_is_the_question(self):
        ex = qa_pair_to_finetune_example(_pair("A grappled creature has its speed reduced to zero."))
        assert ex.user == "How does grappling work?"

    def test_context_and_assistant_are_leading_slices_of_the_answer(self):
        answer = (
            "A grappled creature has its speed reduced to zero. "
            "The condition ends if the grappler is incapacitated. "
            "It also ends if the creature is moved out of the grappler's reach."
        )
        ex = qa_pair_to_finetune_example(_pair(answer), context_max_chars=200, answer_max_chars=60)
        assert answer.startswith(ex.context)
        assert answer.startswith(ex.assistant)

    def test_assistant_is_no_longer_than_context(self):
        answer = (
            "A grappled creature has its speed reduced to zero. "
            "The condition ends if the grappler is incapacitated. "
            "It also ends if the creature is moved out of the grappler's reach."
        )
        ex = qa_pair_to_finetune_example(_pair(answer), context_max_chars=200, answer_max_chars=60)
        assert len(ex.assistant) <= len(ex.context)

    def test_respects_context_max_chars(self):
        long_answer = "This is one sentence. " * 50
        ex = qa_pair_to_finetune_example(_pair(long_answer), context_max_chars=100, answer_max_chars=50)
        assert len(ex.context) <= 100 or ex.context == "This is one sentence."  # single-sentence floor

    def test_respects_answer_max_chars(self):
        long_answer = "This is one sentence. " * 50
        ex = qa_pair_to_finetune_example(_pair(long_answer), context_max_chars=400, answer_max_chars=30)
        assert len(ex.assistant) <= 30 or ex.assistant == "This is one sentence."

    def test_does_not_cut_off_mid_sentence(self):
        answer = "Short first sentence here now. This second one is considerably longer than the budget allows."
        ex = qa_pair_to_finetune_example(_pair(answer), context_max_chars=40, answer_max_chars=40)
        # Either exactly the first sentence, or the first sentence is a clean prefix.
        assert ex.context == "Short first sentence here now."

    def test_single_long_sentence_exceeding_budget_is_still_returned(self):
        """A budget smaller than even the first sentence must not produce an
        empty string -- better to exceed the budget than drop the answer."""
        answer = "This single sentence is deliberately much longer than the tiny budget given to it here."
        ex = qa_pair_to_finetune_example(_pair(answer), context_max_chars=10, answer_max_chars=10)
        assert ex is not None
        assert ex.context == answer
        assert ex.assistant == answer

    def test_unpunctuated_line_list_does_not_blow_through_the_budget(self):
        """Regression: a list with one item per line and no terminal
        punctuation (observed on the real corpus -- an abbreviation table)
        looked like a single giant 'sentence' to a sentence-only splitter
        and exceeded the budget by 10x or more. Line breaks must be treated
        as unit boundaries too, not just '.', '!', '?'.
        """
        answer = "\n".join(f"Item{i} description for entry number {i}" for i in range(50))
        ex = qa_pair_to_finetune_example(_pair(answer), context_max_chars=100, answer_max_chars=50)
        assert len(ex.context) <= 100
        assert len(ex.assistant) <= 50

    def test_too_short_answer_returns_none(self):
        assert qa_pair_to_finetune_example(_pair("Yes.")) is None

    def test_empty_answer_returns_none(self):
        assert qa_pair_to_finetune_example(_pair("")) is None

    def test_default_budgets_keep_answer_shorter_than_context_budget(self):
        ex = qa_pair_to_finetune_example(_pair("This is a perfectly normal-length community answer. " * 5))
        assert len(ex.assistant) <= 350
        assert len(ex.context) <= 600


class TestFinetuneExample:
    def test_to_json_line_has_conversationdataset_fields(self):
        ex = FinetuneExample(user="Q?", assistant="A.", context="Ctx.")
        obj = json.loads(ex.to_json_line())
        assert obj == {"user": "Q?", "assistant": "A.", "context": "Ctx."}

    def test_to_json_line_is_a_single_line(self):
        ex = FinetuneExample(user="Q?", assistant="A.", context="Ctx.")
        assert "\n" not in ex.to_json_line()


class TestQaPairsToFinetuneExamples:
    def test_converts_multiple_pairs(self):
        pairs = [
            _pair("A grappled creature has its speed reduced to zero.", "Q1?"),
            _pair("You roll the d20 twice and take the higher result.", "Q2?"),
        ]
        examples = qa_pairs_to_finetune_examples(pairs)
        assert len(examples) == 2
        assert {e.user for e in examples} == {"Q1?", "Q2?"}

    def test_drops_pairs_too_short_to_split(self):
        pairs = [
            _pair("A grappled creature has its speed reduced to zero.", "Q1?"),
            _pair("Yes.", "Q2?"),
        ]
        examples = qa_pairs_to_finetune_examples(pairs)
        assert len(examples) == 1
        assert examples[0].user == "Q1?"
