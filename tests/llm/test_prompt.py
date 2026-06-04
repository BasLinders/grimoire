"""Tests for PromptBuilder.

Gate criteria:
- Prompt always starts with BOS.
- USR token appears before AST token.
- AST token is the last token in the sequence.
- Context block (SEP…SEP) is present when results contain next_tokens.
- Context block is absent when results list is empty or all next_tokens are None.
- Context is trimmed when it would exceed the budget.
- Query-only path (no results) produces a valid prompt.
"""

import pytest

from grimoire.corpus.corpus import QueryResult
from grimoire.llm.inference.prompt import PromptBuilder
from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    SEP_ID,
    USR_ID,
)


# ---------------------------------------------------------------------------
# Fixture: tiny trained tokenizer
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer() -> BytePairEncoder:
    """A minimal BPE tokenizer trained on a short corpus."""
    enc = BytePairEncoder()
    corpus = [
        "the quick brown fox jumps over the lazy dog " * 20,
        "a grappled creature loses its speed " * 20,
        "fire bolt scorching ray magic missile " * 20,
    ]
    enc.train(corpus, vocab_size=512)
    return enc


def _result(next_token: str | None, score: float = 1.0) -> QueryResult:
    return QueryResult(
        multi_token=("a", "b", "c", "d"),
        next_token=next_token,
        score=score,
        source=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prompt_starts_with_bos(tokenizer: BytePairEncoder) -> None:
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    ids = builder.build("hello world")
    assert ids[0] == BOS_ID, "Prompt must start with BOS."


def test_prompt_ends_with_ast(tokenizer: BytePairEncoder) -> None:
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    ids = builder.build("hello world")
    assert ids[-1] == AST_ID, "Prompt must end with AST."


def test_usr_before_ast(tokenizer: BytePairEncoder) -> None:
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    ids = builder.build("hello world")
    usr_pos = ids.index(USR_ID)
    ast_pos = ids.index(AST_ID)
    assert usr_pos < ast_pos, "USR must appear before AST."


def test_no_results_no_sep(tokenizer: BytePairEncoder) -> None:
    """Without corpus results the SEP token should not appear."""
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    ids = builder.build("hello world", results=[])
    assert SEP_ID not in ids, "SEP should be absent when there are no results."


def test_with_results_has_sep(tokenizer: BytePairEncoder) -> None:
    """When results include next_tokens, SEP should appear twice (open + close)."""
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    results = [_result("grappl"), _result("speed")]
    ids = builder.build("what is grapple", results=results)
    sep_count = ids.count(SEP_ID)
    assert sep_count == 2, f"Expected 2 SEP tokens, got {sep_count}."


def test_results_with_no_next_tokens_no_sep(tokenizer: BytePairEncoder) -> None:
    """If all results have next_token=None, SEP should still be absent."""
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    results = [_result(None), _result(None)]
    ids = builder.build("query", results=results)
    assert SEP_ID not in ids


def test_context_trimmed_within_budget(tokenizer: BytePairEncoder) -> None:
    """The full prompt must not exceed max_context_tokens."""
    budget = 30
    builder = PromptBuilder(tokenizer, max_context_tokens=budget)
    # Many results to stress the budget.
    results = [_result(f"word{i}") for i in range(50)]
    ids = builder.build("short query", results=results)
    assert len(ids) <= budget, (
        f"Prompt length {len(ids)} exceeds budget {budget}."
    )


def test_query_only_structure(tokenizer: BytePairEncoder) -> None:
    """Query-only prompt: [BOS, USR, *query_ids, AST]."""
    builder = PromptBuilder(tokenizer, max_context_tokens=256)
    ids = builder.build("fire bolt")
    assert ids[0] == BOS_ID
    assert ids[1] == USR_ID
    assert ids[-1] == AST_ID
    # No SEP anywhere.
    assert SEP_ID not in ids
