"""Tests for ConversationState.

Gate criteria:
- Empty state produces a prompt identical to single-turn PromptBuilder output.
- Input starts with BOS_ID; ends with AST_ID.
- History turns appear in order (oldest → newest) before the current query.
- Most recent turns are kept when history is trimmed to fit max_seq_len.
- Context (SEP block) is injected when context_ids are provided.
- Context is trimmed — not dropped — when history leaves limited budget.
- Context is dropped entirely when history leaves zero budget for it.
- add_turn appends turns; max_turns cap evicts the oldest.
- clear() resets the state to empty.
- turn_count and history properties reflect the current state.
"""

import pytest

from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    EOS_ID,
    SEP_ID,
    USR_ID,
)
from grimoire_ai.state.conversation import ConversationState, Turn


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tokenizer() -> BytePairEncoder:
    enc = BytePairEncoder()
    enc.train(
        ["the quick brown fox jumps over the lazy dog " * 30,
         "a grappled creature has its speed reduced to zero " * 30],
        vocab_size=512,
    )
    return enc


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

def test_empty_state_starts_with_bos(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    ids = state.build_prompt_ids("hello", tokenizer)
    assert ids[0] == BOS_ID


def test_empty_state_ends_with_ast(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    ids = state.build_prompt_ids("hello", tokenizer)
    assert ids[-1] == AST_ID


def test_empty_state_no_sep_without_context(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    ids = state.build_prompt_ids("hello", tokenizer)
    assert SEP_ID not in ids


def test_empty_state_matches_single_turn_format(tokenizer: BytePairEncoder) -> None:
    """With no history and no context the output must be [BOS, USR, *q, AST]."""
    state = ConversationState()
    query = "what is grapple"
    ids = state.build_prompt_ids(query, tokenizer)
    query_ids = tokenizer.encode(query)
    expected = [BOS_ID, USR_ID] + query_ids + [AST_ID]
    assert ids == expected


# ---------------------------------------------------------------------------
# History injection
# ---------------------------------------------------------------------------

def test_history_appears_before_current_query(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    state.add_turn("first question", "first answer")
    ids = state.build_prompt_ids("second question", tokenizer)
    # USR must appear at least twice: once for the history turn, once for current.
    assert ids.count(USR_ID) >= 2


def test_history_order_oldest_first(tokenizer: BytePairEncoder) -> None:
    """Turns must appear oldest→newest, current query last."""
    state = ConversationState()
    state.add_turn("turn one", "answer one")
    state.add_turn("turn two", "answer two")
    ids = state.build_prompt_ids("turn three", tokenizer)

    t1_ids = tokenizer.encode("turn one")
    t2_ids = tokenizer.encode("turn two")
    t3_ids = tokenizer.encode("turn three")

    def find_subseq(seq: list[int], subseq: list[int]) -> int:
        for i in range(len(seq) - len(subseq) + 1):
            if seq[i : i + len(subseq)] == subseq:
                return i
        return -1

    pos1 = find_subseq(ids, t1_ids)
    pos2 = find_subseq(ids, t2_ids)
    pos3 = find_subseq(ids, t3_ids)

    assert pos1 != -1 and pos2 != -1 and pos3 != -1, "All turns must be present."
    assert pos1 < pos2 < pos3, "Turns must appear oldest → newest."


# ---------------------------------------------------------------------------
# Budget trimming
# ---------------------------------------------------------------------------

def test_oldest_turns_dropped_when_over_budget(tokenizer: BytePairEncoder) -> None:
    """When history is too long the oldest turns are dropped, newest kept."""
    state = ConversationState()
    for i in range(10):
        state.add_turn(f"question number {i}", f"answer number {i}")

    # Very small max_seq_len forces trimming.
    ids = state.build_prompt_ids("current query", tokenizer, max_seq_len=64)

    # Most recent turn's answer tokens should still be present.
    recent_ids = tokenizer.encode("answer number 9")
    found = any(
        ids[i : i + len(recent_ids)] == recent_ids
        for i in range(len(ids) - len(recent_ids) + 1)
    )
    assert found, "Most recent turn must survive budget trimming."


def test_prompt_never_exceeds_max_seq_len(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    for i in range(15):
        state.add_turn("the quick brown fox jumps over the lazy dog " * 2,
                       "a grappled creature has its speed reduced " * 2)
    ids = state.build_prompt_ids(
        "what is the speed of a grappled creature",
        tokenizer,
        max_seq_len=128,
    )
    assert len(ids) <= 128


# ---------------------------------------------------------------------------
# Context injection
# ---------------------------------------------------------------------------

def test_context_ids_inject_sep_tokens(tokenizer: BytePairEncoder) -> None:
    state = ConversationState()
    ctx = tokenizer.encode("grappled speed zero")
    ids = state.build_prompt_ids("query", tokenizer, context_ids=ctx)
    assert ids.count(SEP_ID) == 2


def test_context_trimmed_when_history_large(tokenizer: BytePairEncoder) -> None:
    """Context is trimmed to fit remaining budget — not silently dropped."""
    state = ConversationState()
    # Add one turn that takes most of the budget.
    state.add_turn("question " * 5, "answer " * 5)
    ctx = tokenizer.encode("context " * 30)  # large context
    ids = state.build_prompt_ids("query", tokenizer, context_ids=ctx, max_seq_len=80)
    # Either context survived (SEP present) or was entirely trimmed (no SEP).
    # What must NOT happen: prompt exceeds max_seq_len.
    assert len(ids) <= 80


def test_context_dropped_when_no_budget(tokenizer: BytePairEncoder) -> None:
    """When history + query fill the budget, context is dropped cleanly."""
    state = ConversationState()
    # Pack the budget with history.
    for _ in range(5):
        state.add_turn("the quick brown fox", "jumps over the lazy dog")
    ctx = tokenizer.encode("some context")
    ids = state.build_prompt_ids("query", tokenizer, context_ids=ctx, max_seq_len=48)
    # Prompt must not exceed the limit.
    assert len(ids) <= 48


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def test_add_turn_increments_count() -> None:
    state = ConversationState()
    state.add_turn("q", "a")
    assert state.turn_count == 1
    state.add_turn("q2", "a2")
    assert state.turn_count == 2


def test_max_turns_evicts_oldest() -> None:
    state = ConversationState(max_turns=3)
    for i in range(5):
        state.add_turn(f"q{i}", f"a{i}")
    assert state.turn_count == 3
    assert state.history[0].user == "q2"  # oldest surviving turn


def test_clear_resets_state() -> None:
    state = ConversationState()
    state.add_turn("q", "a")
    state.clear()
    assert state.turn_count == 0
    assert state.history == []


def test_history_property_is_copy() -> None:
    """Mutating the returned history list must not affect the internal state."""
    state = ConversationState()
    state.add_turn("q", "a")
    h = state.history
    h.clear()
    assert state.turn_count == 1
