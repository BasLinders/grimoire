"""Tests for ConversationDataset.

Gate criteria:
- Dataset loads valid JSONL and returns the correct number of examples.
- Input/target have the same length (causal shift: both are len(full) - 1).
- Prompt portion of the target is fully masked to PAD_ID.
- Response portion of the target is not all PAD.
- With a context field, SEP tokens appear in the input.
- Without a context field, no SEP in the input.
- Input always starts with BOS_ID.
- Final non-PAD target token is EOS_ID.
- Missing file raises FileNotFoundError.
- Empty file raises ValueError.
- Sequences longer than max_seq_len are truncated and end with EOS.
- Response-only loss: cross_entropy on a masked target equals loss on
  response tokens only.
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from grimoire.llm.data.conversation import ConversationDataset
from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    USR_ID,
)


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


def _write_jsonl(tmp_dir: str, examples: list[dict]) -> str:
    path = str(Path(tmp_dir) / "examples.jsonl")
    Path(path).write_text(
        "\n".join(json.dumps(e) for e in examples), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_loads_examples(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [
            {"user": "What is grapple?", "assistant": "It reduces speed to zero."},
            {"user": "What is a fox?",   "assistant": "A quick brown animal."},
        ])
        ds = ConversationDataset(path, tokenizer)
    assert len(ds) == 2


def test_missing_file_raises() -> None:
    enc = BytePairEncoder()
    with pytest.raises(FileNotFoundError):
        ConversationDataset("/tmp/grimoire_no_such_finetune.jsonl", enc)


def test_empty_file_raises(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "empty.jsonl")
        Path(path).write_text("", encoding="utf-8")
        with pytest.raises(ValueError):
            ConversationDataset(path, tokenizer)


# ---------------------------------------------------------------------------
# Sequence structure
# ---------------------------------------------------------------------------

def test_input_target_same_length(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [
            {"user": "hello", "assistant": "world"},
        ])
        ds = ConversationDataset(path, tokenizer)
    inp, tgt = ds[0]
    assert inp.shape == tgt.shape, "input and target must have the same length."


def test_input_starts_with_bos(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "hi", "assistant": "hello"}])
        ds = ConversationDataset(path, tokenizer)
    inp, _ = ds[0]
    assert inp[0].item() == BOS_ID


def test_no_context_no_sep(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "hi", "assistant": "hello"}])
        ds = ConversationDataset(path, tokenizer)
    inp, _ = ds[0]
    assert SEP_ID not in inp.tolist(), "No context field should mean no SEP token."


def test_with_context_has_sep(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{
            "user": "what is grapple",
            "assistant": "reduces speed",
            "context": "A grappled creature has its speed reduced to zero.",
        }])
        ds = ConversationDataset(path, tokenizer)
    inp, _ = ds[0]
    assert inp.tolist().count(SEP_ID) == 2, "Context field should produce two SEP tokens."


def test_final_response_token_is_eos(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "hi", "assistant": "hello there"}])
        ds = ConversationDataset(path, tokenizer)
    _, tgt = ds[0]
    non_pad = [t for t in tgt.tolist() if t != PAD_ID]
    assert non_pad[-1] == EOS_ID, "Last non-PAD target token must be EOS."


# ---------------------------------------------------------------------------
# Loss masking
# ---------------------------------------------------------------------------

def test_prompt_tokens_masked_in_target(tokenizer: BytePairEncoder) -> None:
    """All target positions corresponding to prompt tokens must be PAD_ID."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "what is speed", "assistant": "fast"}])
        ds = ConversationDataset(path, tokenizer)
    inp, tgt = ds[0]

    # Find where AST appears in the input — everything before it is prompt.
    inp_list = inp.tolist()
    ast_pos = inp_list.index(AST_ID)
    assert all(t == PAD_ID for t in tgt[:ast_pos].tolist()), (
        "All target positions before AST must be PAD_ID."
    )


def test_response_tokens_not_all_pad(tokenizer: BytePairEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "q", "assistant": "answer here"}])
        ds = ConversationDataset(path, tokenizer)
    _, tgt = ds[0]
    assert any(t != PAD_ID for t in tgt.tolist()), (
        "At least some target tokens must be non-PAD (the response)."
    )


def test_response_only_loss_equivalence(tokenizer: BytePairEncoder) -> None:
    """cross_entropy with PAD ignore_index equals manual response-only loss."""
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "what", "assistant": "this is the answer"}])
        ds = ConversationDataset(path, tokenizer)
    inp, tgt = ds[0]
    vocab_size = tokenizer.vocab_size

    # Fake uniform logits (batch=1).
    logits = torch.zeros(1, len(inp), vocab_size)

    masked_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        tgt.view(-1),
        ignore_index=PAD_ID,
    )

    # Manual: compute loss only over non-PAD positions.
    mask = tgt != PAD_ID
    manual_loss = F.cross_entropy(
        logits.view(-1, vocab_size)[mask],
        tgt.view(-1)[mask],
    )

    assert torch.isclose(masked_loss, manual_loss, atol=1e-5), (
        "Masked cross_entropy must equal manual response-only loss."
    )


# ---------------------------------------------------------------------------
# Truncation
# ---------------------------------------------------------------------------

def test_truncation_preserves_eos(tokenizer: BytePairEncoder) -> None:
    """Long examples must be truncated to max_seq_len and end with EOS."""
    long_answer = "the quick brown fox " * 100
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_jsonl(tmp, [{"user": "tell me", "assistant": long_answer}])
        ds = ConversationDataset(path, tokenizer, max_seq_len=32)
    inp, tgt = ds[0]
    assert len(inp) <= 32, f"Input length {len(inp)} exceeds max_seq_len=32."
    non_pad = [t for t in tgt.tolist() if t != PAD_ID]
    assert non_pad[-1] == EOS_ID, "Truncated example must still end with EOS."
