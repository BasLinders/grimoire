"""Tests for the Saga fine-tuning dataset and related scripts.

Covers:
  - JSONL structure: every line has required fields, valid JSON.
  - Content: all three domains represented (D&D rules, encounter math,
    probability/statistics).
  - ConversationDataset: loads without error, produces correct tensor shapes,
    response-only masking is applied.
  - validate_finetune_data script: runs without error on the real dataset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_DATA = Path(__file__).parent.parent / "scripts" / "finetune_data" / "saga_v1.jsonl"


# ---------------------------------------------------------------------------
# JSONL structure
# ---------------------------------------------------------------------------

def _examples() -> list[dict]:
    return [json.loads(l) for l in _DATA.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_dataset_file_exists():
    assert _DATA.exists(), "data/finetune/saga_v1.jsonl is missing."


def test_all_examples_have_user_and_assistant():
    for i, ex in enumerate(_examples(), 1):
        assert "user" in ex,      f"Example {i} missing 'user' field."
        assert "assistant" in ex, f"Example {i} missing 'assistant' field."


def test_no_empty_fields():
    for i, ex in enumerate(_examples(), 1):
        assert ex["user"].strip(),      f"Example {i} has empty 'user'."
        assert ex["assistant"].strip(), f"Example {i} has empty 'assistant'."


def test_at_least_twenty_examples():
    assert len(_examples()) >= 20, "Dataset should have at least 20 examples."


def test_context_field_is_string_when_present():
    for i, ex in enumerate(_examples(), 1):
        if "context" in ex:
            assert isinstance(ex["context"], str), f"Example {i}: 'context' must be a string."


# ---------------------------------------------------------------------------
# Domain coverage
# ---------------------------------------------------------------------------

def test_covers_dnd_rules():
    texts = " ".join(ex["user"] + " " + ex["assistant"] for ex in _examples()).lower()
    assert any(kw in texts for kw in ("grappled", "saving throw", "condition", "spell slot"))


def test_covers_encounter_math():
    texts = " ".join(ex["user"] + " " + ex["assistant"] for ex in _examples()).lower()
    assert any(kw in texts for kw in ("xp", "challenge rating", "multiplier", "deadly"))


def test_covers_probability():
    texts = " ".join(ex["user"] + " " + ex["assistant"] for ex in _examples()).lower()
    assert any(kw in texts for kw in ("probability", "average", "expected", "percent", "chance"))


# ---------------------------------------------------------------------------
# ConversationDataset integration
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_tokenizer(tmp_path_factory):
    """Train and return a minimal BPE tokenizer on the dataset text."""
    from grimoire.llm.tokenizer.bpe import BytePairEncoder

    examples = _examples()
    corpus = [ex["user"] + " " + ex["assistant"] + " " + ex.get("context", "") for ex in examples]
    enc = BytePairEncoder()
    enc.train(corpus, vocab_size=512)
    path = str(tmp_path_factory.mktemp("tok") / "bpe.json")
    enc.save(path)
    return enc


def test_conversation_dataset_loads(tiny_tokenizer):
    from grimoire.llm.data.conversation import ConversationDataset

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=256)
    assert len(ds) > 0


def test_conversation_dataset_returns_tensors(tiny_tokenizer):
    from grimoire.llm.data.conversation import ConversationDataset

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=256)
    inp, tgt = ds[0]
    assert isinstance(inp, torch.Tensor)
    assert isinstance(tgt, torch.Tensor)
    assert inp.dtype == torch.long
    assert tgt.dtype == torch.long


def test_conversation_dataset_same_length(tiny_tokenizer):
    from grimoire.llm.data.conversation import ConversationDataset

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=256)
    inp, tgt = ds[0]
    assert inp.shape == tgt.shape, "input_ids and target_ids must have the same length."


def test_conversation_dataset_respects_max_seq_len(tiny_tokenizer):
    from grimoire.llm.data.conversation import ConversationDataset

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=128)
    for i in range(len(ds)):
        inp, tgt = ds[i]
        assert inp.shape[0] <= 128, f"Example {i} input exceeds max_seq_len."


def test_response_masking_has_pad_at_start(tiny_tokenizer):
    """The target tensor should begin with PAD tokens (prompt is masked)."""
    from grimoire.llm.data.conversation import ConversationDataset
    from grimoire.llm.tokenizer.special_tokens import PAD_ID

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=256)
    _, tgt = ds[0]
    # At least the BOS position should be masked.
    assert tgt[0].item() == PAD_ID, "First target token should be PAD (prompt masking)."


def test_response_masking_has_non_pad_at_end(tiny_tokenizer):
    """The target tensor should end with real tokens (the response)."""
    from grimoire.llm.data.conversation import ConversationDataset
    from grimoire.llm.tokenizer.special_tokens import PAD_ID

    ds = ConversationDataset(path=str(_DATA), tokenizer=tiny_tokenizer, max_seq_len=256)
    _, tgt = ds[0]
    assert tgt[-1].item() != PAD_ID, "Last target token should not be PAD (response present)."


# ---------------------------------------------------------------------------
# validate_finetune_data script
# ---------------------------------------------------------------------------

def test_validate_script_runs_without_error(tmp_path):
    """validate_finetune_data.py should exit 0 on the real dataset when a tokenizer exists."""
    from grimoire.llm.tokenizer.bpe import BytePairEncoder

    examples = _examples()
    corpus = [ex["user"] + " " + ex["assistant"] + " " + ex.get("context", "") for ex in examples]
    enc = BytePairEncoder()
    enc.train(corpus, vocab_size=512)
    vocab_path = str(tmp_path / "bpe.json")
    enc.save(vocab_path)

    script = Path(__file__).parent.parent / "scripts" / "validate_finetune_data.py"
    result = subprocess.run(
        [sys.executable, str(script),
         "--data", str(_DATA),
         "--vocab", vocab_path,
         "--max-seq-len", "512"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "All examples valid." in result.stdout
