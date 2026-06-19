"""Tests for the fine-tune step auto-suggestion in the Gradio UI.

``_suggest_ft_steps_from_data`` derives a total step count from the
fine-tune dataset's example count rather than Chinchilla's pretraining
token/parameter ratio, since overfitting a small dataset — not
undertraining — is the dominant risk when fine-tuning an existing
checkpoint. LoRA adapter runs use a separate, more generous tier table
since freezing the base weights makes overfitting far less likely than
with full fine-tuning.
"""

from __future__ import annotations

import json

import pytest

gr = pytest.importorskip("gradio")

from grimoire_ai.ui import app  # noqa: E402

_BASE = "Base (instruction fine-tune)"
_AGENT = "Agent (LoRA adapter)"


def _write_jsonl(tmp_path, n: int):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"user": f"q{i}", "assistant": f"a{i}"}) + "\n")
    return str(path)


def test_blank_path_returns_no_update():
    updates = app._suggest_ft_steps_from_data("", 4, 4, _BASE)
    assert all(u == gr.update() for u in updates)


def test_missing_file_returns_no_update(tmp_path):
    updates = app._suggest_ft_steps_from_data(str(tmp_path / "missing.jsonl"), 4, 4, _BASE)
    assert all(u == gr.update() for u in updates)


def test_small_dataset_uses_fewer_epochs(tmp_path):
    path = _write_jsonl(tmp_path, 10)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 1, 1, _BASE)
    # 10 examples, 2-epoch tier, effective batch 1 -> 20 steps.
    assert steps_update["value"] == 20


def test_large_dataset_uses_more_epochs(tmp_path):
    path = _write_jsonl(tmp_path, 1000)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 4, 4, _BASE)
    # 1000 examples, default 6 epochs (>500 tier), effective batch 16 -> 375 steps.
    assert steps_update["value"] == 375


def test_step_floor_is_respected(tmp_path):
    path = _write_jsonl(tmp_path, 2)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 64, 4, _BASE)
    assert steps_update["value"] >= 10


def test_lora_mode_uses_more_epochs_than_base(tmp_path):
    path = _write_jsonl(tmp_path, 111)
    base_steps, *_ = app._suggest_ft_steps_from_data(path, 4, 1, _BASE)
    lora_steps, *_ = app._suggest_ft_steps_from_data(path, 4, 1, _AGENT)
    assert lora_steps["value"] > base_steps["value"]


def test_lora_small_dataset_uses_lora_tier(tmp_path):
    path = _write_jsonl(tmp_path, 10)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 1, 1, _AGENT)
    # 10 examples, 4-epoch LoRA tier, effective batch 1 -> 40 steps.
    assert steps_update["value"] == 40


def test_lora_step_floor_is_respected(tmp_path):
    path = _write_jsonl(tmp_path, 2)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 64, 4, _AGENT)
    assert steps_update["value"] >= 10
