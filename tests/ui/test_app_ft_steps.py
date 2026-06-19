"""Tests for the fine-tune step auto-suggestion in the Gradio UI.

``_suggest_ft_steps_from_data`` derives a total step count from the
fine-tune dataset's example count rather than Chinchilla's pretraining
token/parameter ratio, since overfitting a small dataset — not
undertraining — is the dominant risk when fine-tuning an existing
checkpoint.
"""

from __future__ import annotations

import json

import pytest

gr = pytest.importorskip("gradio")

from grimoire_ai.ui import app  # noqa: E402


def _write_jsonl(tmp_path, n: int):
    path = tmp_path / "data.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"user": f"q{i}", "assistant": f"a{i}"}) + "\n")
    return str(path)


def test_blank_path_returns_no_update():
    updates = app._suggest_ft_steps_from_data("", 4, 4)
    assert all(u == gr.update() for u in updates)


def test_missing_file_returns_no_update(tmp_path):
    updates = app._suggest_ft_steps_from_data(str(tmp_path / "missing.jsonl"), 4, 4)
    assert all(u == gr.update() for u in updates)


def test_small_dataset_uses_fewer_epochs(tmp_path):
    path = _write_jsonl(tmp_path, 10)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 1, 1)
    # 10 examples, 2-epoch tier, effective batch 1 -> 20 steps.
    assert steps_update["value"] == 20


def test_large_dataset_uses_more_epochs(tmp_path):
    path = _write_jsonl(tmp_path, 1000)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 4, 4)
    # 1000 examples, 6-epoch tier (>500), effective batch 16 -> 375 steps.
    assert steps_update["value"] == 375


def test_step_floor_is_respected(tmp_path):
    path = _write_jsonl(tmp_path, 2)
    steps_update, *_ = app._suggest_ft_steps_from_data(path, 64, 4)
    assert steps_update["value"] >= 10
