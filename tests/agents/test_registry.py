"""Unit tests for AgentRegistry."""

from __future__ import annotations

import json
import pytest

from grimoire_ai.agents.registry import AgentConfig, AgentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_registry(tmp_path, data: dict) -> str:
    p = tmp_path / "agents.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


MINIMAL = {
    "saga": {
        "display_name": "Saga",
        "description": "D&D assistant.",
        "checkpoint": "ckpt/saga.pt",
        "vocab": "data/bpe.json",
        "corpus_dirs": [],
        "gen_config": {"max_new_tokens": 128, "temperature": 0.8},
    }
}

TWO_AGENTS = {
    **MINIMAL,
    "oracle": {
        "display_name": "Oracle",
        "description": "General assistant.",
        "checkpoint": "ckpt/oracle.pt",
        "vocab": "data/bpe.json",
    },
}


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def test_registry_loads_valid_file(tmp_path):
    path = _write_registry(tmp_path, MINIMAL)
    reg = AgentRegistry(path)
    assert "saga" in reg.keys()


def test_registry_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        AgentRegistry(str(tmp_path / "no_such_file.json"))


def test_registry_raises_for_invalid_json(tmp_path):
    bad = tmp_path / "agents.json"
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        AgentRegistry(str(bad))


def test_registry_raises_for_missing_required_field(tmp_path):
    data = {"broken": {"display_name": "Broken"}}  # missing checkpoint and vocab
    path = _write_registry(tmp_path, data)
    with pytest.raises(ValueError, match="missing required field"):
        AgentRegistry(path)


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def test_keys_returns_all_agent_keys(tmp_path):
    path = _write_registry(tmp_path, TWO_AGENTS)
    reg = AgentRegistry(path)
    assert set(reg.keys()) == {"saga", "oracle"}


def test_display_names_order_matches_json(tmp_path):
    path = _write_registry(tmp_path, TWO_AGENTS)
    reg = AgentRegistry(path)
    assert reg.display_names() == ["Saga", "Oracle"]


def test_get_returns_correct_config(tmp_path):
    path = _write_registry(tmp_path, MINIMAL)
    reg = AgentRegistry(path)
    cfg = reg.get("saga")
    assert isinstance(cfg, AgentConfig)
    assert cfg.key == "saga"
    assert cfg.display_name == "Saga"
    assert cfg.description == "D&D assistant."


def test_get_raises_for_unknown_key(tmp_path):
    path = _write_registry(tmp_path, MINIMAL)
    reg = AgentRegistry(path)
    with pytest.raises(KeyError):
        reg.get("nonexistent")


def test_get_by_display_name(tmp_path):
    path = _write_registry(tmp_path, TWO_AGENTS)
    reg = AgentRegistry(path)
    cfg = reg.get_by_display_name("Oracle")
    assert cfg.key == "oracle"


def test_get_by_display_name_raises_for_unknown(tmp_path):
    path = _write_registry(tmp_path, MINIMAL)
    reg = AgentRegistry(path)
    with pytest.raises(KeyError):
        reg.get_by_display_name("Nonexistent")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_corpus_dirs_defaults_to_empty(tmp_path):
    data = {
        "minimal": {
            "display_name": "Minimal",
            "checkpoint": "ckpt/m.pt",
            "vocab": "data/bpe.json",
        }
    }
    path = _write_registry(tmp_path, data)
    cfg = AgentRegistry(path).get("minimal")
    assert cfg.corpus_dirs == []


def test_gen_config_defaults_to_empty(tmp_path):
    data = {
        "minimal": {
            "display_name": "Minimal",
            "checkpoint": "ckpt/m.pt",
            "vocab": "data/bpe.json",
        }
    }
    path = _write_registry(tmp_path, data)
    cfg = AgentRegistry(path).get("minimal")
    assert cfg.gen_config == {}


def test_description_defaults_to_empty_string(tmp_path):
    data = {
        "nodesc": {
            "display_name": "NoDesc",
            "checkpoint": "ckpt/n.pt",
            "vocab": "data/bpe.json",
        }
    }
    path = _write_registry(tmp_path, data)
    cfg = AgentRegistry(path).get("nodesc")
    assert cfg.description == ""


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_paths_are_resolved_relative_to_json_file(tmp_path):
    path = _write_registry(tmp_path, MINIMAL)
    reg = AgentRegistry(path)
    cfg = reg.get("saga")
    # _resolve is internal but we can verify checkpoint path is absolute
    resolved = reg._resolve(cfg.checkpoint)
    assert resolved.is_absolute()
    assert str(tmp_path) in str(resolved)
