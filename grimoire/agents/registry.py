"""Agent registry: load and query named agent configurations.

An agent configuration lives in a JSON file (default: ``agents.json`` in the
project root).  Each entry describes everything needed to instantiate an
``InferenceEngine`` for that agent:

    {
        "saga": {
            "display_name": "Saga",
            "description": "D&D, mathematics, and data-science assistant.",
            "checkpoint":   "checkpoints/saga/latest.pt",
            "vocab":        "data/tokenizer/bpe.json",
            "corpus_dirs":  ["data/corpus/saga/"],
            "gen_config": {
                "max_new_tokens": 256,
                "temperature":    0.8,
                "top_k":          50,
                "top_p":          0.9
            }
        }
    }

All paths are resolved relative to the directory that contains the JSON file.
``corpus_dirs`` is optional; omit it (or set it to ``[]``) for a model-only
agent with no retrieval corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for a single named agent."""

    key: str
    display_name: str
    description: str
    checkpoint: str
    vocab: str
    corpus_dirs: list[str] = field(default_factory=list)
    gen_config: dict = field(default_factory=dict)


class AgentRegistry:
    """Loads and exposes named agent configurations from a JSON file.

    Args:
        path: Path to the agents JSON file.  Defaults to ``agents.json``
              in the current working directory.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the JSON is malformed or a required field is missing.
    """

    _REQUIRED = ("display_name", "checkpoint", "vocab")

    def __init__(self, path: str | Path = "agents.json") -> None:
        self._path = Path(path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Agent registry not found: {self._path}")
        self._base = self._path.parent
        self._agents: dict[str, AgentConfig] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def keys(self) -> list[str]:
        """Return the list of agent keys in registry order."""
        return list(self._agents.keys())

    def display_names(self) -> list[str]:
        """Return display names in registry order."""
        return [a.display_name for a in self._agents.values()]

    def get(self, key: str) -> AgentConfig:
        """Return the ``AgentConfig`` for *key*.

        Raises:
            KeyError: If *key* is not in the registry.
        """
        if key not in self._agents:
            raise KeyError(f"Agent '{key}' not found in registry.")
        return self._agents[key]

    def get_by_display_name(self, display_name: str) -> AgentConfig:
        """Return the ``AgentConfig`` whose display_name matches *display_name*.

        Raises:
            KeyError: If no agent has that display name.
        """
        for agent in self._agents.values():
            if agent.display_name == display_name:
                return agent
        raise KeyError(f"No agent with display name '{display_name}'.")

    def build_engine(self, key: str):
        """Instantiate and return an ``InferenceEngine`` for *key*.

        Loads the corpus from all ``corpus_dirs`` if present.  Paths are
        resolved relative to the registry file's directory.

        Returns:
            ``InferenceEngine`` ready for inference.
        """
        from grimoire.corpus.corpus import GrimoireCorpus
        from grimoire.llm.inference.engine import InferenceEngine
        from grimoire.llm.inference.sampler import GenerationConfig

        cfg = self.get(key)

        corpus: Optional[GrimoireCorpus] = None
        if cfg.corpus_dirs:
            corpus = GrimoireCorpus()
            for corpus_dir in cfg.corpus_dirs:
                p = self._resolve(corpus_dir)
                if not p.exists():
                    continue
                for txt_file in sorted(p.glob("*.txt")):
                    corpus.add_text(
                        txt_file.read_text(encoding="utf-8", errors="replace"),
                        source=txt_file.stem,
                    )

        gen_config = GenerationConfig(**cfg.gen_config) if cfg.gen_config else None

        return InferenceEngine(
            checkpoint_path=str(self._resolve(cfg.checkpoint)),
            tokenizer_path=str(self._resolve(cfg.vocab)),
            corpus=corpus,
            gen_config=gen_config,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, rel: str) -> Path:
        return (self._base / rel).resolve()

    def _load(self) -> None:
        try:
            raw: dict = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {self._path}: {exc}") from exc

        for key, entry in raw.items():
            for req in self._REQUIRED:
                if req not in entry:
                    raise ValueError(
                        f"Agent '{key}' is missing required field '{req}'."
                    )
            self._agents[key] = AgentConfig(
                key=key,
                display_name=entry["display_name"],
                description=entry.get("description", ""),
                checkpoint=entry["checkpoint"],
                vocab=entry["vocab"],
                corpus_dirs=entry.get("corpus_dirs", []),
                gen_config=entry.get("gen_config", {}),
            )
