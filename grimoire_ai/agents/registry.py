"""Agent registry: load and query named agent configurations.

An agent configuration lives in a JSON file (default: ``agents.json`` in the
project root).  Each entry describes everything needed to instantiate an
``InferenceEngine`` for that agent:

    {
        "general": {
            "display_name": "General",
            "checkpoint":   "checkpoints/base_chat.pt",
            "vocab":        "data/tokenizer/bpe.json"
        },
        "saga": {
            "display_name": "Saga",
            "description":  "D&D, mathematics, and data-science assistant.",
            "checkpoint":   "checkpoints/base_chat.pt",
            "vocab":        "data/tokenizer/bpe.json",
            "lora_path":    "checkpoints/lora/saga.lora",
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
``corpus_dirs`` and ``lora_path`` are optional.  When ``lora_path`` points to
a ``.lora`` file the adapter is applied automatically at engine-load time.
For the multi-agent router all agents should share the same ``checkpoint``
so LoRA adapters can be swapped without reloading the base model.
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
    lora_path: str = ""


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

    def build_engine(self, key: str, quantize: bool = False):
        """Instantiate and return an ``InferenceEngine`` for *key*.

        Loads the corpus from all ``corpus_dirs`` if present.  Paths are
        resolved relative to the registry file's directory.  The lexical
        n-gram index is cached to ``{corpus_dir}/.cache/lexical.pkl`` and
        reused on subsequent loads as long as the source ``.txt`` files have
        not changed.

        Args:
            key: Agent key in the registry.
            quantize: Apply dynamic int8 quantization after loading.

        Returns:
            ``InferenceEngine`` ready for inference.
        """
        from grimoire_ai.llm.inference.engine import InferenceEngine
        from grimoire_ai.llm.inference.sampler import GenerationConfig

        cfg = self.get(key)

        ckpt_path = self._resolve(cfg.checkpoint)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        corpus = self._scan_corpus_dirs(cfg.corpus_dirs) if cfg.corpus_dirs else None

        gen_config = GenerationConfig(**cfg.gen_config) if cfg.gen_config else None

        engine = InferenceEngine(
            checkpoint_path=str(ckpt_path),
            tokenizer_path=str(self._resolve(cfg.vocab)),
            corpus=corpus,
            gen_config=gen_config,
            quantize=quantize,
        )

        if cfg.lora_path:
            import warnings
            lp = self._resolve(cfg.lora_path)
            if lp.is_file():
                engine.load_lora(str(lp))
            else:
                warnings.warn(
                    f"LoRA file not found for agent '{key}': {lp}. Using base weights.",
                    stacklevel=2,
                )

        return engine

    @property
    def default_key(self) -> str:
        """The first agent key in the registry, used as the routing fallback."""
        return next(iter(self._agents))

    def build_router(self, threshold: float = 0.05) -> "AgentRouter":
        """Build an ``AgentRouter`` using each agent's lexical corpus.

        Only agents with ``corpus_dirs`` are scoreable.  Agents without a
        corpus can only win as the ``default_key`` fallback.

        Args:
            threshold: Minimum Jaccard score for a routing decision to be
                accepted.  Queries below this score are routed to the default.

        Returns:
            ``AgentRouter`` ready for ``route(query)`` calls.
        """
        from grimoire_ai.agents.router import AgentRouter

        corpora: dict[str, "GrimoireCorpus"] = {}
        for key, cfg in self._agents.items():
            if not cfg.corpus_dirs:
                continue
            corpus = self._scan_corpus_dirs(cfg.corpus_dirs)
            if corpus is not None:
                corpora[key] = corpus

        return AgentRouter(
            corpora=corpora,
            default_key=self.default_key,
            threshold=threshold,
        )

    def build_multi_agent_engine(
        self,
        threshold: float = 0.05,
        quantize: bool = False,
    ) -> "MultiAgentEngine":
        """Build a ``MultiAgentEngine`` that routes each query to the best agent.

        The engine shares a single model checkpoint (the default agent's) and
        swaps LoRA adapters and corpora between turns as needed.  Agents whose
        ``lora_path`` file is absent are treated as using base weights only.

        Args:
            threshold: Routing confidence threshold passed to ``AgentRouter``.
            quantize: Apply int8 quantization to the base model.

        Returns:
            ``MultiAgentEngine`` ready for ``chat_stream()`` calls.
        """
        from grimoire_ai.agents.router import MultiAgentEngine

        router = self.build_router(threshold=threshold)

        # Build the shared base engine from the default agent's checkpoint.
        engine = self.build_engine(self.default_key, quantize=quantize)
        # build_engine may have applied the default agent's LoRA; unload it
        # so the engine starts from base weights and _switch_to handles the rest.
        engine.unload_lora()

        # Collect resolved lora_paths and corpora for all agents.
        import warnings
        lora_paths: dict[str, str] = {}
        corpora: dict[str, object] = {}
        for key, cfg in self._agents.items():
            if cfg.lora_path:
                lp = self._resolve(cfg.lora_path)
                if lp.is_file():
                    lora_paths[key] = str(lp)
                else:
                    warnings.warn(
                        f"LoRA file not found for agent '{key}': {lp}. "
                        "Agent will use base weights.",
                        stacklevel=2,
                    )
            corpora[key] = router._corpora.get(key)

        return MultiAgentEngine(
            engine=engine,
            router=router,
            lora_paths=lora_paths,
            corpora=corpora,
            default_key=self.default_key,
        )

    def _scan_corpus_dirs(self, corpus_dirs: list[str]) -> "Optional[GrimoireCorpus]":
        """Collect .txt files from *corpus_dirs* and return a cached GrimoireCorpus.

        Returns ``None`` when no ``.txt`` files are found in any of the listed
        directories so callers can distinguish "no corpus" from an empty one.
        """
        txt_files: list[Path] = []
        for corpus_dir in corpus_dirs:
            p = self._resolve(corpus_dir)
            if p.exists():
                txt_files.extend(sorted(p.glob("*.txt")))
        if not txt_files:
            return None
        cache_path = txt_files[0].parent / ".cache" / "lexical.pkl"
        return self._build_corpus_cached(txt_files, cache_path)

    @staticmethod
    def _build_corpus_cached(txt_files: list[Path], cache_path: Path) -> "GrimoireCorpus":
        """Return a GrimoireCorpus from cache if fresh; otherwise build and cache it."""
        import pickle
        from grimoire_ai.corpus.corpus import GrimoireCorpus

        newest_src = max(f.stat().st_mtime for f in txt_files)
        if cache_path.is_file() and cache_path.stat().st_mtime >= newest_src:
            try:
                with cache_path.open("rb") as fh:
                    return pickle.load(fh)
            except Exception:
                pass  # corrupt or stale pickle → rebuild below

        corpus = GrimoireCorpus()
        for txt_file in txt_files:
            corpus.add_text(
                txt_file.read_text(encoding="utf-8", errors="replace"),
                source=txt_file.stem,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache_path.open("wb") as fh:
                pickle.dump(corpus, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass  # non-fatal if write fails
        return corpus

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
                lora_path=entry.get("lora_path", ""),
            )
