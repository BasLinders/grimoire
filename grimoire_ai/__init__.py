"""Grimoire — a hybrid SLM engine for CPU-efficient, corpus-grounded agents.

Grimoire pairs Granville's Radial Basis Function (RBF) interpolator with a
small quantized LLM to produce an agent engine that is accurate, explainable,
and runnable on a home machine without a GPU.

Architecture (by phase):
    Phase 1 — corpus:  Ingest text, stem, index multi-tokens.
    Phase 2 — rbf:     Granville's RBF kernel for retrieval scoring.
    Phase 3 — llm:     Thin wrapper around a local quantized LLM (ollama).
    Phase 4 — router:  Intent detection; routes to RBF or tool integrations.
    Phase 5 — adapters: Load corpus from files, URLs, and structured sources.

Quick start:
    >>> from grimoire import GrimoireCorpus
    >>> corpus = GrimoireCorpus()
    >>> corpus.add_text("A grappled creature has its speed reduced to zero.")
    >>> results = corpus.query("grapple speed")
"""

from grimoire_ai.corpus import GrimoireCorpus

__all__ = ["GrimoireCorpus"]
__version__ = "0.1.0"
