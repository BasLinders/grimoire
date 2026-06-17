"""Grimoire — a hybrid SLM engine for CPU-efficient, corpus-grounded agents.

Grimoire pairs a semantic retrieval engine with a scratch-built decoder-only
transformer LLM: the retrieval engine grounds answers in a domain corpus,
the LLM provides coherent conversational language, and a retrieval-score
router decides per-query whether grounding is needed. Both retrieval and
generation run on the same model — there is no separate embedding server.

Package layout:
    corpus:   Ingest text, stem, index multi-tokens; Jaccard lexical
              retrieval (``GrimoireCorpus``) — the dependency-free fallback
              used at ingest-time previews and before a model is trained.
    llm:      The transformer itself (tokenizer, model, training,
              inference) plus semantic retrieval
              (``llm.inference.semantic.SemanticRetriever``), which embeds
              passages with the trained model's own representations and is
              the primary retrieval path once a checkpoint exists.
    agents:   ``AgentRegistry`` / ``AgentRouter`` — named corpus+checkpoint
              configurations and automatic multi-agent dispatch.
    state:    ``ConversationState`` — rolling multi-turn history.
    tools:    ``MathTool`` — safe arithmetic evaluation injected as context.
    cli / ui: Terminal chat loop and the Gradio training/inference app.

See README.md for the full architecture diagram and usage examples.

Quick start:
    >>> from grimoire_ai import GrimoireCorpus
    >>> corpus = GrimoireCorpus()
    >>> corpus.add_text("A grappled creature has its speed reduced to zero.")
    >>> results = corpus.query("grapple speed")
"""

from grimoire_ai.corpus import GrimoireCorpus

__all__ = ["GrimoireCorpus"]
__version__ = "0.1.0"
