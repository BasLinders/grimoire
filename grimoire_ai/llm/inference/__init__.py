"""Grimoire inference pipeline.

Wires the BPE tokenizer, GrimoireTransformer, and GrimoireCorpus into a
single end-to-end generation flow:

    corpus query → PromptBuilder → generate() → BPE decode → plain text

Public API
----------
    from grimoire_ai.llm.inference.prompt import PromptBuilder
    from grimoire_ai.llm.inference.sampler import GenerationConfig, generate
    from grimoire_ai.llm.inference.engine import InferenceEngine
"""
