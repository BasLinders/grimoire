"""Grimoire inference pipeline.

Wires the BPE tokenizer, GrimoireTransformer, and GrimoireCorpus into a
single end-to-end generation flow:

    corpus query → PromptBuilder → generate() → BPE decode → plain text

Public API
----------
    from grimoire.llm.inference.prompt import PromptBuilder
    from grimoire.llm.inference.sampler import GenerationConfig, generate
    from grimoire.llm.inference.engine import InferenceEngine
"""
