"""Evaluation harness for GrimoireTransformer.

Three evaluators and a top-level harness that runs them together:

    perplexity  — bits-per-character on a held-out corpus slice
    retrieval   — hit-rate of top-1 RAG results over a fixed query set
    quiz        — keyword-recall on a factual Q&A set (D&D / math)
    harness     — orchestrates all three; writes JSON results to data/eval/
"""
