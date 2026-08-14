# Architecture Optimization — Candidate Improvements

Candidate architectural changes for Grimoire's hybrid neural LM + retrieval
system, gathered from recent research and prioritized by effort-to-payoff.
Originally written as "none of these are started" — status per item is now
tracked below since that's no longer true.

## Status

| # | Item | Status |
|---|---|---|
| 1 | Multi-Head Latent Attention | ✓ shipped — [PR #176](https://github.com/BasLinders/grimoire/pull/176) (module), [#177](https://github.com/BasLinders/grimoire/pull/177) (wiring), [#178](https://github.com/BasLinders/grimoire/pull/178) (UI) |
| 2 | Multi-Token Prediction | ✓ shipped — [PR #180](https://github.com/BasLinders/grimoire/pull/180) |
| 3 | RETRO-style chunked cross-attention | in progress — see below |
| 4 | Contrastive retrieval fine-tuning | ✓ already existed — `grimoire_ai/llm/training/embed_tune.py` / `scripts/embed_tune.py`, predates this document (this list was written without checking for it first) |
| 5 | Grammar-constrained decoding | ✓ shipped — [PR #179](https://github.com/BasLinders/grimoire/pull/179) |

## 1. Multi-Head Latent Attention (MLA)

Replace/extend the current GQA attention block with MLA: compress K/V into a
low-rank latent vector instead of caching full per-head K/V. A 2025 paper
studying this at small-model scale (comparable to Grimoire's 25M–250M
presets) reports a 45% KV-cache memory reduction with only a 0.3%
validation-loss increase, and a 1.4x inference speedup at half-rank latent
dimension. This matters more here than for a typical LLM because retrieved
corpus excerpts inflate prompt length every turn — MLA shrinks the
memory/speed cost of that context specifically. Would slot into
`grimoire_ai/llm/model/` alongside the existing GQA/RoPE attention module.

Source: [Latent Multi-Head Attention for Small Language Models](https://arxiv.org/abs/2506.09342)

## 2. Multi-Token Prediction (MTP)

Training-objective change: predict the next *k* tokens per step instead of
only the next token. Improves sample efficiency during pretraining (useful
given the corpus-scarcity failure modes noted in `expansion_PLAN.md`), and
gives a natural path to self-speculative decoding — inference speedup
without building/maintaining a separate draft model, directly de-risking
Phase 2.5 item #3 in `PLAN.md` (speculative decoding, currently "not
started").

Source: [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/pdf/2405.04434)

## 3. RETRO-style chunked cross-attention

Replace prompt-concatenation retrieval with a dedicated cross-attention path
inside the transformer that attends to retrieved chunks directly, instead of
stuffing excerpts into the prompt via `PromptBuilder`. This avoids retrieved
context eating the (small) context window or getting diluted by attention
across irrelevant tokens. Retro-li specifically validates this approach at
small-model + small-database scale — closer to Grimoire's corpus than the
original 7.5B/trillion-token RETRO — and notes it demands more accurate
neighbor retrieval when the database is sparse, a real consideration given
the semantic index's current staleness (see
[[project_embedding_retrieval_priority]] memory). Bigger architectural lift
than 1–2, but the one most likely to fix retrieval-grounding quality rather
than just speed.

Source: [Retro-li: Small-Scale Retrieval Augmented Generation](https://arxiv.org/html/2410.00004v2)

## 4. Contrastive fine-tuning of retrieval embeddings — already implemented

`SemanticRetriever` reuses `GrimoireTransformer.embed()` — embeddings
trained for the generation objective, doing double duty for retrieval.
Adding a small contrastive auxiliary loss (positive/negative passage pairs)
tunes those embeddings specifically for retrieval quality without touching
the main architecture.

This exists already: `grimoire_ai/llm/training/embed_tune.py` implements
in-batch InfoNCE (SimCSE self-pairs, `DocumentGroupedBatchSampler` for hard
same-document negatives, and a supervised `train_step_pairs` path over real
(question, answer) pairs), trains a LoRA adapter rather than the full model
(so it never risks the generation-quality regression a full-model contrastive
fine-tune could cause), and ships with `scripts/embed_tune.py` and 49 passing
tests. This document was written without checking for it first.

Source: [Improving Text Embeddings for Smaller Language Models Using Contrastive Fine-tuning](https://arxiv.org/abs/2408.00690)

## 5. Grammar-constrained decoding for stat-block output

`training_PLAN.md` flags two recurring failure modes: invented CR/XP values
and degenerate repetition loops (`does does does...`). Constraining
generation to a grammar/schema when a query asks for a monster stat block or
numeric fact eliminates both classes structurally — the decoder cannot emit
an ill-formed or looping continuation. 2025 results show grammar constraints
can substitute for in-context examples on resource-constrained models. This
is decode-time only, no retraining needed, and could plug in next to the
existing `<TOOL:python>` tag-resolution logic in `MathTool`.

Source: [Grammar-Constrained Decoding Makes Large Language Models Better Logical Parsers](https://aclanthology.org/2025.acl-industry.34/)

---

## Recommendation

If picking one item to start with: **#1 (MLA)** has the best effort/impact
ratio — a contained change to one module, published numbers at Grimoire's
exact model scale, and it compounds with everything else since it makes
every downstream retrieval-heavy prompt cheaper. **#5 (grammar-constrained
decoding)** is the fastest fix for the specific accuracy bug already being
chased in `training_PLAN.md`.

## Further reading

- [A Systematic Review of Key RAG Systems: Progress, Gaps, and Future Directions](https://arxiv.org/html/2507.18910v1)
