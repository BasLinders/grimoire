# Architecture Optimization — Candidate Improvements

Candidate architectural changes for Grimoire's hybrid neural LM + retrieval
system, gathered from recent research and prioritized by effort-to-payoff.
Originally written as "none of these are started" — status per item is now
tracked below since that's no longer true.

## Status

| # | Item | Status |
|---|---|---|
| 1 | Multi-Head Latent Attention | ✓ shipped — [PR #176](https://github.com/BasLinders/grimoire/pull/176) (module), [#177](https://github.com/BasLinders/grimoire/pull/177) (wiring), [#178](https://github.com/BasLinders/grimoire/pull/178) (UI), [#182](https://github.com/BasLinders/grimoire/pull/182) (LoRA support) |
| 2 | Multi-Token Prediction | ✓ shipped — [PR #180](https://github.com/BasLinders/grimoire/pull/180) |
| 3 | RETRO-style chunked cross-attention | ✓ shipped — [PR #181](https://github.com/BasLinders/grimoire/pull/181) (module + wiring + training pipeline) |
| 4 | Contrastive retrieval fine-tuning | ✓ already existed — `grimoire_ai/llm/training/embed_tune.py` / `scripts/embed_tune.py`, predates this document (this list was written without checking for it first) |
| 5 | Grammar-constrained decoding | ✓ shipped — [PR #179](https://github.com/BasLinders/grimoire/pull/179) |
| 6 | Training-recipe stability & optimizer improvements | not started |
| 7 | Cross-encoder reranking | not started |
| 8 | Adaptive/corrective retrieval (CRAG / Self-RAG) | not started |
| 9 | Data-centric: curation & synthetic augmentation | not started |

## Known limitations in shipped items

Carried over from each item's own PR rather than fixed there — listed here
so they don't get lost:

- **MLA + GGUF export** (#1): still `NotImplementedError`. Correct support
  means a distinct GGUF architecture branch matching llama.cpp's actual
  `deepseek2` tensor/metadata conventions, which couldn't be verified
  without a real llama.cpp binary to test an export against.
- **MTP heads are parallel, not sequential** (#2): a deliberately lighter
  variant of DeepSeek-V3's actual depth-chained MTP design, unvalidated
  against it. `evaluate()` also excludes the MTP loss on purpose (keeps
  `val_loss` comparable across MTP-enabled and MTP-disabled runs).
- **RETRO's two scope cuts** (#3): whole-window attention instead of
  per-chunk-windowed retrieval, and reusing the token embedding table
  instead of a dedicated neighbor encoder. Both documented as first-pass
  simplifications, not settled design. `evaluate()` excluding RETRO's
  neighbor context is a *real* gap here (unlike MTP's deliberate exclusion)
  — CCA output genuinely changes predictions, so `val_loss` understates
  what a RETRO-enabled model actually does.

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

## 6. Training-recipe stability & optimizer improvements

A bundle of training-recipe changes rather than one architectural rewrite —
each is a small, independent knob, so they can land incrementally instead
of as one big PR. A 2026 recipe (IMU-1, 430M params) combining these
reports matching the benchmark performance of models trained on 56× more
data:

- **QK-norm, per-head gating, value residual connections, LayerNorm
  scaling** — training-stability tricks applied inside the attention/FFN
  blocks. Directly applicable to both `GroupedQueryAttention` and
  `MultiHeadLatentAttention`.
- **Muon/NorMuon optimizer** (orthogonalized momentum) as an AdamW
  alternative — reported efficiency gains over Adam-family optimizers at
  comparable scale.
- **muP (maximal update parametrization)** — tune hyperparameters once on
  `small-25M`, transfer to `medium-85M`/`large-250M` without re-tuning.
  Directly relevant since Grimoire already maintains three size presets.
- **WSD (warmup-stable-decay) LR schedule** instead of the current cosine
  decay — decouples a run's final length from `total_steps` chosen
  upfront, which fits the Chinchilla-suggested-steps workflow already in
  the Pre-train tab better than a schedule that must know the total step
  count in advance.
- **Post-hoc EMA of checkpoints** — cheap accuracy bump, no training
  changes needed, just an averaging pass over saved checkpoints.

Sources: [IMU-1: Sample-Efficient Pre-training of Small Language Models](https://arxiv.org/abs/2602.02522), [Practical Efficiency of Muon for Pretraining](https://arxiv.org/pdf/2505.02222)

## 7. Cross-encoder reranking

`SemanticRetriever` currently does plain cosine top-k with no reranking
step — the single result set `PromptBuilder` gets is whatever the
embedding similarity search returned, unfiltered. Adding a lightweight
cross-encoder reranker after retrieval, before context injection, is
described consistently across 2025–2026 sources as the single biggest
precision gain available to any RAG pipeline (10–25% over retrieval alone),
and measurably reduces hallucinations. Two small options fit this project's
scale: `ms-marco-MiniLM-L-12-v2` (33M params) or `ms-marco-TinyBERT-L-2-v2`
(14M params, 2–5ms/pair on CPU) — both already installable via the same
`sentence-transformers` optional dependency `EXTERNAL_ENCODERS` uses.

Sources: [Rethinking the Reranker: Boundary-Aware Evidence Selection for RAG](https://arxiv.org/pdf/2602.03689), [Shallow Cross-Encoders for Low-Latency Retrieval](https://arxiv.org/pdf/2403.20222)

## 8. Adaptive/corrective retrieval (CRAG / Self-RAG)

Two related but different-sized ideas for deciding *when* retrieval is
trustworthy, both complementary to (not a replacement for) the existing
`retrieval_threshold` router in `InferenceEngine._retrieve()`:

- **Corrective RAG (CRAG)** — a lightweight retrieval evaluator scores
  retrieved passages and lets the system adaptively drop, demote, or
  fall back when retrieval quality is poor. Smaller lift; layers on top of
  the existing threshold-router mechanism rather than replacing it.
- **Self-RAG** — the model itself learns (via special reflection tokens)
  when to retrieve and critiques its own retrieved context. Bigger lift —
  needs dedicated fine-tune data with reflection-token annotations, not
  just a decode-time or retrieval-time addition.

Source: [Classifying and Addressing the Diversity of Errors in RAG Systems](https://arxiv.org/pdf/2510.13975)

## 9. Data-centric: curation & synthetic augmentation

`expansion_PLAN.md` already concluded Grimoire's failure modes "look like
data-scarcity symptoms, not architecture-too-small symptoms" — which makes
this the item most likely to matter most, and the one most different in
kind from everything else on this list (a corpus/data-pipeline change, not
a model change):

- **Curation over volume** — a February 2026 result (DatologyAI) found
  targeted per-corpus curation matching baselines at 4–10× lower compute
  versus adding more raw tokens at the same quality bar.
- **Synthetic augmentation via rephrasing** — mixing rephrased/rewritten
  corpus text at roughly a 1/3 ratio with natural text gave a 5–10×
  speedup in one 2025 study; pure synthetic data risks model collapse, so
  this is a mixing ratio to tune carefully, not a free multiplier on
  corpus size.
- **EntiGraph-style entity recombination** — synthesize new training text
  by connecting entities extracted from the existing domain corpus.
  Plausible fit for expanding the Saga/D&D corpus specifically, and
  compatible with the existing derived-adventure pilot's practice of
  verifying facts against SRD/Open5e rather than inventing them.

Sources: [BeyondWeb: Lessons from Scaling Synthetic Data for Trillion-scale Pretraining](https://arxiv.org/pdf/2508.10975), [Demystifying Synthetic Data in LLM Pre-training](https://aclanthology.org/2025.emnlp-main.544.pdf)

---

## Recommendation

Of the newer items, **#7 (cross-encoder reranking)** has the best
effort/impact ratio in this document now — a bolt-on to `SemanticRetriever`
that touches no model code, with the most consistent "biggest single win"
framing across sources. **#9 (data-centric work)** is the highest-value
item *if* Grimoire's own prior conclusion (data, not architecture, is the
ceiling) is trusted — but it's a bigger, fuzzier undertaking than #7, and a
corpus/data-pipeline effort rather than a code change.

Of the original five: **#1 (MLA)** had the best effort/impact ratio when
this list was first written — a contained change to one module, published
numbers at Grimoire's exact model scale, and it compounds with everything
else since it makes every downstream retrieval-heavy prompt cheaper. **#5
(grammar-constrained decoding)** was the fastest fix for the specific
accuracy bug already being chased in `training_PLAN.md`. Both have since
shipped.

## Further reading

- [A Systematic Review of Key RAG Systems: Progress, Gaps, and Future Directions](https://arxiv.org/html/2507.18910v1)
