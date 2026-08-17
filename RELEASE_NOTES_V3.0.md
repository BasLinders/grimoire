# Grimoire v3.0

*Released August 2026 — 239 commits since v2.0*

V3.0 is the biggest release in the project's history: three new model-architecture options, a corrective retrieval layer, cross-encoder reranking, grammar-constrained decoding, a reworked evaluation scorer, and a corpus toolkit that now filters and synthesizes data instead of just collecting it. The training/eval UI and the chat UI have also been split into two separate apps. Every addition is opt-in and every existing checkpoint keeps working unchanged — this is a major release because of the scope and the new capabilities, not because anything breaks.

---

## Model architecture: three new opt-in options

All three default to the pre-existing behaviour, so nothing changes for existing checkpoints unless you explicitly turn them on for a new training run.

**Multi-Head Latent Attention (MLA)** — an alternative to GQA that compresses K/V into a shared low-rank latent with decoupled RoPE, using a matrix-absorption trick so cached decoding never re-expands full per-head K/V. Reported ~45% KV-cache reduction at ~0.3% validation-loss cost, which matters more here than in a generic LLM because retrieved corpus excerpts inflate prompt length on every turn. Selected via `TransformerConfig.attention_type: "gqa" | "mla"` and a new dropdown + dimension fields in the Pre-train tab. LoRA adapters now target MLA's projection matrices (`w_dkv`/`w_uk`/`w_uv`/`w_qc`/`w_qr`/`w_kr`) as well as GQA's. Default remains `"gqa"`.

**Multi-Token Prediction (MTP)** — optional auxiliary prediction heads (`TransformerConfig.n_predict`, default `0`) used only during pretraining to improve sample efficiency. A deliberately lighter "parallel heads" design rather than DeepSeek-V3's depth-chained variant. Evaluation loss intentionally excludes the MTP term so `val_loss` stays comparable across MTP-on/off runs.

**RETRO-style Chunked Cross-Attention (CCA)** — the model can now attend directly to retrieved neighbor chunks inside the transformer, not just via prompt concatenation. Enabled per-block via `TransformerConfig.retro_layers` (`None` by default). Training-side wiring pairs precomputed neighbor ids with training windows (`NeighborAugmentedDataset`), completing the RETRO pipeline end to end. Two scope cuts vs. the original RETRO paper: it reuses the token embedding table instead of a dedicated neighbor encoder, and does whole-window rather than per-chunk-windowed attention.

**Apple Silicon (MPS) support** — `select_device()` now returns MPS as a third option alongside CUDA and CPU. `torch.compile`, `cudnn.benchmark`, and AMP remain CUDA-only, so MPS training runs on GPU without those extra speedups.

---

## Retrieval quality: reranking and corrective retrieval

**Cross-encoder reranking** — a second retrieval stage that rescores a widened first-stage candidate pool with a cross-encoder (`ms-marco-TinyBERT-L-2-v2` or `ms-marco-MiniLM-L-12-v2` via `sentence-transformers`) before taking the final top-k. Opt-in, off by default; selectable from a dropdown in the Chat UI.

**CRAG (Corrective RAG)** — a per-passage confidence filter layered on top of the existing top-1 retrieval-score gate, not replacing it. Passages are classified Correct / Ambiguous / Incorrect against two thresholds (matching the original CRAG paper's design rather than an earlier single-cutoff approximation); Ambiguous passages are demoted rather than dropped, since there's no web-search fallback to combine them with in this project. When no passage clears the confidence bar, an optional `corrective_retry` issues one widened local re-query — the project's local-only substitute for CRAG's web-search fallback. Runs after reranking when both are enabled. Now wired into the agent registry and enabled for Saga.

**Persistent RAG index** — semantic embeddings are pre-computed once and cached as a numpy memmap (with an optional FAISS `IndexFlatIP` for larger corpora), with MD5-hash staleness checks against source files and checkpoint. Replaces the previous per-session recomputation. A "Build/Rebuild index" button lives in the new Corpus tab.

**Grammar-constrained decoding** — two opt-in logit-masking guards: a stat-block constraint that restricts tokens after a recognized field label (e.g. "Challenge Rating:") to well-formed continuations, and a repetition-loop guard that hard-bans the token that would extend an already-established exact repeat cycle. Neither changes generation when not supplied. Known limitation: the loop guard only catches exact token-sequence repeats, not templated loops with varying content (e.g. alternating numeric values in an otherwise-repeated sentence) — documented, not yet fixed.

---

## Evaluation

The perplexity / retrieval-hit-rate / D&D-quiz harness from Phase 2 is unchanged in shape, but the quiz scorer was substantially reworked: token-F1 now searches all contiguous windows of the response against the reference answer instead of scoring the whole free-form generation against it, and excludes vocabulary shared with the question itself. This closes a scoring gap where verbose-but-correct answers were penalized relative to terse ones. `scripts/evaluate.py` also gained `--corpus-limit` (fixed-seed sampling, a mitigation for the corpus index's unbounded memory use — see Known Limitations) and quiz-specific loop-guard/repetition-penalty flags.

---

## Tool calling

The arithmetic tool (`MathTool`) — query-side detection, `ast`-based safe evaluation, response-side `<TOOL:python>` tag resolution — is unchanged in shape but now supports scipy statistics functions (`norm_cdf`, `binom_pmf`, `t_ppf`, and more) alongside the original arithmetic set, with a graceful fallback when scipy isn't installed.

---

## LoRA / fine-tuning

LoRA adapters, hot-swappable per-agent without reloading the base model, are unchanged in shape but now also target MLA's attention projections when a checkpoint uses MLA instead of GQA.

**Contrastive embedding fine-tuning** — trains a dedicated LoRA adapter for retrieval quality via in-batch InfoNCE with hard same-document negatives, plus a supervised path over real (question, answer) pairs extracted from the Stack Exchange RPG dump. Now supports AMP and `torch.compile`.

---

## Agent routing

`AgentRouter` / `MultiAgentEngine` (automatic per-query dispatch across agents, sub-second LoRA + corpus hot-swap) are unchanged in shape; the agent registry now wires CRAG's filter and corrective retry into an agent's configuration, and Saga runs with it enabled.

---

## Corpus toolkit: from collection to curation

**Heuristic quality filter** — a dependency-free, Gopher/C4-style filter (`grimoire_ai/llm/data/quality_filter.py`) that screens documents on character/word minimums, alpha ratio, symbol-junk ratio, mean word length, short-line ratio (catches nav-menu boilerplate), and 3-gram repetition. Opt-in via `--quality-filter` on the preprocessing script, with an optional report. Already used in production: the latest pretrain run dropped 21 documents this way, mostly MathML-noise pages.

**EntiGraph-style entity recombination** — `scripts/generate_open5e_entigraph.py` synthesizes training text by connecting entities from Open5e's structured API data. General LLM-rephrasing augmentation was evaluated and deliberately ruled out (model-collapse risk).

**Q&A pair extraction** — parses the Stack Exchange RPG dump into (query, passage) pairs for both contrastive embedding training and instruction fine-tune data generation.

**Two corpus quality incidents were found and fixed during this cycle:**
- A degenerate fine-tune checkpoint shipped from `--accepted-only` Q&A data where single-answer questions collapsed context and response into the same text, causing question-echoing/repetition. Caught by qualitative review, fixed by rebuilding with `--min-score 1`.
- Several Open5e scraper endpoints (armor, weapons, races, magic items, backgrounds, feats) were mostly third-party homebrew mislabeled as official SRD content, and a formatter bug was leaking `document__url`/license URLs into corpus text. Both fixed; corpus rebuilt filtered to `wotc-srd` only, tracked through three reruns (`weighted_clean` → `v2` → `v3`).

**Also new:** Gutenberg catalog-based bulk scraping (filters the official bulk CSV locally instead of scraping search-result pages, which Gutenberg explicitly warns against), a Stack Exchange RPG data-dump scraper, corrected MinHash+LSH dedup (deterministic SHA-1 shingling, multiplicative hashing), source-based sample weighting, and stratified validation splitting by weight tier so held-out data proportionally covers every tier instead of depending on luck.

---

## UI: split into two apps

The single `app.py` Gradio app has been replaced by two separate apps:

- **Training/eval app** (`grimoire-ui`, port 7860) — Preprocess, Pre-train, Fine-tune, Scale, Evaluate, Ingest, and Corpus tabs.
- **Chat app** (`grimoire-chat-ui`, new console script, port 7861) — the conversational interface, now standalone.

New controls added on top of the split: MLA attention-type dropdown and dimension fields (Pre-train), reranker dropdown, CRAG checkbox and threshold sliders, and repetition-loop-guard checkbox (Chat), sample-weight-building and stratified-validation controls (Pre-train), and a Build/Rebuild index button (Corpus).

---

## Training pipeline

- AMP now defaults to bf16 where the GPU supports it (Ampere+), falling back to fp16 on older CUDA — changes default numerics on newer hardware, though training still runs correctly either way.
- `torch.compile` mode is now an exposed setting rather than hardcoded.
- A cross-cutting stability pass fixed 20 bugs across the model, training, inference, data, and eval pipeline, plus targeted fixes for: training loss inflated by gradient-accumulation steps, post-resume loss skew and a stuck Start button after Stop, guaranteed final checkpoint on non-boundary exit, and embedding resize on vocab-extend resume.

---

## Developer / packaging

- Docker image and `docker-compose.yml` added.
- Console script entry points: `grimoire-chat`, `grimoire-ui`, `grimoire-chat-ui`, `grimoire-train`, `grimoire-finetune`, `grimoire-preprocess`.

---

## Known limitations

- **GGUF export does not support MLA checkpoints** — it raises `NotImplementedError` for any non-GQA attention type, since llama.cpp's conventions for this architecture couldn't be verified without a real binary to test against. GQA checkpoints (the default) export as before.
- **Corpus index memory use is unbounded** — the lexical index holds all data in RAM (confirmed ~10GB resident for 1,469 files, causing OS paging on large corpora). `--corpus-limit` mitigates this for evaluation runs only; production chat-time retrieval is unaffected in scope but not yet fixed in footprint.
- **Repetition loop guard only catches exact repeats**, not templated loops with varying content.

---

## Upgrade notes

- **Existing checkpoints are fully compatible.** `attention_type`, `retro_layers`, and `n_predict` all default to values that reproduce pre-v3.0 behaviour exactly, and checkpoints saved before these settings existed are defaulted transparently on load.
- **Repoint any scripts using the old single `app.py`.** It has been deleted; use `train_app.py` (`grimoire-ui`) for training/eval and the new `chat_app.py` (`grimoire-chat-ui`) for chat.
- The `ui` extra's `gradio` floor is now `>=6.0`.
- All new retrieval and decoding features (reranking, CRAG, constrained decoding, MLA, MTP, RETRO) are off by default — no config changes are needed to keep existing behaviour.
