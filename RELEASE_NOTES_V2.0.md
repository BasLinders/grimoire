# Grimoire v2.0

*Released June 2026 — 67 commits since V1.0*

V2.0 is a complete upgrade of the training pipeline, the retrieval engine, and the corpus toolkit. Every major subsystem has been extended or replaced. The model produces better output, trains faster, and the full workflow — from raw text to a running model — is substantially more capable.

---

## Training quality: six statistical optimisations

Six research-grade techniques have been added to the training pipeline, all opt-in so existing runs are unchanged.

**Entropy-adaptive temperature** — inference now adjusts sampling temperature in real time based on the model's prediction entropy. When the model is confident (low entropy) temperature drops toward a floor; when it's uncertain (high entropy) it rises toward a ceiling. This produces sharper answers on facts the model knows and more varied output where it's genuinely uncertain. The static temperature slider is hidden automatically when this mode is active.

**Bootstrap confidence-interval early stopping** — training stops when the validation loss stops improving, using non-parametric bootstrapping to distinguish real improvement from noise. A 95 % confidence interval is estimated from the last window of validation batches; only improvements larger than the noise band reset the patience counter. `EarlyStopper` is wired into the trainer and configurable via the training config.

**Stochastic Weight Averaging (SWA)** — after a configurable fraction of training (default: the last 25 %), the trainer averages model weights across checkpoints into a wider, flatter loss basin. SWA models generalise better with no additional inference cost. The averaged checkpoint is saved separately at the end of training.

**Importance-weighted sampling** — a new script (`scripts/score_difficulty.py`) measures per-window cross-entropy loss under a checkpoint and writes a weight file. When that file is passed to training, harder sequences are sampled more often via `WeightedRandomSampler`. The distribution shifts as training progresses and the script is re-run.

**Near-duplicate document removal** — `grimoire_ai/llm/data/dedup.py` implements MinHash + LSH deduplication. Documents are shingled into word 5-grams, compressed into 128-permutation MinHash signatures, and bucketed by LSH banding. Candidate pairs whose full-signature agreement meets the threshold are clustered with a union-find; only the longest representative of each cluster is kept. Pass `--dedup` to `preprocessing.py` to enable. Off by default.

**Learning-rate range test** — `scripts/lr_range_test.py` sweeps the LR geometrically from a minimum to a maximum over a configurable number of steps, records EMA-smoothed loss, and suggests a peak LR at the point of steepest descent divided by 10. Outputs a CSV and, if matplotlib is present, a PNG curve. Removes the guesswork from setting `peak_lr`.

---

## Retrieval engine: semantic search

V1.0 offered lexical (BM25-style) retrieval. V2.0 adds semantic retrieval alongside it.

- The model's own token representations are used to embed corpus chunks via cosine similarity — no external model required in the default configuration.
- External sentence-transformer backends (e.g. `all-MiniLM-L6-v2`) are also supported for higher-quality embeddings when `sentence-transformers` is installed.
- A retrieval router selects lexical or semantic retrieval (or both) based on configuration.
- Both the lexical and semantic indices are now **cached to disk** on first build and reloaded in milliseconds on subsequent starts. The 5–10 minute startup delay on large corpora is gone.
- Retrieval config is unified in the UI under a single panel.

---

## Training performance

- **Flash Attention** via PyTorch's scaled dot-product attention (`torch.nn.functional.scaled_dot_product_attention`) — significant VRAM reduction and throughput gain on supported hardware, transparent fallback otherwise.
- **`torch.compile`** applied to the model for faster kernel fusion. Triton errors on Windows are suppressed with a clean eager fallback so the Windows experience is unaffected.
- **`cudnn.benchmark`** enabled; data transfers use `non_blocking=True` to overlap host/device copies with compute.
- **Validation loss** is computed and logged after every eval interval, streamed live to the Pre-train tab.
- **LR scheduler state** is correctly saved and restored on checkpoint resume — previously restarting from a checkpoint reset the cosine schedule to its initial LR.

---

## Corpus toolkit

**New scrapers:**

| Script | Source | Notes |
|---|---|---|
| `scripts/scrape_wikipedia.py` | Wikipedia MediaWiki API | Math/data-science and fantasy/mythology category trees, BFS traversal |
| `scripts/scrape_gutenberg_extended.py` | Project Gutenberg | ~60 additional books: George MacDonald, William Morris, Edgar Rice Burroughs, Andrew Lang colour fairy books, H. G. Wells, Jules Verne, and mathematical texts |
| `scripts/generate_lore.py` | Synthetic generator | D&D stat blocks, spells, magic items, locations, and 20 data-science concept explanations — deterministic at `--seed 42`, scales with `--count` |
| `scripts/scrape_open5e.py` | Open5e API | SRD monsters, spells, magic items |
| `scripts/scrape_forgotten_realms.py` | Forgotten Realms wiki | Lore articles via MediaWiki action API |
| `scripts/scrape_gutenberg.py` | Project Gutenberg | Fantasy and mythology public-domain books (curated list) |

**Wikipedia rate-limit resilience** — the scraper retries on HTTP 429 and 5xx with exponential backoff starting at 2 s, honouring the `Retry-After` header. Per-category progress is printed during the collection phase so long runs are no longer silent.

---

## UI

- **Streaming chat** — responses are streamed token-by-token instead of displayed all at once after a progress bar. The model appears to "type" its answer.
- **Dataset builder** — a panel in the Chat tab lets you collect fine-tuning pairs directly from conversations. Pairs are appended to an existing JSONL file so you can build a dataset incrementally across sessions.
- **Scale tab** — a Chinchilla scaling calculator. Enter a token budget or a model size and it computes the optimal counterpart, suggested training steps, and estimated compute.
- **Model size presets** — one-click presets for small (25 M), medium (85 M), and large (250 M) parameter budgets configure the full model architecture automatically.
- **Model config display** — the Chat tab shows the loaded model's architecture so you know exactly what checkpoint is running.
- **Multiple file upload** — the Ingest tab accepts multiple files in a single upload.
- **Tooltips** — every UI field has an inline tooltip explaining its meaning and sensible defaults.
- **Adaptive temperature UI** — enabling adaptive temperature hides the static temperature slider, since the two controls are mutually exclusive.
- **Button feedback** — buttons show a pressed state via CSS `:active` so clicks feel responsive.
- Contrast fixes across light and dark mode to meet WCAG AA on all interactive elements.

---

## Developer / packaging

- Package renamed from `grimoire` to `grimoire-ai` / `grimoire_ai` throughout.
- BATCH launch file and icon added for Windows one-click startup.
- `saga_v2.jsonl` — 36 expanded fine-tuning pairs replacing the original slim set.
- Wikitext markup stripper added as a shared utility for wiki-based scrapers.
- Stale OS directory cache flushed before corpus glob so newly added files are always picked up.

---

## Upgrade notes

- Existing checkpoints are fully compatible — no architecture changes.
- The `grimoire` import name has changed to `grimoire_ai`. Update any custom scripts accordingly.
- Near-duplicate removal (`--dedup`) and all six statistical optimisations are **off by default**. No config changes are needed to keep existing behaviour.
- The corpus index cache is stored alongside the corpus by default. On first load after upgrading the index will rebuild once and be cached for all subsequent starts.
