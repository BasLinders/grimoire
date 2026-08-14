# Inference & Retrieval Optimization — Candidate Improvements

Candidate changes to make *inference-time* code — generation, retrieval,
corpus-prep scripts, and the UI backend that drives them — run faster or do
less redundant work. This is a third, distinct axis from the other two docs:
[`speed_optimization.md`](speed_optimization.md) is scoped to the *training
loop* only (`Trainer`, `EmbedTuner`, data collation), and
[`architecture_optimization.md`](architecture_optimization.md) items #6-9
are model-quality/architecture changes, not speed. Nothing below overlaps
either — this is the "everything after a checkpoint is trained" path:
loading it, running it, and retrieving context for it.

Written after an audit of `grimoire_ai/llm/inference/`,
`grimoire_ai/llm/model/`, `grimoire_ai/corpus/`, the retrieval modules, and
the corpus-prep scripts under `scripts/`, looking specifically for
redundant computation, quadratic patterns, and asymmetries between two
near-identical code paths where only one got an optimization. Several items
below were found exactly that way — a fix present in one function/branch
but not its sibling.

## Status

| # | Item | Where | Status |
|---|---|---|---|
| 1 | `output_head` projects the full prompt during prefill, not just the last token | `model/transformer.py` | not started |
| 2 | Cached (decode-step) attention never uses fused SDPA | `model/attention.py` | not started |
| 3 | `generate()` computes softmax twice in top-p sampling | `inference/sampler.py` | not started |
| 4 | `repetition_penalty` loop does per-token scalar tensor writes | `inference/sampler.py` | not started |
| 5 | `chat_stream()` re-decodes the full sequence on every yielded token | `inference/engine.py` | not started |
| 6 | `StatBlockConstraint.mask()` compounds full-redecode + per-step vocab scan | `inference/constrained_decoding.py` | not started |
| 7 | RETRO neighbor-precompute script embeds one window at a time | `scripts/build_retrieval_neighbors.py` | not started |
| 8 | Lexical (Jaccard) corpus query is an O(N) linear scan | `corpus/corpus.py` | not started |
| 9 | External-encoder (MiniLM/MPNet) index rebuilds from scratch on every UI load | `ui/chat_app.py` | not started |

## 1. `output_head` runs on the full prompt during prefill, when only the last position is used

```python
x = self.final_norm(x)
logits = self.output_head(x)
```
([`transformer.py:257-258`](../grimoire_ai/llm/model/transformer.py#L257-L258))

`forward()` always projects the full `(batch, seq_len, d_model)` hidden
state through `output_head` (`nn.Linear(d_model, vocab_size)`). During the
prompt/prefill pass, both `generate()` and `generate_stream()`
([`sampler.py:210`](../grimoire_ai/llm/inference/sampler.py#L210),
[`sampler.py:337`](../grimoire_ai/llm/inference/sampler.py#L337)) call
`model(prompt_tensor, use_cache=True)` and then take only `logits[0, -1, :]`
— every other position's logits are computed and immediately discarded.
With `vocab_size=16384` vs `d_model=512` on the small preset
([`config.py:103-104`](../grimoire_ai/llm/model/config.py#L103-L104)),
`output_head` is a ~8.4M-param matmul, run over the full prompt length (up
to `max_context_tokens=512`) instead of 1 token — comparable in FLOPs to a
couple of transformer blocks, wasted on every single generation call.

This matters more here than in a typical chat LLM because RAG-grounded
prompts (the whole point of this project) are exactly the long-prompt case.

**Fix shape:** slice to the last position (`x[:, -1:, :]`) before
`output_head` on the inference prefill call, while leaving the
full-sequence projection intact for the training path (which needs every
position for the loss) and for `return_mtp_logits` (MTP heads need the full
trunk output too). **Confidence:** high. **Impact:** high, scales linearly
with prompt length.

## 2. Cached decode steps never use the fused SDPA/Flash-Attention path

```python
use_sdpa = past_kv is None and hasattr(F, "scaled_dot_product_attention")
```
([`attention.py:265`](../grimoire_ai/llm/model/attention.py#L265))

SDPA is only attempted on the prefill pass. Every subsequent single-token
decode step (`past_kv is not None`) falls through to the manual
`matmul → +mask → softmax → nan_to_num → dropout → matmul` path
([`attention.py:294-305`](../grimoire_ai/llm/model/attention.py#L294-L305)),
even though a `seq_len=1` query with no `attention_mask` — true for every
call from `sampler.py`, confirmed neither `generate()` nor
`generate_stream()` ever passes `attention_mask` during generation — has a
trivially no-op causal mask and could safely use
`F.scaled_dot_product_attention(q, k, v, is_causal=False)`. The existing
comment justifying the fallback ("SDPA's is_causal flag assumes full
sequences") is correct for the padded/masked case but doesn't address the
simpler no-mask decode case that generation actually is.

This is the code path exercised on *every generated token of every
response* — the dominant cost center of interactive inference — currently
skipping PyTorch's fused/Flash-Attention kernel in favor of eager ops plus
an extra `nan_to_num` and a `Dropout` module call.

**Fix shape:** widen the `use_sdpa` gate to also cover
`past_kv is not None and attention_mask is None`, dispatching to SDPA with
`is_causal=False` in that case (the KV cache already contains the correct
causal history; a single new-token query attending to all cached keys needs
no additional masking). **Confidence:** high. **Impact:** medium-high.

## 3. `generate()` computes softmax twice in top-p sampling — the exact bug `generate_stream()` already fixed

```python
sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > config.top_p
```
([`sampler.py:386-389`](../grimoire_ai/llm/inference/sampler.py#L386-L389),
in `generate()`)

Compare to `generate_stream()`
([`sampler.py:253-257`](../grimoire_ai/llm/inference/sampler.py#L253-L257)),
which has an explicit comment — *"Compute softmax once and reuse to avoid
fragile double-call"* — and does exactly that. `generate()` still has the
double-call bug its sibling function documents fixing; the fix just never
got ported over. `top_p=0.9` is `GenerationConfig`'s default, so this runs
on the default path for every generated token in the non-streaming
`respond()`/`chat()` path.

**Fix shape:** mirror `generate_stream()`'s pattern — compute
`F.softmax(sorted_logits, dim=-1)` once into a local, reuse it for both the
cumsum and the subtraction. **Confidence:** high (verbatim diff between two
near-identical functions). **Impact:** low-medium in isolation, but 100%
wasted and trivial to fix.

## 4. `repetition_penalty` loop does per-token Python-level scalar tensor writes

```python
if config.repetition_penalty != 1.0 and generated:
    for token_id in set(generated):
        if next_logits[token_id] > 0:
            next_logits[token_id] /= config.repetition_penalty
        else:
            next_logits[token_id] *= config.repetition_penalty
```
(identically in both `generate()` and `generate_stream()`,
[`sampler.py:350-355`](../grimoire_ai/llm/inference/sampler.py#L350-L355) /
[`sampler.py:216-221`](../grimoire_ai/llm/inference/sampler.py#L216-L221))

A Python loop over `set(generated)` (grows with response length) doing
scalar tensor indexing (read + write) per iteration, once per decode step —
O(n) per step, O(n²) over a full generation. Scalar tensor indexing is
particularly costly on CUDA/MPS, where each access is effectively its own
tiny kernel launch. Opt-in (`repetition_penalty` defaults to `1.0` = off),
but a real, user-facing knob.

**Fix shape:** vectorize with `torch.where`/index tensors — build a tensor
of unique generated token ids once (`torch.unique`) and apply the
sign-dependent scaling in one vectorized op instead of a Python loop.
**Confidence:** high. **Impact:** medium, opt-in only.

## 5. Streaming chat re-decodes the entire generated sequence on every yielded token

```python
for token_id in generate_stream(...):
    generated_ids.append(token_id)
    # Decode the full sequence so far to handle multi-byte BPE tokens.
    yield self.tokenizer.decode(generated_ids).strip()
```
([`engine.py:510-521`](../grimoire_ai/llm/inference/engine.py#L510-L521),
`chat_stream()`)

`tokenizer.decode()` is O(len(ids)); calling it on every newly-appended
token makes total decode work O(n²) in response length. This is the live
UI streaming path — `chat_app.py`'s `chat()` handler iterates
`engine_state.chat_stream(...)` directly. In absolute terms this is likely
small next to the model forward pass (decode is a cheap per-character
loop), but it's a genuinely unscoped quadratic pattern that gets worse the
higher a user sets `max_new_tokens`.

**Fix shape:** the comment ("handle multi-byte BPE tokens") suggests the
full redecode exists because a single new token might complete a
multi-byte/multi-token character sequence that only decodes correctly with
some trailing context — so a full redecode-from-scratch isn't purely
accidental. A bounded fix (redecode only the last few tokens' worth of
context plus the fixed prefix text, not the whole sequence) would keep
correctness while dropping the quadratic factor. **Confidence:** high
pattern exists. **Impact:** low-medium.

## 6. `StatBlockConstraint.mask()` compounds the same full-redecode pattern, plus its own per-step scan

```python
def mask(self, logits, generated, tokenizer):
    text_so_far = tokenizer.decode(generated)          # O(n) every step
    active = self._active_field(text_so_far)            # 4x regex .finditer() over growing text
    ...
    for token_id, token_text in self._alphabet_token_ids.items():
        candidate = value_so_far + token_text
        if len(candidate) <= spec.max_value_len and spec.grammar.is_valid_prefix(candidate):
            allowed.add(token_id)
```
([`constrained_decoding.py:366-396`](../grimoire_ai/llm/inference/constrained_decoding.py#L366-L396),
called every decode step from both `generate()` and `generate_stream()`
when `stat_block_constraint` is set)

Every step: a full `decode(generated)`, 4 regex passes over the resulting
(growing) string, and — while a field is active — an iteration over the
digit/comma/slash/space vocab subset running `is_valid_prefix` on each.
Opt-in (`stat_block_constraint_enabled` checkbox in the UI), typically
active only for the short span of a stat-block response, but stacks a
second independent O(n²) source on top of item #5 when both streaming and
this constraint are active together.

**Fix shape:** cache the decoded prefix and field-boundary regex results
across steps instead of recomputing from scratch each time (only the
newest token needs incorporating). **Confidence:** high. **Impact:**
medium, opt-in.

## 7. RETRO neighbor-precompute script embeds one window at a time

```python
for start in range(0, len(dataset), embed_batch_size):
    batch_indices = range(start, min(start + embed_batch_size, len(dataset)))
    window_texts = [...]          # decoded in a "batch" of embed_batch_size
    for local_i, idx in enumerate(batch_indices):
        window_text = window_texts[local_i]
        results = retriever.query(window_text, top_k=query_k)   # one window at a time
```
([`scripts/build_retrieval_neighbors.py:152-169`](../scripts/build_retrieval_neighbors.py#L152-L169))

`SemanticRetriever.query()`
([`semantic.py:304`](../grimoire_ai/retrieval/semantic.py#L304)) always
calls `self._embed_fn([text])` — a batch-size-1 forward pass. The outer
loop groups windows into chunks of `embed_batch_size` (default 32) purely
for token-decode and progress reporting; the actually expensive part
(embedding via the transformer, which `InferenceEngine.embed` already
supports batching for) runs one window at a time inside `retriever.query()`
regardless. For a real pretraining corpus this is the RETRO
neighbor-precompute step, run once per training window — potentially tens
or hundreds of thousands of unbatched forward passes, an order of magnitude
slower in throughput than the same work batched on a GPU.

**Fix shape:** add a batched-query path to `SemanticRetriever` (embed a
list of texts in one `_embed_fn` call, then do the top-k search per row),
and have the script call it once per `embed_batch_size` chunk instead of
per window. **Confidence:** high. **Impact:** high — offline script, but
directly gates how practical the RETRO neighbor-precompute step is at real
corpus scale.

## 8. Lexical (Jaccard) corpus query is a full O(N) linear scan

```python
for mt, entry in self._index.all_entries().items():
    overlap = len(set(mt) & query_set)
    if overlap > 0:
        union = len(set(mt) | query_set)
        scores[mt] = (overlap / union, entry.frequency)
```
([`corpus.py:244-249`](../grimoire_ai/corpus/corpus.py#L244-L249))

`CorpusIndex` is a hash map giving O(1) exact-tuple lookup, and its own
module docstring explicitly claims O(1) lookups — but `GrimoireCorpus.query()`
doesn't use point lookups at all; it computes Jaccard similarity against
*every* stored multi-token on every query, with no inverted index from
individual stemmed words to candidate n-grams. This is a live,
user-selectable path: the UI exposes "Lexical (Jaccard)" as an encoder
option explicitly described as "no neural embedding, CPU-only, instant
startup" — i.e. marketed as the fast/lightweight choice, which is exactly
the choice most likely to be picked on hardware where an O(N) per-query
scan matters most. Unlike `SemanticRetriever`/`RagIndex` (FAISS + brute-force
fallback, both reasoned about explicitly), the lexical path has no
equivalent optimization or acknowledgment of the cost.

**Fix shape:** build a secondary inverted index (stemmed word → set of
n-grams containing it) at ingest time, and use it to narrow the candidate
set before computing Jaccard scores, instead of scanning every stored
n-gram. **Confidence:** high. **Impact:** medium, scales with corpus size.

## 9. External-encoder retrieval index rebuilds from scratch on every UI load

```python
if use_external:
    embed_fn = make_external_embed_fn(EXTERNAL_ENCODERS[encoder])
    retriever = SemanticRetriever(embed_fn=embed_fn)
    for text, source in documents:
        retriever.add_text(text, source=source)
    retriever.index()          # always re-embeds everything, no persistence
    engine.corpus = retriever
else:
    # Default: model's own decoder embeddings — use persistent index when fresh.
    index_dir = _semantic_index_dir(...)
    if index_dir and _index_is_fresh(index_dir, ...):
        engine.corpus = SemanticRetriever.from_index(index_dir, embed_fn=engine.embed)
```
(present identically in `load_agent`
[`chat_app.py:190-199`](../grimoire_ai/ui/chat_app.py#L190-L199) and
`load_engine`
[`chat_app.py:322-337`](../grimoire_ai/ui/chat_app.py#L322-L337))

The native "Model (decoder embeddings)" branch has a full `RagIndex`
freshness check (MD5-hashed sources + checkpoint, with an mtime-based
shortcut) and persists the built index. The `use_external` branch has none
of that — every "Load" click with MiniLM or MPNet selected re-embeds the
*entire* corpus through the external sentence-transformers model
synchronously on the Gradio callback thread, even if nothing changed since
the last load. Per [[project_embedding_retrieval_priority]], MiniLM/MPNet
are fallback/benchmark encoders rather than the priority path, but the UI
still exposes them as a first-class selectable option that blocks on every
reload regardless of priority.

**Fix shape:** extend the same freshness-check + persistence pattern
(`_index_is_fresh`/`save_index`/`from_index`) to the external-encoder
branch, keyed by encoder name so switching encoders still triggers a
rebuild but reloading the same one doesn't. **Confidence:** high — the
asymmetry with the sibling branch is unambiguous. **Impact:**
medium-high, blocks the UI on every reload.

## Considered and ruled out

Checked and found already handled correctly — listed so a future audit
doesn't re-check them:

- **`SemanticRetriever`/`RagIndex`** — already FAISS `IndexFlatIP` with
  brute-force fallback, length-sorted batching before embedding (~3.4x
  documented saving), and MD5-hash staleness caching with an mtime/size
  shortcut.
- **BPE tokenizer** (`bpe.py`) — `_encode_word` memoizes per-word
  encodings in `_word_cache`; the word-split regex is compiled once at
  module scope; `decode` is a single O(n) pass.
- **`scripts/dedup_corpus.py`** — scoped to `new_files × all_files`, not
  `all_files × all_files`; MinHash comparison is a vectorized numpy `==`
  over a 64-element signature.
- **`scripts/build_source_weights.py` / `data/sample_weights.py`** —
  `compute_window_weights` uses `np.searchsorted`, already vectorized.
- **`scripts/score_difficulty.py`** — already batches via `DataLoader`
  over the full corpus.
- **RoPE/causal-mask precompute** (`attention.py`) — computed once in
  `__init__`, registered as buffers, not recomputed per forward call.
- **UI `chat()` handler** (`chat_app.py`) — `history_to_messages` is
  computed once before the streaming loop starts, not per token.
- **Prompt-building** (`prompt.py`) — single-pass string join + one
  `tokenizer.encode()` call, no redundant work.

One documentation-only issue noted in passing (not a perf bug): `sampler.py`'s
module docstring still says generation "re-runs the full forward pass on
every step... deferred to Phase 5" — false today, a full KV cache is
already implemented. Worth a one-line fix so it doesn't mislead someone
into re-implementing something that already exists.

## Recommendation

**#1-#3 first** — all three sit directly in the hot path of every single
generation call (prefill projection, decode-step attention, sampling), are
individually contained (one function or one conditional each), and #3 in
particular is close to a copy-paste of a fix that already exists two
functions away. **#7 and #9** next — both offline/load-time rather than
per-token, but #7 gates whether RETRO neighbor-precompute is practical at
real corpus scale and #9 is a synchronous UI stall on a path users can
actually select today. **#4, #5, #6, #8** last — all real, but either
opt-in (#4, #6), small in absolute terms (#5), or scale with corpus size
rather than being hit on every request (#8).
