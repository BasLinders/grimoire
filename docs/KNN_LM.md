# Scoped kNN-LM + best-of-N: next-token and answer scoring

Plan and design record for adding a scoped kNN-LM layer and best-of-N
answer reranking to Grimoire's retrieval-augmented generation path.

## Goal

Improve two scoring mechanisms, and measure them with the existing quiz eval
(`grimoire_ai/llm/eval/quiz.py`: token F1, keyword recall, pass rate):

1. **Next-token scoring.** Interpolate the LM's next-token distribution with
   a kNN distribution built from the passages retrieved for *this* query, so
   the model is pulled toward tokens that actually appear in its context.
2. **Finished-answer scoring.** Sample N candidate answers and pick the
   best-grounded one with a score that is independent of the sampler.

**Out of scope:** passage retrieval and ranking. kNN-LM does not change which
passages are retrieved. That is `SemanticRetriever` + `embed_tune` + the
cross-encoder `Reranker` (see `architecture_optimization.md` items #4 and #7).

## Origin

The design started as an external nanoGPT-oriented bundle (`knn_fusion.py`,
`bon_rerank.py`, `pipeline.py`, `eval_factual.py`, `pairs.py`, `dpo.py` +
tests + `SPEC.md`). The math is sound, but the bundle assumed nanoGPT's
interface throughout and does not run against `GrimoireTransformer` as-is.
The port below keeps the math and replaces the plumbing.

## Design

### kNN distribution and fusion (kept from the bundle)

- Store: hidden state at position *t* (output of `final_norm`, the input to
  the tied `output_head`) → token id at *t+1*.
- `p_knn`: softmax over `-d² / T` across the k nearest keys, aggregated by
  value token.
- Fusion in log space: `log((1-λ)·p_lm + λ·p_knn)`.
- `λ` comes from a small learned gate over `[log1p(d_min), log1p(d_mean),
  LM entropy]`, capped at `lam_max` (0.8). The gate also owns the learned
  temperature `T`.

### What changes in the port

| Bundle | Grimoire port | Why |
|---|---|---|
| Store keys from passages encoded standalone, one extra forward per passage | Keys taken from the **prompt prefill pass**, over the token span between the two `<SEP>` markers | Free (prefill already computes them), and keys and queries then come from the same in-prompt context instead of drifting apart |
| FAISS `IndexFlatL2`, CPU round-trip every step | `torch.cdist` + `topk` on the model's device | The store is ~hundreds of keys; FAISS is overkill and the per-step GPU→CPU copy dominates. No new dependency |
| Own batched sampler (`generate_candidates`), no KV cache | Optional logit hook inside `sampler.generate` | Keeps the KV cache, repetition penalty, `RepetitionLoopGuard`, stat-block constraint, top-k/top-p. A separate sampler would regress the loop fixes from `training_PLAN.md` Steps 7-10 |
| `fit_gate` scores the reference answer **without its prompt** | Scores answer tokens only, conditioned on the full `<BOS><SEP>ctx<SEP><USR>q<AST>` prompt | The bundle calibrates the gate on a regime that never occurs at inference, and its LM-vs-fused NLL diagnostic flatters kNN (the LM is guessing without the question) |
| Gate not persisted | Saved next to the checkpoint, tagged with the checkpoint it was fitted on | The gate is only valid for the hidden-state geometry it was fitted to |
| `init_temperature = 10` | Initialized from the median key distance | Squared L2 between RMSNorm outputs at `d_model=512` is far larger than 10, so the default starts as near 1-NN |
| Forward hook on `ln_f`, always on | Captured only during generation | `final_norm` also runs inside `embed()` (semantic retrieval with `--encoder model`) |
| `model(x, y)` → `(logits, loss)` | `model(x)` → logits tensor | Grimoire's 2nd positional arg is `attention_mask`, and it returns a bare tensor. `logits, _ = model(...)` silently unpacks the batch dim when batch == 2 and raises otherwise |

### Best-of-N (finished-answer scoring)

`grounding_score` = 60% claim support (numbers + capitalized entities found in
the passages) + 40% content-word precision, plus a small mean-logprob
tiebreak. Fixes needed before use:

- **Dice notation counts as ungrounded.** `_NUM` splits `2d6` into `2` and
  `6`, but `normalize` keeps `2d6` as one token, so the claim never matches.
  Match dice (`\d*d\d+`) as a single claim.
- Every capitalized sentence start counts as an "entity". Strip the
  sentence-initial word unless it recurs mid-sentence.

Abstain / extractive fallback stays eval-only by default: a pasted passage
sentence is a poor chat reply.

### Integration points

- `grimoire_ai/llm/inference/knn_lm.py`: store, distribution, gate, fusion,
  gate persistence.
- `sampler.generate` / `generate_stream`: optional per-step logit hook.
- `InferenceEngine(knn_lm=..., n_candidates=...)`: same optional-layer
  pattern as `reranker=` / `crag_filter=`.
- `scripts/eval/evaluate.py`: ablation flags on the existing quiz eval
  instead of the bundle's parallel `eval_factual.py` harness.

## PR plan

| PR | Scope | Who runs it | Status |
|---|---|---|---|
| 1 | `knn_lm.py`: torch `ScopedStore`, `knn_distribution`, `LambdaGate` (median-distance init), `fuse`, gate save/load + checkpoint tag. Unit tests. No engine changes | Code + unit tests | ✓ merged (#215) |
| 2 | Sampler logit hook; store built from prefill hidden states over the `<SEP>` span; `InferenceEngine(knn_lm=...)`. Test: λ forced to 0 + fixed seed is token-identical to plain `generate()` | Code + unit tests | in progress |
| 3 | `scripts/eval/fit_knn_gate.py`: prompt-conditioned gate fitting on `{user, assistant, context}` records; prints LM vs fused NLL; saves gate. **Go/no-go** | User (GPU, checkpoint) | not started |
| 4 | `evaluate.py` ablation: `--knn-gate PATH`, `--best-of-n N`, `copy_ratio` column. 4-row table: plain / +kNN / +BoN / +both | User | not started |
| 5 | Best-of-N with the fixed grounding score; `InferenceEngine(n_candidates=N)` reusing `generate()` | Code + unit tests | not started |

**Go/no-go after PR 3:** if fused NLL is not below LM NLL on the
prompt-conditioned dev set, kNN is not helping. Stop the kNN track and check
retrieval quality and key/query alignment before tuning further. PR 5 stands
on its own either way.

### Dropped / deferred

- `pipeline.py`: duplicates `InferenceEngine`. Dropped.
- `eval_factual.py`: folded into the existing quiz eval (PR 4). Its
  `retrieval_hit` metric needs gold-source annotations the quiz doesn't have
  yet.
- `pairs.py` / `dpo.py` (preference tuning): deferred until PR 4 shows
  kNN + BoN actually helps. When picked up, train a LoRA adapter instead of
  all weights (cheaper, and the frozen base doubles as the reference model).
  The bundle's DPO code has the same `model(x, y)` interface bug.

## Data caveats

- **Gate-fitting dev set must not be SFT training data.** On memorized
  examples the plain LM looks so good that the gate learns to ignore kNN.
  `scripts/finetune/data/saga_v1.jsonl` has only 31 records with `context`;
  `data/finetune/saga_se_qa.jsonl` is untracked (regenerate with
  `scripts/finetune/build_finetune_data_from_qa.py`). Hold a slice out of
  both SFT and gate fitting.
- **`scripts/eval/data/saga_quiz.jsonl` (49 items) is the test set.** Never
  fit or tune on it.
- The retrieval corpus overlaps the pretraining corpus, so the bundle's
  "eval docs not in LM training data" split rule can't hold as written.
  Results measure in-domain grounding, not generalization to unseen docs.

## Guardrails to watch in the eval table

- `copy_ratio` climbing toward 1.0 means the model is pasting passages.
- Creative registers (no retrieval) should be unaffected. With no
  `<SEP>` context there is no store, so the hook is a no-op. Verify by hand.
