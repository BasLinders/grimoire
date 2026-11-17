"""Fit the scoped kNN-LM gate (``knn_lm.LambdaGate``) on held-out examples.

PR 3 in ``docs/KNN_LM.md``. Also the go/no-go check for the whole kNN track:
if the fused model's held-out NLL isn't below the plain LM's, the store isn't
adding information the model can't already get by attending to the prompt.

How fitting works
-----------------
Each example is a real inference-shaped sequence::

    <BOS><SEP>{context}<SEP><USR>{question}<AST>{answer}<EOS>

built with the same ``PromptBuilder`` the engine uses. One forward pass per
example gives the ``final_norm`` hidden states for every position. The store
is built from the context block exactly as ``KNNSession.prefill`` builds it
at inference. Then every *answer* position is searched against it.

Only answer tokens are scored, and always conditioned on the full prompt.
(The original nanoGPT bundle scored the reference answer with no prompt in
front of it, which calibrates the gate for a regime that never happens and
makes kNN look better than it is.)

Per answer position, ``collect_features`` caches just what the loss needs:

- neighbor distances ``D`` (padded to ``k`` with ``+inf``),
- which neighbors' value token equals the target (``hit``),
- the LM's log-prob of the target,
- the gate's input features (``LambdaGate.features``).

The fused target log-prob is then
``log((1-λ)·p_lm(y) + λ·p_knn(y))`` with
``p_knn(y) = Σ_{hit} softmax(-D/T)``, which is exactly what ``fuse`` computes
for the target column. ``fit_gate`` minimises its mean NLL over the cache
with full-batch Adam: no model in the loop, so fitting takes seconds.

Data hygiene (see ``docs/KNN_LM.md``)
-------------------------------------
Fit on examples the model was *not* fine-tuned on. On memorised examples the
LM already predicts the answer, so the gate learns to ignore the store. The
quiz set is the test set and must never be used here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from grimoire_ai.llm.inference.knn_lm import (
    KNNConfig,
    LambdaGate,
    ScopedStore,
    context_span,
)


@dataclass
class GateExample:
    """One fitting example: an inference-shaped prompt and its reference answer.

    Attributes:
        prompt_ids: ``[BOS, SEP, *ctx, SEP, USR, *question, AST]``.
        answer_ids: Reference answer tokens, usually ending in ``EOS``.
    """

    prompt_ids: list[int]
    answer_ids: list[int]


@dataclass
class GateFeatures:
    """Cached per-answer-position quantities for gate fitting.

    All tensors share the leading dimension ``N`` (answer positions across
    every usable example) and live on CPU.

    Attributes:
        D: ``(N, k)`` squared distances, ascending, ``+inf`` padded.
        hit: ``(N, k)`` whether each neighbor's value token is the target.
        lm_target_logp: ``(N,)`` LM log-prob of the target token.
        feats: ``(N, 3)`` ``LambdaGate.features`` inputs.
        example: ``(N,)`` index of the example each position came from (for
            splitting by example, never by position).
        n_examples: Number of examples that produced positions.
        n_skipped: Examples skipped (no context block, or no room for any
            answer token).
    """

    D: torch.Tensor
    hit: torch.Tensor
    lm_target_logp: torch.Tensor
    feats: torch.Tensor
    example: torch.Tensor
    n_examples: int
    n_skipped: int

    @property
    def n_positions(self) -> int:
        return self.D.size(0)


@torch.no_grad()
def collect_features(
    model,
    examples: list[GateExample],
    cfg: KNNConfig,
    device: "str | torch.device" = "cpu",
    on_progress: Optional[Callable[[str], None]] = None,
) -> GateFeatures:
    """Run each example once and cache what gate fitting needs.

    Sequences longer than ``model.config.max_seq_len`` have their answer
    truncated (never the prompt, which would break the context block).

    Args:
        model: A ``GrimoireTransformer``. Put in eval mode here.
        examples: Fitting examples.
        cfg: ``k`` (neighbors per position) is the only field used here.
        device: Device the model runs on.
        on_progress: Optional callback for periodic progress lines.

    Returns:
        ``GateFeatures`` on CPU.

    Raises:
        ValueError: If no example produced any scorable position.
    """
    model.eval()
    max_len = model.config.max_seq_len
    Ds, hits, lm_ts, feats, ex_idx = [], [], [], [], []
    n_used = n_skipped = 0

    captured: dict[str, torch.Tensor] = {}
    handle = model.final_norm.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("h", out.detach())
    )
    try:
        for i, ex in enumerate(examples):
            span = context_span(ex.prompt_ids)
            room = max_len - len(ex.prompt_ids)
            if span is None or room < 1 or not ex.answer_ids:
                n_skipped += 1
                continue
            ids = ex.prompt_ids + ex.answer_ids[:room]
            P = len(ex.prompt_ids)

            logits = model(torch.tensor([ids], device=device))
            hidden = captured["h"][0]

            store = ScopedStore(hidden.size(-1), device=hidden.device)
            store.add_transitions(hidden, ids, *span)

            # Position t predicts ids[t + 1]; answer tokens are ids[P:].
            pos = torch.arange(P - 1, len(ids) - 1, device=hidden.device)
            targets = torch.tensor(ids[P:], device=hidden.device)

            D, I = store.search(hidden[pos], cfg.k)
            hit = store.values[I] == targets[:, None]
            lm_lp = logits[0, pos].float().log_softmax(-1)
            lm_t = lm_lp.gather(-1, targets[:, None]).squeeze(-1)

            pad = cfg.k - D.size(1)
            if pad > 0:
                D = torch.cat([D, D.new_full((D.size(0), pad), float("inf"))], dim=1)
                hit = torch.cat([hit, hit.new_zeros((hit.size(0), pad))], dim=1)

            Ds.append(D.cpu())
            hits.append(hit.cpu())
            lm_ts.append(lm_t.cpu())
            feats.append(LambdaGate.features(D, lm_lp).cpu())
            ex_idx.append(torch.full((len(pos),), n_used, dtype=torch.long))
            n_used += 1

            if on_progress and (n_used == 1 or n_used % 100 == 0):
                on_progress(f"  collected {n_used} example(s) ({i + 1}/{len(examples)} seen) …")
    finally:
        handle.remove()

    if not Ds:
        raise ValueError(
            f"no usable examples: all {n_skipped} lacked a context block or had "
            "no room for answer tokens"
        )
    return GateFeatures(
        D=torch.cat(Ds),
        hit=torch.cat(hits),
        lm_target_logp=torch.cat(lm_ts),
        feats=torch.cat(feats),
        example=torch.cat(ex_idx),
        n_examples=n_used,
        n_skipped=n_skipped,
    )


def fused_target_logp(
    gate: LambdaGate, f: GateFeatures, mask: Optional[torch.Tensor] = None, eps: float = 1e-10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused log-prob of each target token, and the λ used.

    Equal to the target column of ``knn_lm.fuse`` (same clamp and ``eps``),
    computed from the cache without the model.

    Args:
        gate: The gate to evaluate.
        f: Cached features.
        mask: Optional ``(N,)`` bool selecting positions.
        eps: Floor on ``p_knn`` before the log (``KNNConfig.eps``).

    Returns:
        ``(logp, lam)``, both ``(n,)``.
    """
    D, hit, lm_t, feats = f.D, f.hit, f.lm_target_logp, f.feats
    if mask is not None:
        D, hit, lm_t, feats = D[mask], hit[mask], lm_t[mask], feats[mask]
    w = torch.softmax(-D / gate.temperature, dim=-1)
    p_knn_t = (w * hit).sum(-1)
    lam = gate.lambda_from_features(feats).clamp(1e-6, 1 - 1e-6)
    logp = torch.logaddexp(
        torch.log1p(-lam) + lm_t,
        torch.log(p_knn_t + eps) + torch.log(lam),
    )
    return logp, lam


def split_by_example(
    f: GateFeatures, val_frac: float = 0.2, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Position masks ``(train, val)``, splitting whole examples.

    Splitting by position would leak: neighboring positions of the same
    answer share a store and most of their context.

    Raises:
        ValueError: If there are fewer than 2 examples, or ``val_frac`` is
            outside ``(0, 1)``.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if f.n_examples < 2:
        raise ValueError("need at least 2 examples to hold one out")
    order = list(range(f.n_examples))
    random.Random(seed).shuffle(order)
    n_val = min(max(1, round(f.n_examples * val_frac)), f.n_examples - 1)
    val_ex = torch.tensor(sorted(order[:n_val]))
    val = torch.isin(f.example, val_ex)
    return ~val, val


def fit_gate(
    f: GateFeatures,
    train_mask: torch.Tensor,
    cfg: KNNConfig,
    epochs: int = 300,
    lr: float = 0.05,
    on_progress: Optional[Callable[[str], None]] = None,
) -> LambdaGate:
    """Fit gate weights + temperature by minimising mean fused NLL.

    Starts from ``LambdaGate.from_config(cfg)`` with the temperature
    calibrated to the median training distance, then runs full-batch Adam.

    Args:
        f: Cached features.
        train_mask: ``(N,)`` positions to fit on.
        cfg: ``lam_max``, ``init_temperature`` and ``eps`` are used.
        epochs: Full-batch optimisation steps.
        lr: Adam learning rate.
        on_progress: Optional callback for periodic loss lines.

    Returns:
        The fitted gate, in eval mode.
    """
    gate = LambdaGate.from_config(cfg)
    gate.calibrate_temperature(f.D[train_mask])
    opt = torch.optim.Adam(gate.parameters(), lr=lr)
    for epoch in range(epochs):
        logp, _ = fused_target_logp(gate, f, train_mask, cfg.eps)
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if on_progress and (epoch == 0 or (epoch + 1) % 100 == 0 or epoch == epochs - 1):
            on_progress(
                f"  epoch {epoch + 1}/{epochs}: train fused NLL {loss.item():.4f} "
                f"(T={gate.temperature.item():.2f})"
            )
    return gate.eval()


@torch.no_grad()
def gate_report(
    gate: LambdaGate, f: GateFeatures, mask: torch.Tensor, eps: float = 1e-10
) -> dict:
    """Held-out diagnostics for the go/no-go decision.

    Returns:
        Dict with ``n_tokens``, ``lm_nll`` and ``fused_nll`` (nats/token),
        ``delta_nll`` (fused − LM; negative means kNN helps),
        ``mean_lambda``, and ``coverage`` (share of positions whose target
        appears among the retrieved neighbors at all; an upper bound on
        where kNN *can* help).
    """
    logp, lam = fused_target_logp(gate, f, mask, eps)
    lm_nll = float(-f.lm_target_logp[mask].mean())
    fused_nll = float(-logp.mean())
    return {
        "n_tokens": int(mask.sum()),
        "lm_nll": lm_nll,
        "fused_nll": fused_nll,
        "delta_nll": fused_nll - lm_nll,
        "mean_lambda": float(lam.mean()),
        "coverage": float(f.hit[mask].any(-1).float().mean()),
    }
