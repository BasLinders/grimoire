"""Scoped kNN-LM: re-score next-token predictions against the retrieved passages.

A classic kNN-LM (Khandelwal et al., 2020, "Generalization through
Memorization") keeps a datastore of ``(hidden state at t -> token at t+1)``
pairs, looks up the nearest stored states to the current decoding state, and
interpolates the resulting token distribution with the LM's own:

    p(y) = (1 - λ) · p_lm(y) + λ · p_knn(y)

This module is the *scoped* variant: instead of one global datastore over the
whole corpus, a tiny store is built per query from the passages retrieved for
that query. It nudges generation toward tokens that actually follow similar
contexts in the passages the answer is supposed to be grounded in.

See ``docs/KNN_LM.md`` for the full design, the PR plan, and the go/no-go
check this layer has to pass before it's worth tuning.

Pieces
------
- ``ScopedStore``: torch-backed flat store of keys (``final_norm`` outputs,
  i.e. the vectors the tied ``output_head`` reads) and value token ids. Exact
  squared-L2 search with ``topk`` on the model's own device. A per-query store
  holds a few hundred keys, so no ANN index (and no FAISS dependency) is
  needed, and there's no GPU->CPU round trip per decoding step.
- ``knn_distribution``: softmax over negative distances, aggregated by value
  token, as a ``(B, vocab)`` distribution.
- ``LambdaGate``: maps ``[log1p(d_min), log1p(d_mean), LM entropy]`` to
  ``λ ∈ (0, lam_max)`` and owns the learned kNN softmax temperature.
- ``fuse`` / ``fused_logprobs``: the log-space interpolation.
- ``save_gate`` / ``load_gate``: persistence, tagged with the checkpoint the
  gate was fitted on. A gate is only valid for the hidden-state geometry it
  was fitted to; loading it against a different checkpoint is an error.

Wiring into the sampler and ``InferenceEngine`` is a separate step (PR 2 in
``docs/KNN_LM.md``). Nothing here touches the model or the sampler.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from grimoire_ai.llm.inference.rag_index import _md5

_GATE_FORMAT_VERSION = 1


@dataclass
class KNNConfig:
    """Hyperparameters for kNN lookup and fusion.

    Attributes:
        k: Number of nearest neighbors retrieved per decoding step. Clamped
            to the store size, so a small store simply uses every key.
        lam_max: Hard cap on the kNN interpolation weight λ. Keeps the LM in
            charge of fluency even when the gate is maximally confident in
            the store.
        init_temperature: Starting value of the kNN softmax temperature
            (divides squared L2 distance). Only a placeholder until
            ``LambdaGate.calibrate_temperature`` sets it from real
            distances; squared distances between ``final_norm`` outputs at
            ``d_model=512`` are far larger than any fixed small constant.
        eps: Floor added to ``p_knn`` before taking its log, so tokens the
            store never saw get a large but finite penalty.
    """

    k: int = 16
    lam_max: float = 0.8
    init_temperature: float = 10.0
    eps: float = 1e-10


class ScopedStore:
    """Per-query flat store of ``(hidden state -> next token)`` pairs.

    Keys live on ``device`` as float32. Search is exact squared-L2 via
    ``torch.cdist`` + ``topk``. A scoped store is a few hundred keys at
    most, where exact search on-device is both simplest and fastest.
    """

    def __init__(self, dim: int, device: "str | torch.device" = "cpu") -> None:
        """Create an empty store.

        Args:
            dim: Key dimension (the model's ``d_model``).
            device: Device the keys are kept on. Should match the model's,
                so per-step queries never leave the device.
        """
        self.dim = dim
        self.device = torch.device(device)
        self.keys = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self.values = torch.empty(0, dtype=torch.long, device=self.device)

    @property
    def size(self) -> int:
        """Number of stored ``(key, value)`` pairs."""
        return self.keys.size(0)

    def add(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Append ``(key, value)`` pairs.

        Args:
            keys: ``(N, dim)`` hidden states.
            values: ``(N,)`` token ids, where ``values[i]`` is the token that
                followed the position ``keys[i]`` was taken from.

        Raises:
            ValueError: On a shape mismatch.
        """
        if keys.dim() != 2 or keys.size(1) != self.dim:
            raise ValueError(f"keys must be (N, {self.dim}), got {tuple(keys.shape)}")
        if values.dim() != 1 or values.size(0) != keys.size(0):
            raise ValueError(
                f"values must be ({keys.size(0)},), got {tuple(values.shape)}"
            )
        self.keys = torch.cat([self.keys, keys.detach().to(self.device, torch.float32)])
        self.values = torch.cat([self.values, values.detach().to(self.device, torch.long)])

    def add_transitions(
        self, hidden: torch.Tensor, ids: "torch.Tensor | list[int]", start: int, end: int
    ) -> None:
        """Add every transition *into* the token span ``ids[start:end]``.

        The hidden state at position ``t`` is what the model uses to predict
        ``ids[t + 1]``, so the pairs added are
        ``hidden[t] -> ids[t + 1]`` for ``t in [start - 1, end - 1)``. The
        value tokens are exactly ``ids[start:end]``, and the first key is the
        position just before the span (e.g. the opening ``<SEP>`` of a
        context block), whose job is to predict the span's first token.

        This is the shape the prompt-prefill path needs: one forward pass
        over ``<BOS><SEP>{context}<SEP><USR>...``, then
        ``add_transitions(hidden, ids, ctx_start, ctx_end)``.

        Args:
            hidden: ``(T, dim)`` hidden states for one sequence.
            ids: The ``T`` token ids of that sequence.
            start: First index of the value span. Must be ``>= 1`` (position
                0 has no preceding hidden state).
            end: One past the last index of the value span, ``<= T``.

        Raises:
            ValueError: If the span is out of range.
        """
        ids_t = torch.as_tensor(ids, dtype=torch.long)
        T = hidden.size(0)
        if ids_t.numel() != T:
            raise ValueError(f"hidden has {T} positions but ids has {ids_t.numel()}")
        if not 1 <= start <= end <= T:
            raise ValueError(f"span [{start}, {end}) invalid for a length-{T} sequence")
        if start == end:
            return
        self.add(hidden[start - 1 : end - 1], ids_t[start:end])

    def search(self, queries: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact k-nearest-neighbor search.

        Args:
            queries: ``(B, dim)`` query hidden states. Any device/dtype; they
                are moved to the store's device as float32.
            k: Neighbors per query. Clamped to ``self.size``.

        Returns:
            ``(D, I)``, both ``(B, min(k, size))``: squared L2 distances in
            ascending order, and the matching key indices. Returned on the
            *queries'* device.

        Raises:
            ValueError: If the store is empty.
        """
        if self.size == 0:
            raise ValueError("cannot search an empty ScopedStore")
        k = min(k, self.size)
        q = queries.detach().to(self.device, torch.float32)
        d2 = torch.cdist(q, self.keys).pow(2)
        D, I = d2.topk(k, dim=-1, largest=False)
        return D.to(queries.device), I.to(queries.device)


def knn_distribution(
    D: torch.Tensor,
    I: torch.Tensor,
    values: torch.Tensor,
    vocab_size: int,
    temperature: "torch.Tensor | float",
) -> torch.Tensor:
    """Turn neighbor distances into a next-token distribution.

    Each neighbor gets weight ``softmax(-D / temperature)``; weights are then
    summed per value token, so several neighbors predicting the same token
    reinforce each other.

    Args:
        D: ``(B, k)`` squared distances.
        I: ``(B, k)`` key indices into ``values``. Negative entries mark a
            missing neighbor and get zero weight.
        values: ``(N,)`` token id per stored key.
        vocab_size: Size of the output distribution.
        temperature: Positive scalar (float or 0-d tensor). Gradients flow
            through it when it's a learned parameter.

    Returns:
        ``(B, vocab_size)`` probabilities summing to 1 per row (given at
        least one valid neighbor per row).
    """
    valid = I >= 0
    logits = (-D / temperature).masked_fill(~valid, float("-inf"))
    w = torch.softmax(logits, dim=-1)
    tok = values.to(D.device)[I.clamp(min=0)]
    p = torch.zeros(D.size(0), vocab_size, device=D.device, dtype=w.dtype)
    p.scatter_add_(1, tok, w)
    return p


class LambdaGate(nn.Module):
    """Decides, per decoding step, how much to trust the store.

    Features: ``[log1p(d_min), log1p(d_mean), LM entropy]``. Close neighbors
    and an uncertain LM should push λ up; far neighbors and a confident LM
    should push it down. The mapping is a single linear layer + scaled
    sigmoid, so ``λ ∈ (0, lam_max)``.

    Initialised with zero weights and bias ``-1``, i.e. a constant
    ``λ = lam_max · sigmoid(-1)`` (≈0.215 at ``lam_max=0.8``) until fitted.

    Also owns the kNN softmax temperature, stored as ``log_temp`` so it stays
    positive under gradient updates.
    """

    def __init__(self, lam_max: float = 0.8, init_temperature: float = 10.0) -> None:
        super().__init__()
        if not 0.0 < lam_max < 1.0:
            raise ValueError(f"lam_max must be in (0, 1), got {lam_max}")
        if init_temperature <= 0:
            raise ValueError(f"init_temperature must be > 0, got {init_temperature}")
        self.lin = nn.Linear(3, 1)
        nn.init.zeros_(self.lin.weight)
        nn.init.constant_(self.lin.bias, -1.0)
        self.log_temp = nn.Parameter(torch.tensor(float(init_temperature)).log())
        self.lam_max = lam_max

    @classmethod
    def from_config(cls, cfg: KNNConfig) -> "LambdaGate":
        """Build a gate using ``cfg.lam_max`` and ``cfg.init_temperature``."""
        return cls(lam_max=cfg.lam_max, init_temperature=cfg.init_temperature)

    @property
    def temperature(self) -> torch.Tensor:
        """The (positive) kNN softmax temperature, as a 0-d tensor."""
        return self.log_temp.exp()

    @torch.no_grad()
    def calibrate_temperature(self, D: torch.Tensor) -> float:
        """Set the temperature to the median of observed neighbor distances.

        With ``T`` equal to a typical distance, ``-D / T`` lands around
        ``-1`` instead of ``-hundreds``, so the softmax starts out spreading
        weight across neighbors rather than collapsing to 1-NN. Call once
        on distances from real searches before fitting.

        Args:
            D: Squared distances from ``ScopedStore.search``, any shape.

        Returns:
            The temperature that was set.

        Raises:
            ValueError: If ``D`` has no positive finite entries.
        """
        d = D[torch.isfinite(D) & (D > 0)]
        if d.numel() == 0:
            raise ValueError("no positive finite distances to calibrate from")
        t = float(d.float().median())
        self.log_temp.copy_(torch.tensor(t).log())
        return t

    def forward(self, D: torch.Tensor, lm_logp: torch.Tensor) -> torch.Tensor:
        """Compute λ for each row.

        Args:
            D: ``(B, k)`` ascending squared distances (``D[:, 0]`` nearest).
            lm_logp: ``(B, vocab)`` LM log-probabilities.

        Returns:
            ``(B,)`` interpolation weights in ``(0, lam_max)``.
        """
        lm_logp = lm_logp.float()
        D = D.float()
        ent = -(lm_logp.exp() * lm_logp).nan_to_num(0.0).sum(-1)
        feats = torch.stack([torch.log1p(D[:, 0]), torch.log1p(D.mean(-1)), ent], dim=-1)
        return self.lam_max * torch.sigmoid(self.lin(feats)).squeeze(-1)


def fuse(
    lm_logits: torch.Tensor, p_knn: torch.Tensor, lam: torch.Tensor, eps: float = 1e-10
) -> torch.Tensor:
    """Log-space interpolation ``log((1 - λ) · p_lm + λ · p_knn)``.

    Uses ``logaddexp`` rather than mixing in probability space, so tokens
    with tiny LM probability don't underflow.

    Args:
        lm_logits: ``(B, vocab)`` LM logits or log-probs (re-normalised here
            either way).
        p_knn: ``(B, vocab)`` kNN probabilities.
        lam: ``(B,)`` interpolation weights.
        eps: Floor on ``p_knn`` before the log.

    Returns:
        ``(B, vocab)`` float32 log-probabilities.
    """
    lam = lam.float().clamp(1e-6, 1 - 1e-6)
    lm_lp = lm_logits.float().log_softmax(-1)
    return torch.logaddexp(
        lm_lp + torch.log1p(-lam)[:, None],
        torch.log(p_knn.float() + eps) + torch.log(lam)[:, None],
    )


def fused_logprobs(
    lm_logits: torch.Tensor,
    hidden: torch.Tensor,
    store: Optional[ScopedStore],
    gate: LambdaGate,
    cfg: KNNConfig,
) -> torch.Tensor:
    """Next-token log-probs with kNN fusion, or plain LM log-probs.

    Args:
        lm_logits: ``(B, vocab)`` LM logits for the next position.
        hidden: ``(B, dim)`` ``final_norm`` outputs at that position.
        store: The scoped store, or ``None``.
        gate: Fitted (or freshly initialised) ``LambdaGate``.
        cfg: kNN hyperparameters.

    Returns:
        ``(B, vocab)`` float32 log-probabilities. Identical to
        ``lm_logits.log_softmax(-1)`` when there's no store or it's empty,
        which makes the layer a strict no-op for prompts without a retrieved
        context block.
    """
    lm_lp = lm_logits.float().log_softmax(-1)
    if store is None or store.size == 0:
        return lm_lp
    D, I = store.search(hidden, cfg.k)
    p_knn = knn_distribution(D, I, store.values, lm_lp.size(-1), gate.temperature)
    lam = gate(D, lm_lp)
    return fuse(lm_lp, p_knn, lam, cfg.eps)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def checkpoint_tag(checkpoint_path: "str | Path") -> str:
    """Content fingerprint (MD5) of a model checkpoint.

    Same scheme ``RagIndex`` uses to detect a stale semantic index: a content
    change invalidates the gate, a copy or a touched timestamp does not.
    """
    return _md5(checkpoint_path)


def save_gate(
    path: "str | Path", gate: LambdaGate, cfg: KNNConfig, checkpoint_tag: str
) -> None:
    """Write a fitted gate + its config to ``path``.

    Args:
        path: Output file (conventionally ``<checkpoint>.knn_gate.pt``).
        gate: The fitted gate.
        cfg: The ``KNNConfig`` it was fitted with. Stored so inference uses
            the same ``k`` and ``lam_max``.
        checkpoint_tag: ``checkpoint_tag(ckpt_path)`` of the checkpoint the
            gate was fitted against.
    """
    torch.save(
        {
            "version": _GATE_FORMAT_VERSION,
            "gate_state": gate.state_dict(),
            "config": asdict(cfg),
            "checkpoint_tag": checkpoint_tag,
        },
        path,
    )


def load_gate(
    path: "str | Path", expected_checkpoint_tag: Optional[str] = None
) -> tuple[LambdaGate, KNNConfig, str]:
    """Load a gate written by ``save_gate``.

    Args:
        path: File written by ``save_gate``.
        expected_checkpoint_tag: When given, must match the stored tag.
            Pass ``checkpoint_tag(ckpt_path)`` of the checkpoint you're
            about to run, so a gate fitted to other weights is never used
            silently.

    Returns:
        ``(gate, cfg, stored_checkpoint_tag)``, gate in eval mode.

    Raises:
        ValueError: On an unknown format version or a checkpoint-tag
            mismatch.
    """
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob.get("version") != _GATE_FORMAT_VERSION:
        raise ValueError(f"unsupported kNN gate format version: {blob.get('version')!r}")
    stored_tag = blob["checkpoint_tag"]
    if expected_checkpoint_tag is not None and stored_tag != expected_checkpoint_tag:
        raise ValueError(
            "kNN gate was fitted on a different checkpoint "
            f"(gate: {stored_tag}, current: {expected_checkpoint_tag}). "
            "Re-fit it: the gate is only valid for the hidden-state geometry "
            "it was fitted to."
        )
    cfg = KNNConfig(**blob["config"])
    gate = LambdaGate.from_config(cfg)
    gate.load_state_dict(blob["gate_state"])
    gate.eval()
    return gate, cfg, stored_tag
