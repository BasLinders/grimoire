"""Grouped Query Attention with Rotary Position Embedding (RoPE).

This module implements two ideas from recent efficient-LLM research:

Rotary Position Embedding (RoPE)
---------------------------------
Standard transformers add a positional signal to token embeddings before
the first layer.  RoPE instead encodes position by *rotating* each query
and key vector just before the dot-product is computed, using a rotation
angle that depends on both the token's position and the dimension index.

For a head of dimension ``d``, positions ``m`` and ``n``, and dimension
pair ``(2i, 2i+1)``, the rotation angle is ``m × θ_i`` where:

    θ_i = 1 / (rope_theta ^ (2i / d))

Applying the rotation to Q and K before the dot-product means the
attention score between position ``m`` and position ``n`` depends on
their *relative* offset ``m - n`` rather than their absolute positions.
This generalises better and uses zero extra parameters.

Implementation: represent the rotation via the closed-form formula

    rotate(x, m) = x ⊙ cos(m·θ) + rotate_half(x) ⊙ sin(m·θ)

where ``rotate_half`` swaps the two halves of the vector and negates the
first half: ``[-x₂, x₁]`` for ``x = [x₁, x₂]``.

Grouped Query Attention (GQA)
------------------------------
Standard multi-head attention (MHA) maintains one set of K and V
projections per query head.  For ``n_heads=8`` and ``head_dim=64``, the
KV cache at inference stores 8 × 64 = 512 values per layer per token.

GQA groups the query heads and assigns one shared K/V head per group.
With ``n_kv_heads=2`` and ``n_heads=8``, there are 4 query heads per KV
group, reducing KV cache memory by 4×.  During the forward pass, the K
and V tensors are expanded (repeated) to match the number of query heads
before the dot-product — a cheap operation that does not increase the
number of stored parameters.

Setting ``n_kv_heads == n_heads`` recovers standard MHA.
Setting ``n_kv_heads == 1`` gives Multi-Query Attention (MQA).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from grimoire.llm.model.config import TransformerConfig


def _precompute_rope_tables(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for RoPE.

    For each position ``m`` in ``[0, max_seq_len)`` and each dimension
    pair index ``i`` in ``[0, head_dim // 2)``, compute:

        freq_i  = 1 / (theta ^ (2i / head_dim))
        cos[m, i] = cos(m × freq_i)
        sin[m, i] = sin(m × freq_i)

    These tables are computed once and registered as non-trainable buffers
    on the ``GroupedQueryAttention`` module.

    Args:
        head_dim: Dimension of each attention head.  Must be even.
        max_seq_len: Maximum sequence length to precompute tables for.
        theta: RoPE base frequency.  Defaults to 10000.0.

    Returns:
        A tuple ``(cos_table, sin_table)`` each of shape
        ``(max_seq_len, head_dim // 2)``.
    """
    half = head_dim // 2
    i = torch.arange(0, half, dtype=torch.float32)
    freqs = 1.0 / (theta ** (i / half))               # (half,)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)             # (max_seq_len, half)
    return torch.cos(angles), torch.sin(angles)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate a tensor by swapping and negating its two halves.

    For a vector ``x = [x₁, x₂]`` (each half of size ``head_dim // 2``),
    returns ``[-x₂, x₁]``.  Combined with the cosine/sine tables this
    implements the RoPE rotation formula.

    Args:
        x: Tensor of shape ``(..., head_dim)`` where the last dimension
            is even.

    Returns:
        Tensor of the same shape as ``x``.
    """
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation to a query or key tensor.

    Implements:  ``x_rot = x ⊙ cos + rotate_half(x) ⊙ sin``

    The cos/sin tables cover half the head dimension.  They are repeated
    along the last axis to cover the full head dimension before the
    element-wise multiplication.

    Args:
        x: Tensor of shape ``(batch, n_heads, seq_len, head_dim)``.
        cos: Cosine table of shape ``(seq_len, head_dim // 2)``.
        sin: Sine table of shape ``(seq_len, head_dim // 2)``.

    Returns:
        Rotated tensor of the same shape as ``x``.
    """
    # Expand cos/sin from (seq_len, half) to (1, 1, seq_len, head_dim)
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(0).unsqueeze(0)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


class GroupedQueryAttention(nn.Module):
    """Causal self-attention with GQA and RoPE.

    Attributes:
        n_heads: Total number of query heads.
        n_kv_heads: Number of key/value heads (shared across query groups).
        n_groups: Number of query heads per KV group (``n_heads // n_kv_heads``).
        head_dim: Dimension of each individual head (``d_model // n_heads``).
        q_proj: Linear projection from ``d_model`` to ``n_heads × head_dim``.
            No bias — consistent with Llama and empirically equivalent.
        k_proj: Linear projection from ``d_model`` to ``n_kv_heads × head_dim``.
        v_proj: Linear projection from ``d_model`` to ``n_kv_heads × head_dim``.
        o_proj: Output projection from ``n_heads × head_dim`` to ``d_model``.
        _dropout: Attention-weight dropout.
        _cos: Precomputed RoPE cosine table, shape ``(max_seq_len, head_dim//2)``.
        _sin: Precomputed RoPE sine table, shape ``(max_seq_len, head_dim//2)``.
        _mask: Upper-triangular causal mask of shape ``(max_seq_len, max_seq_len)``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise projections and precompute RoPE and causal mask buffers.

        Args:
            config: Model configuration.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_groups = config.n_groups
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        self._dropout = nn.Dropout(config.dropout)

        cos, sin = _precompute_rope_tables(
            config.head_dim, config.max_seq_len, config.rope_theta
        )
        self.register_buffer("_cos", cos, persistent=True)
        self.register_buffer("_sin", sin, persistent=True)

        # Causal mask: -inf above the diagonal prevents attending to future tokens.
        mask = torch.full(
            (config.max_seq_len, config.max_seq_len), float("-inf")
        ).triu(diagonal=1)
        self.register_buffer("_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute grouped query attention over a sequence.

        Steps:
        1. Project ``x`` into queries, keys, and values.
        2. Reshape into per-head views.
        3. Apply RoPE rotation to queries and keys.
        4. Expand K and V to match the number of query heads (GQA repeat).
        5. Compute scaled dot-product attention with causal mask.
        6. Concatenate heads and project back to ``d_model``.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional boolean or float tensor of shape
                ``(batch, seq_len)`` where ``True`` / ``1`` marks real tokens
                and ``False`` / ``0`` marks padding.  Padding positions are
                set to ``-inf`` before the softmax so they receive zero weight.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        batch, seq_len, _ = x.shape

        # --- Project and reshape ----------------------------------------
        # Q: (batch, n_heads,    seq_len, head_dim)
        # K: (batch, n_kv_heads, seq_len, head_dim)
        # V: (batch, n_kv_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # --- Apply RoPE to Q and K only (not V) -------------------------
        cos = self._cos[:seq_len]
        sin = self._sin[:seq_len]
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # --- Expand K and V for GQA ------------------------------------
        # Repeat each KV head n_groups times so dimensions align with Q.
        # expand + reshape is cheaper than repeat (no memory copy on most backends).
        k = k.unsqueeze(2).expand(batch, self.n_kv_heads, self.n_groups, seq_len, self.head_dim)
        k = k.reshape(batch, self.n_heads, seq_len, self.head_dim)
        v = v.unsqueeze(2).expand(batch, self.n_kv_heads, self.n_groups, seq_len, self.head_dim)
        v = v.reshape(batch, self.n_heads, seq_len, self.head_dim)

        # --- Scaled dot-product attention --------------------------------
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, n_heads, seq_len, seq_len)

        # Apply causal mask (slice to current seq_len × seq_len).
        scores = scores + self._mask[:seq_len, :seq_len]

        # Apply padding mask if provided.
        if attention_mask is not None:
            # attention_mask: (batch, seq_len), 1=real token, 0=padding.
            # Broadcast to (batch, 1, 1, seq_len) and set padding positions
            # to -inf using masked_fill.  Multiplication by -inf is avoided
            # because 0 * -inf = NaN in IEEE 754.
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self._dropout(weights)

        # --- Merge heads and project ------------------------------------
        # (batch, n_heads, seq_len, head_dim) → (batch, seq_len, n_heads × head_dim)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)
