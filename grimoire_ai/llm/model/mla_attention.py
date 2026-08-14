"""Multi-Head Latent Attention (MLA) with decoupled RoPE.

An alternative to ``GroupedQueryAttention`` (see ``attention.py``) that
targets the same problem GQA does — KV-cache memory at inference time — via
a different mechanism: low-rank compression instead of head-sharing.

Motivation
----------
GQA shrinks the KV cache by sharing each K/V head across a *group* of query
heads. MLA instead projects K and V down to a single small shared latent
vector ``c_kv`` (dimension ``kv_latent_dim``, far smaller than
``n_heads * head_dim``) and reconstructs full per-head K/V from it on demand.
Only ``c_kv`` needs to be cached across generation steps, not the expanded
per-head tensors — see DeepSeek-V2 (Liu et al., 2024) and, for the small-model
regime this codebase targets, "Latent Multi-Head Attention for Small Language
Models" (arXiv:2506.09342), which reports a 45% KV-cache reduction at
``kv_latent_dim = d_model / 2`` with only a 0.3% validation-loss increase
relative to standard multi-head attention.

Compare cache size per token:
    GQA (this repo's default, n_kv_heads=2, head_dim=64): 2 * 64 * 2 = 256
    MLA (default here, kv_latent_dim=2*head_dim, rope_head_dim=head_dim//2):
        2*64 + 32 = 160
For baselines more aggressive than GQA (e.g. MQA, n_kv_heads=1) the default
``kv_latent_dim`` here may *not* beat the cache size — override
``TransformerConfig.mla_kv_latent_dim`` down to see a win in that regime.

Decoupled RoPE — why K/V can't just be compressed directly
------------------------------------------------------------
RoPE rotates Q and K by an angle that depends on absolute token position
before the dot product. If the cached state were the *rotated* per-head K,
compression would buy nothing: you'd still need to store a full-size,
position-dependent tensor. MLA works around this by splitting each head's
``head_dim`` into two parts:

- a **content** part (``content_dim = head_dim - rope_head_dim``), derived
  from the compressed latent and *not* rotated — position-agnostic, so the
  latent alone is enough to reconstruct it at any later step;
- a small **decoupled RoPE** part (``rope_head_dim``), computed directly
  from the input (not the latent) and rotated normally. For K, this part is
  a single vector *shared across all heads* rather than one per head,
  keeping it cheap to cache.

The attention score is the sum of the two parts' dot products
(``q_content · k_content + q_rope · k_rope``), which is exactly the ordinary
dot product over the concatenated vector — the split changes nothing about
what the model computes, only how it's stored.

Matrix absorption — how the cache is actually used without re-expanding it
-----------------------------------------------------------------------------
Naively, using a cached latent still requires up-projecting the *entire*
cached history back to full per-head K/V at every generation step — an
O(n) reprojection per step, i.e. O(n^2) total, which would defeat the
purpose of caching. MLA avoids this with the "absorption" trick: since
``k_content = W_uk @ c_kv``, the score
``q_content · k_content = q_content · (W_uk @ c_kv) = (W_uk^T @ q_content) · c_kv``.
Precomputing ``q_absorbed = W_uk^T @ q_content`` once per *new* token lets the
score be computed directly against the raw cached latents — no per-step
re-expansion of history. The same trick applies to the output
(``V`` is never materialized either): ``attn_weights @ V`` is reordered to
``(attn_weights @ c_kv) @ W_uv^T``, aggregating in latent space first.

This module uses the ordinary materialized computation (concatenate
content + rope, run standard scaled-dot-product attention) for the
first/prefill pass (``past_kv is None``), since that pass has no cache to
save and benefits from PyTorch's fused SDPA kernel. It switches to the
absorbed computation only once a cache is being extended
(``past_kv is not None``), where the memory/compute saving actually matters.

LoRA and the absorbed path
---------------------------
``_forward_absorbed`` reads ``w_uk``/``w_uv``'s weight matrices directly
(``self.w_uk.weight``) rather than calling them, since the whole point of
absorption is to fold those matrices into the query/output computation
instead of materializing K/V through a normal forward pass. If
``add_lora_adapters()`` has wrapped either in a ``LoRALinear``, a plain
``.weight`` attribute access no longer exists (``LoRALinear`` stores
``base_weight`` and the ``lora_A``/``lora_B`` low-rank factors separately)
— and even if it did, reading only the frozen base weight would silently
drop the adapter's contribution from every cached-decode step after the
first, while the materialized prefill path (which calls ``self.w_uk(c_kv)``
normally) would correctly include it. That first-token/rest-of-response
inconsistency would be a subtle, hard-to-notice correctness bug rather than
a loud failure. ``_effective_weight`` below resolves the true
(base + LoRA delta) matrix for either module type, so absorbed decoding
stays correct whether or not ``w_uk``/``w_uv`` are LoRA-adapted.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grimoire_ai.llm.model.attention import _apply_rope, _precompute_rope_tables
from grimoire_ai.llm.model.config import TransformerConfig


def _effective_weight(module: nn.Module) -> torch.Tensor:
    """Return *module*'s effective weight matrix, accounting for a LoRA wrap.

    Args:
        module: A plain ``nn.Linear`` or a ``LoRALinear`` (see ``lora.py``).

    Returns:
        The weight matrix ``forward()`` would actually use: ``module.weight``
        for a plain ``nn.Linear``, or ``base_weight + (lora_B @ lora_A) *
        scale`` for a ``LoRALinear`` — the same formula ``LoRALinear.merge()``
        uses, just without constructing a new module.
    """
    from grimoire_ai.llm.model.lora import LoRALinear

    if isinstance(module, LoRALinear):
        return module.base_weight + (module.lora_B @ module.lora_A) * module.scale
    return module.weight


class MultiHeadLatentAttention(nn.Module):
    """Causal self-attention with MLA (low-rank KV compression) and RoPE.

    Drop-in compatible with ``GroupedQueryAttention``'s forward signature —
    same ``(output, present_kv)`` contract — but ``present_kv`` here is
    ``(c_kv, k_rope)``, the compact cacheable latents, rather than expanded
    per-head ``(k, v)`` tensors.

    Attributes:
        n_heads: Total number of query/output heads.
        head_dim: Dimension of each individual head (``d_model // n_heads``).
        kv_latent_dim: Shared K/V compression bottleneck dimension.
        rope_head_dim: Dimension of the decoupled RoPE channel per head.
        content_dim: ``head_dim - rope_head_dim`` — the non-rotated part.
        w_dkv: Down-projection from ``d_model`` to the shared KV latent.
        w_uk: Up-projection from the KV latent to per-head content keys.
        w_uv: Up-projection from the KV latent to per-head values.
        w_qc: Per-head query content projection (no compression — queries
            are never cached, so compressing them saves no memory).
        w_qr: Per-head query RoPE-channel projection.
        w_kr: Shared (not per-head) key RoPE-channel projection.
        o_proj: Output projection from ``n_heads * head_dim`` to ``d_model``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise projections and precompute RoPE and causal mask buffers.

        Args:
            config: Model configuration. ``mla_kv_latent_dim`` and
                ``mla_rope_head_dim`` (both optional) control MLA-specific
                sizing; all other MLA dimensions derive from ``head_dim``.

        Raises:
            ValueError: If the resolved ``rope_head_dim`` is not a positive
                even number smaller than ``head_dim``, or if
                ``kv_latent_dim`` is not positive.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        kv_latent_dim = config.mla_kv_latent_dim
        if kv_latent_dim is None:
            kv_latent_dim = 2 * config.head_dim

        rope_head_dim = config.mla_rope_head_dim
        if rope_head_dim is None:
            rope_head_dim = config.head_dim // 2
            if rope_head_dim % 2 != 0:
                rope_head_dim -= 1  # RoPE's rotate-half needs an even split.

        if kv_latent_dim <= 0:
            raise ValueError(f"mla_kv_latent_dim ({kv_latent_dim}) must be positive.")
        if rope_head_dim <= 0 or rope_head_dim % 2 != 0:
            raise ValueError(
                f"mla_rope_head_dim ({rope_head_dim}) must be a positive even number."
            )
        if rope_head_dim >= config.head_dim:
            raise ValueError(
                f"mla_rope_head_dim ({rope_head_dim}) must be smaller than "
                f"head_dim ({config.head_dim}) to leave room for the content part."
            )

        self.kv_latent_dim = kv_latent_dim
        self.rope_head_dim = rope_head_dim
        self.content_dim = config.head_dim - rope_head_dim

        # --- KV compression: the only thing that gets cached ---------------
        self.w_dkv = nn.Linear(config.d_model, kv_latent_dim, bias=False)
        self.w_uk = nn.Linear(kv_latent_dim, config.n_heads * self.content_dim, bias=False)
        self.w_uv = nn.Linear(kv_latent_dim, config.n_heads * config.head_dim, bias=False)

        # --- Query: plain per-head projection, split content + RoPE --------
        self.w_qc = nn.Linear(config.d_model, config.n_heads * self.content_dim, bias=False)
        self.w_qr = nn.Linear(config.d_model, config.n_heads * rope_head_dim, bias=False)

        # --- Decoupled RoPE key: one small vector shared by all heads ------
        self.w_kr = nn.Linear(config.d_model, rope_head_dim, bias=False)

        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self._dropout = nn.Dropout(config.dropout)

        cos, sin = _precompute_rope_tables(rope_head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("_cos", cos, persistent=True)
        self.register_buffer("_sin", sin, persistent=True)

        mask = torch.full(
            (config.max_seq_len, config.max_seq_len), float("-inf")
        ).triu(diagonal=1)
        self.register_buffer("_mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Compute MLA over a sequence, dispatching by whether a cache exists.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional tensor of shape ``(batch, full_seq_len)``
                where ``1`` marks real tokens and ``0`` marks padding.
            past_kv: Cached ``(c_kv, k_rope)`` latents from previous steps,
                each of shape ``(batch, past_len, dim)`` — no head dimension,
                since both are shared across heads. ``None`` on the first
                (prompt) pass and during training.
            use_cache: When ``True``, return the updated latents as the
                second element of the output tuple.

        Returns:
            A tuple ``(output, present_kv)`` where ``output`` has shape
            ``(batch, seq_len, d_model)`` and ``present_kv`` is
            ``(c_kv, k_rope)`` when ``use_cache=True``, else ``None``.
        """
        if past_kv is None:
            return self._forward_materialized(x, attention_mask, use_cache)
        return self._forward_absorbed(x, attention_mask, past_kv, use_cache)

    def _forward_materialized(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Prefill/training path: expand K/V in full and run standard SDPA.

        No cache exists yet, so there is nothing to save by avoiding
        expansion — this path favours the fused SDPA kernel instead.
        """
        batch, seq_len, _ = x.shape
        H, Dh = self.n_heads, self.head_dim

        c_kv = self.w_dkv(x)  # (b, s, kv_latent_dim)
        k_content = self.w_uk(c_kv).view(batch, seq_len, H, self.content_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(batch, seq_len, H, Dh).transpose(1, 2)

        cos = self._cos[:seq_len]
        sin = self._sin[:seq_len]
        k_rope_rot = _apply_rope(self.w_kr(x).unsqueeze(1), cos, sin).squeeze(1)  # (b, s, rope_dim)

        q_content = self.w_qc(x).view(batch, seq_len, H, self.content_dim).transpose(1, 2)
        q_rope = self.w_qr(x).view(batch, seq_len, H, self.rope_head_dim).transpose(1, 2)
        q_rope = _apply_rope(q_rope, cos, sin)

        k = torch.cat([k_content, k_rope_rot.unsqueeze(1).expand(-1, H, -1, -1)], dim=-1)  # (b,H,s,Dh)
        q = torch.cat([q_content, q_rope], dim=-1)  # (b, H, s, Dh)

        present_kv = (c_kv, k_rope_rot) if use_cache else None

        use_sdpa = hasattr(F, "scaled_dot_product_attention")
        if use_sdpa:
            dropout_p = self._dropout.p if self.training else 0.0
            if attention_mask is not None:
                causal = self._mask[:seq_len, :seq_len]
                attn_bias = (
                    causal.to(q.dtype).unsqueeze(0).unsqueeze(0)
                    .expand(batch, 1, seq_len, seq_len).clone()
                )
                attn_bias = attn_bias.masked_fill(
                    attention_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf")
                )
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
                out = torch.nan_to_num(out, nan=0.0)
            else:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        else:
            scale = math.sqrt(Dh)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = self._mask[:seq_len, :seq_len]
            scores = scores + causal
            if attention_mask is not None:
                pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(pad_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = self._dropout(weights)
            out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, H * Dh)
        return self.o_proj(out), present_kv

    def _forward_absorbed(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_kv: tuple[torch.Tensor, torch.Tensor],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Cached-decode path: never materializes full per-head K or V.

        Folds ``W_uk`` into the query projection and ``W_uv`` into the
        output projection so attention is computed directly against the
        cached latents (see module docstring for the derivation).
        """
        batch, seq_len, _ = x.shape
        H, Dc, Dr, Dh, Dl = (
            self.n_heads, self.content_dim, self.rope_head_dim,
            self.head_dim, self.kv_latent_dim,
        )
        c_kv_past, k_rope_past = past_kv
        past_len = c_kv_past.shape[1]

        c_kv_new = self.w_dkv(x)  # (b, s, Dl)
        cos = self._cos[past_len : past_len + seq_len]
        sin = self._sin[past_len : past_len + seq_len]
        k_rope_new = _apply_rope(self.w_kr(x).unsqueeze(1), cos, sin).squeeze(1)  # (b, s, Dr)

        c_kv_full = torch.cat([c_kv_past, c_kv_new], dim=1)  # (b, full, Dl)
        k_rope_full = torch.cat([k_rope_past, k_rope_new], dim=1)  # (b, full, Dr)
        present_kv = (c_kv_full, k_rope_full) if use_cache else None
        full_len = c_kv_full.shape[1]

        q_content = self.w_qc(x).view(batch, seq_len, H, Dc).transpose(1, 2)  # (b,H,s,Dc)
        q_rope = self.w_qr(x).view(batch, seq_len, H, Dr).transpose(1, 2)
        q_rope = _apply_rope(q_rope, cos, sin)  # (b, H, s, Dr)

        # Absorb W_uk into the query side: q_absorbed = W_uk^T @ q_content,
        # computed once per new token, so scoring never re-expands the cache.
        # _effective_weight (not .weight directly) so a LoRA-wrapped w_uk
        # still contributes correctly — see the module docstring.
        w_uk_per_head = _effective_weight(self.w_uk).view(H, Dc, Dl)  # (H, Dc, Dl)
        q_absorbed = torch.einsum("bhsc,hcl->bhsl", q_content, w_uk_per_head)  # (b,H,s,Dl)

        content_scores = torch.einsum("bhsl,bfl->bhsf", q_absorbed, c_kv_full)  # (b,H,s,full)
        rope_scores = torch.einsum("bhsr,bfr->bhsf", q_rope, k_rope_full)       # (b,H,s,full)
        scores = (content_scores + rope_scores) / math.sqrt(Dh)

        causal = self._mask[past_len : past_len + seq_len, :full_len]
        scores = scores + causal
        if attention_mask is not None:
            pad_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(pad_mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self._dropout(weights)

        # Absorb W_uv into the output side: aggregate in latent space first,
        # then project to head_dim — V is never materialized either.
        agg = torch.einsum("bhsf,bfl->bhsl", weights, c_kv_full)  # (b,H,s,Dl)
        w_uv_per_head = _effective_weight(self.w_uv).view(H, Dh, Dl)  # (H, Dh, Dl)
        out = torch.einsum("bhsl,hdl->bhsd", agg, w_uv_per_head)  # (b,H,s,Dh)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, H * Dh)
        return self.o_proj(out), present_kv
