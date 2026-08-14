"""Transformer block combining attention and feed-forward sublayers.

Each block applies two sublayers in sequence, each wrapped in a
pre-norm + residual pattern:

    x = x + Attention(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

Pre-norm (normalise before the sublayer, add the residual after)
-----------------------------------------------------------------
The original "Attention is All You Need" paper used post-norm:
``LayerNorm(x + sublayer(x))``.  Post-norm is harder to train from
scratch because the residual path passes through LayerNorm, which can
produce large gradient magnitudes early in training.

Pre-norm moves normalisation to the *input* of each sublayer.  The
residual connection bypasses normalisation entirely, giving gradients a
clean path from the output all the way to the embedding layer.  This
makes training from random initialisation significantly more stable and
is used in GPT-2, Llama, and most modern decoder-only models.

RMSNorm
-------
Root Mean Square Normalisation (Zhang & Sennrich, 2019) simplifies
LayerNorm by removing the mean-centering step.  For a vector ``x``:

    RMSNorm(x) = x / RMS(x) × γ

where ``RMS(x) = sqrt(mean(x²) + ε)`` and ``γ`` is a learned
per-dimension scale parameter (no bias term).  This is marginally faster
than LayerNorm and empirically equivalent in quality.
"""

from typing import Optional

import torch
import torch.nn as nn

from grimoire_ai.llm.model.attention import GroupedQueryAttention
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.feedforward import SwiGLUFeedForward
from grimoire_ai.llm.model.mla_attention import MultiHeadLatentAttention


def _build_attention(config: TransformerConfig) -> nn.Module:
    """Construct the attention submodule selected by ``config.attention_type``.

    ``GroupedQueryAttention`` and ``MultiHeadLatentAttention`` both implement
    the same ``forward(x, attention_mask, past_kv, use_cache) -> (output,
    present_kv)`` contract, so ``TransformerBlock`` needs no branching beyond
    this one construction site to support either.

    Args:
        config: Model configuration; ``config.attention_type`` picks the
            module (``"gqa"`` default, or ``"mla"``).

    Returns:
        A ``GroupedQueryAttention`` or ``MultiHeadLatentAttention`` instance.
    """
    if config.attention_type == "mla":
        return MultiHeadLatentAttention(config)
    return GroupedQueryAttention(config)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalisation.

    Normalises along the last dimension using the RMS of the activations,
    then scales by a learned parameter vector ``weight`` of size ``d_model``.
    No bias or mean-centering is applied.

    Attributes:
        eps: Small constant added inside the square root for numerical
            stability.  Defaults to ``1e-6``.
        weight: Learned scale parameter of shape ``(d_model,)``,
            initialised to ones.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initialise RMSNorm.

        Args:
            d_model: Size of the last (feature) dimension to normalise.
            eps: Stability constant added to the RMS denominator.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise ``x`` by its root mean square and rescale.

        Args:
            x: Tensor of any shape; normalisation is applied over the
                last dimension.

        Returns:
            Normalised tensor of the same shape as ``x``.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class TransformerBlock(nn.Module):
    """One decoder transformer block: pre-norm attention + pre-norm FFN.

    Applies the standard residual pattern used in all modern decoder-only
    language models:

        h = x + Attention(RMSNorm(x))
        out = h + FFN(RMSNorm(h))

    Attributes:
        attn_norm: RMSNorm applied to ``x`` before the attention sublayer.
        attn: ``GroupedQueryAttention`` or ``MultiHeadLatentAttention``
            module, selected by ``config.attention_type`` (see
            ``_build_attention``).
        ffn_norm: RMSNorm applied to the post-attention hidden state before
            the feed-forward sublayer.
        ffn: ``SwiGLUFeedForward`` module.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise one transformer block.

        Args:
            config: Model configuration passed through to the attention
                and feed-forward submodules.
        """
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = _build_attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run one transformer block.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional padding mask passed through to attention.
            past_kv: Optional cached ``(k, v)`` from previous steps.
            use_cache: When ``True``, return updated KV tensors.

        Returns:
            A tuple ``(output, present_kv)`` where ``output`` has shape
            ``(batch, seq_len, d_model)`` and ``present_kv`` is the updated
            KV cache when ``use_cache=True``, otherwise ``None``.
        """
        attn_out, present_kv = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            past_kv=past_kv,
            use_cache=use_cache,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present_kv
