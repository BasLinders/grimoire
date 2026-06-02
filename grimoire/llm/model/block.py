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

import torch
import torch.nn as nn

from grimoire.llm.model.attention import GroupedQueryAttention
from grimoire.llm.model.config import TransformerConfig
from grimoire.llm.model.feedforward import SwiGLUFeedForward


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
        attn: ``GroupedQueryAttention`` module.
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
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one transformer block.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.
            attention_mask: Optional padding mask of shape
                ``(batch, seq_len)`` passed through to the attention module.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x
