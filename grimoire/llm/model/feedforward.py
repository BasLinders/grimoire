"""SwiGLU feed-forward network for the GrimoireTransformer.

SwiGLU (Swish-Gated Linear Unit) replaces the standard two-layer FFN
with a *gated* variant that multiplies an activation branch against a
linear gate branch.  This gating mechanism lets the network learn which
features to suppress on a per-token basis, which improves expressivity
without adding depth.

Standard FFN (GELU)
-------------------
    FFN(x) = Linear₂(GELU(Linear₁(x)))

    Parameters: 2 × d_model × d_ff

SwiGLU
------
    gate   = Linear_gate(x)           # shape: (*, d_ff)
    up     = Linear_up(x)             # shape: (*, d_ff)
    hidden = SiLU(gate) ⊙ up          # element-wise product
    out    = Linear_down(hidden)       # shape: (*, d_model)

    Parameters: 3 × d_model × d_ff

Because SwiGLU uses three matrices instead of two, the hidden dimension
``d_ff`` must be scaled down to keep the total parameter count equivalent
to a standard ``4 × d_model`` FFN:

    d_ff_swiglu ≈ (2/3) × 4 × d_model = (8/3) × d_model

For ``d_model = 512``: ``(8/3) × 512 ≈ 1365``, rounded up to the nearest
multiple of 64 → ``1408``.  This is the default in ``TransformerConfig``.

SiLU (Sigmoid Linear Unit, also called Swish) is used as the gating
activation: ``SiLU(x) = x × sigmoid(x)``.  It is smooth, non-monotone,
and empirically outperforms ReLU and GELU as the gating function in GLU
variants (Shazeer, 2020; Touvron et al., 2023).

No bias terms are used in any of the three projections, consistent with
the Llama architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from grimoire.llm.model.config import TransformerConfig


class SwiGLUFeedForward(nn.Module):
    """Three-projection gated feed-forward network using SiLU activation.

    Attributes:
        gate_proj: Linear map from ``d_model`` to ``d_ff`` that produces
            the gating signal passed through SiLU.
        up_proj: Linear map from ``d_model`` to ``d_ff`` that produces
            the value signal multiplied by the gate.
        down_proj: Linear map from ``d_ff`` to ``d_model`` that projects
            the gated hidden state back to the residual stream dimension.
        _dropout: Dropout applied to the hidden state before ``down_proj``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise the three projection matrices.

        Args:
            config: Model configuration.  Uses ``d_model``, ``d_ff``,
                and ``dropout``.
        """
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_proj = nn.Linear(config.d_ff, config.d_model, bias=False)
        self._dropout  = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SwiGLU transformation.

        Computes:
            hidden = SiLU(gate_proj(x)) ⊙ up_proj(x)
            out    = down_proj(dropout(hidden))

        The element-wise product ``⊙`` is the gating operation: the SiLU
        output acts as a smooth binary gate that selectively passes or
        suppresses features from ``up_proj(x)``.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Output tensor of shape ``(batch, seq_len, d_model)``.
        """
        gate   = F.silu(self.gate_proj(x))
        up     = self.up_proj(x)
        hidden = self._dropout(gate * up)
        return self.down_proj(hidden)
