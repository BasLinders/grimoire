"""Chunked Cross-Attention (CCA) to retrieved neighbor passages.

Item #3 from docs/architecture_optimization.md — RETRO's core mechanism
(Borgeaud et al., 2021), letting the model attend directly to retrieved
corpus passages inside the transformer instead of only via prompt
concatenation (``PromptBuilder``). Retro-li specifically validates this
approach at a scale much closer to Grimoire's than the original
7.5B/trillion-token RETRO — see the source link below.

This module is the standalone attention mechanism only. It does not yet
retrieve anything itself: callers supply already-retrieved neighbor token
ids (or embeddings). The retrieval-database build (chunking, embedding, ANN
index) already exists and needs no new code — see ``SemanticRetriever`` /
``RagIndex`` in ``grimoire_ai/llm/inference/``. Precomputing per-training-
window neighbor ids and wiring them into ``Trainer`` is deliberately left
for a follow-up change; this module and its ``TransformerBlock`` wiring are
independently testable and useful on their own, matching how MLA shipped as
a standalone module before being wired into model construction.

Two deliberate simplifications relative to full RETRO, sized to this
project's scale (25M-250M parameters, not RETRO's 7.5B):

1. **No separate neighbor encoder.** Real RETRO runs each retrieved chunk
   through its own small bidirectional transformer before cross-attending.
   Building and training a second encoder stack is a large addition on its
   own. Here, neighbor chunks are embedded with the SAME token embedding
   table the main trunk already uses (see ``GrimoireTransformer.forward``'s
   ``neighbor_ids`` handling) — reusing an existing, already-trained
   representation rather than adding a new untrained one.
2. **Whole-window attention, not per-chunk windowing.** Real RETRO retrieves
   different neighbors for each chunk WITHIN a sequence and constrains which
   query positions can see which chunk's neighbors, so a chunk near the
   start of a window can't "see" neighbors retrieved for a chunk near the
   end. This module applies one shared set of retrieved neighbors to every
   query position in the window instead. For Grimoire's short training
   windows (max_seq_len on the order of hundreds of tokens, versus RETRO's
   many-thousand-token contexts) a window is close to a single retrieval
   unit anyway, making this a much smaller approximation than it would be
   at RETRO's original scale.

Both are documented scope reductions, not oversights — exact per-chunk
retrieval windowing and a dedicated neighbor encoder are natural follow-ups
if evaluation shows the simplified version leaves quality on the table.

Source: https://arxiv.org/html/2410.00004v2 (Retro-li: Small-Scale Retrieval
Augmented Generation Supporting Noisy Similarity Searches and Domain Shift
Generalization)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from grimoire_ai.llm.model.config import TransformerConfig


class ChunkedCrossAttention(nn.Module):
    """Cross-attention from token hidden states to retrieved neighbor chunks.

    Structurally a standard multi-head cross-attention block: queries come
    from the trunk's hidden states, keys/values come from embedded neighbor
    tokens. No causal mask — see the module docstring's "whole-window
    attention" simplification; every query position may attend to every
    retrieved neighbor token in the window.

    Attributes:
        n_heads: Number of attention heads (shared with the trunk's
            self-attention head count for simplicity).
        head_dim: Dimension of each head (``d_model // n_heads``).
        q_proj: Query projection from the trunk's hidden states.
        k_proj: Key projection from neighbor embeddings.
        v_proj: Value projection from neighbor embeddings.
        o_proj: Output projection back to ``d_model``.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise projections.

        Args:
            config: Model configuration. Uses ``d_model``, ``n_heads``,
                ``head_dim`` (property), and ``dropout``.
        """
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self._dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_emb: torch.Tensor,
        neighbor_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend from ``x`` to embedded neighbor chunks.

        Args:
            x: Query hidden states, shape ``(batch, seq_len, d_model)`` —
                normally the trunk's hidden state after self-attention,
                pre-normalised by the caller (mirroring the pre-norm
                pattern used everywhere else in this model).
            neighbor_emb: Embedded neighbor tokens, shape
                ``(batch, n_neighbors, neighbor_len, d_model)``. Produced by
                embedding retrieved neighbor token ids with the trunk's
                token embedding table — see the module docstring.
            neighbor_mask: Optional padding mask for neighbor tokens, shape
                ``(batch, n_neighbors, neighbor_len)`` with ``1`` for real
                tokens and ``0`` for padding (a neighbor chunk shorter than
                ``neighbor_len``). ``None`` when every neighbor chunk is
                exactly ``neighbor_len`` tokens (no padding needed).

        Returns:
            Cross-attention output, shape ``(batch, seq_len, d_model)``.
            Not residual-added — the caller (``TransformerBlock``) adds it
            to the residual stream, matching how self-attention and the FFN
            sublayer are wired.
        """
        batch, seq_len, _ = x.shape
        _, n_neighbors, neighbor_len, _ = neighbor_emb.shape
        kv_len = n_neighbors * neighbor_len

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        flat_neighbors = neighbor_emb.reshape(batch, kv_len, -1)
        k = self.k_proj(flat_neighbors).view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat_neighbors).view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_bias: Optional[torch.Tensor] = None
        if neighbor_mask is not None:
            flat_mask = neighbor_mask.reshape(batch, kv_len)
            attn_bias = torch.zeros(batch, 1, 1, kv_len, dtype=q.dtype, device=q.device)
            attn_bias = attn_bias.masked_fill(
                flat_mask.view(batch, 1, 1, kv_len) == 0, float("-inf")
            )

        if hasattr(F, "scaled_dot_product_attention"):
            dropout_p = self._dropout.p if self.training else 0.0
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
            if attn_bias is not None:
                # Guard against NaN from an entirely-padded neighbor row
                # (all -inf produces 0/0 in softmax) — same guard used by
                # GroupedQueryAttention's masked path.
                out = torch.nan_to_num(out, nan=0.0)
        else:
            scale = math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if attn_bias is not None:
                scores = scores + attn_bias
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            weights = self._dropout(weights)
            out = torch.matmul(weights, v)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)
