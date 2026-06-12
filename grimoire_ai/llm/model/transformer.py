"""Full GrimoireTransformer: the decoder-only language model.

This module wires together all architectural components into a single
``nn.Module``:

    Token embedding
        ↓
    N × TransformerBlock (RMSNorm + GQA + RoPE + SwiGLU)
        ↓
    Final RMSNorm
        ↓
    Output linear head (weight-tied to embedding)
        ↓
    Logits shape: (batch, seq_len, vocab_size)

Weight tying
------------
The input embedding matrix (shape ``vocab_size × d_model``) and the
output projection matrix (same shape) share the same underlying tensor.
This means the model uses the same learned representation when embedding
an input token and when predicting that token as output. Benefits:

- Saves ``vocab_size × d_model × 4`` bytes (≈ 32 MB for our config).
- Empirically improves perplexity for small models by regularising the
  output space to stay consistent with the input space
  (Press & Wolf, 2017, "Using the Output Embedding to Improve Language
  Models").

The tie is implemented by setting ``output_head.weight = embedding.weight``
after construction. Both modules then point to the same ``nn.Parameter``.
"""

from typing import Optional

import torch
import torch.nn as nn

from grimoire_ai.llm.model.block import RMSNorm, TransformerBlock
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.embedding import TokenEmbedding


class GrimoireTransformer(nn.Module):
    """Decoder-only transformer language model with GQA, RoPE, and SwiGLU.

    The model takes a sequence of token ids and returns a logit distribution
    over the vocabulary for each position. During training the target is
    to predict the next token at each position (causal language modelling).
    During inference the logits at the *last* position are sampled to
    generate the next token.

    Attributes:
        config: The ``TransformerConfig`` used to construct this model.
        embedding: ``TokenEmbedding`` module.
        blocks: ``nn.ModuleList`` of ``TransformerBlock`` instances.
        final_norm: ``RMSNorm`` applied after the last block.
        output_head: Linear projection from ``d_model`` to ``vocab_size``.
        Weight-tied to ``embedding.weight``; no bias.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Construct the full transformer from a config.

        Args:
            config: Model hyperparameters. All submodules are built from
            this single object so the model is fully determined by it.
        """
        super().__init__()
        self.config = config

        self.embedding = TokenEmbedding(config)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        # Import RMSNorm from block to avoid re-defining it.
        from grimoire_ai.llm.model.block import RMSNorm
        self.final_norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share the embedding matrix with the output projection.
        # After this assignment both modules hold a reference to the same
        # nn.Parameter tensor; updating one updates the other automatically.
        self.output_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise all weight matrices with small normal values.

        Uses the GPT-2 convention of scaling residual projection weights by
        ``1 / sqrt(2 × n_layers)`` to prevent the residual stream from
        growing in magnitude with depth. All other weights are initialised
        with ``std = 0.02``; biases (where present) are zeroed.
        """
        residual_scale = (2 * self.config.n_layers) ** -0.5
        for name, param in self.named_parameters():
            if param.dim() < 2:
                # 1-D params are either biases (zero them) or RMSNorm scale
                # weights (leave them at their default of ones).
                if "bias" in name:
                    nn.init.zeros_(param)
            elif "o_proj" in name or "down_proj" in name:
                # Residual projections — scale down to prevent depth blow-up.
                nn.init.normal_(param, mean=0.0, std=0.02 * residual_scale)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Compute next-token logits for a batch of token sequences.

        Args:
            input_ids: Integer tensor of shape ``(batch, seq_len)``
                containing token ids.  Values must be in ``[0, vocab_size)``.
            attention_mask: Optional tensor of shape ``(batch, seq_len)``
                with ``1`` for real tokens and ``0`` for padding.
            past_kvs: Per-layer KV cache from the previous generation step,
                as a list of ``(k, v)`` tuples (one per layer).  ``None``
                on the first step and during training.
            use_cache: When ``True``, return ``(logits, present_kvs)`` so
                the sampler can pass the cache to the next step.  Defaults
                to ``False`` — the training path never sets this, so its
                return type and behaviour are unchanged.

        Returns:
            When ``use_cache=False`` (default / training): a float tensor of
            shape ``(batch, seq_len, vocab_size)``.

            When ``use_cache=True`` (inference): a tuple
            ``(logits, present_kvs)`` where ``present_kvs`` is a list of
            ``(k, v)`` tensors, one per layer, ready for the next step.
        """
        x = self.embedding(input_ids)
        present_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for i, block in enumerate(self.blocks):
            layer_past = past_kvs[i] if past_kvs is not None else None
            x, present_kv = block(
                x,
                attention_mask=attention_mask,
                past_kv=layer_past,
                use_cache=use_cache,
            )
            if use_cache and present_kv is not None:
                present_kvs.append(present_kv)

        x = self.final_norm(x)
        logits = self.output_head(x)

        if use_cache:
            return logits, present_kvs
        return logits

    @torch.no_grad()
    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce a dense sentence embedding for each input sequence.

        Runs the same trunk as ``forward`` (token embedding → blocks →
        final RMSNorm) but stops *before* the output projection head, then
        mean-pools the final hidden states across the sequence dimension.
        The result is the model's own learned representation of the text —
        suitable for semantic similarity search (cosine retrieval) using
        exactly the domain knowledge the model was trained on.

        Unlike ``forward``, this method never uses a KV cache and always runs
        the full bidirectional-within-window trunk in a single pass. It is
        decorated with ``torch.no_grad`` because embeddings are only consumed
        at inference time.

        Args:
            input_ids: Integer tensor of shape ``(batch, seq_len)`` with
                token ids in ``[0, vocab_size)``.
            attention_mask: Optional tensor of shape ``(batch, seq_len)`` with
                ``1`` for real tokens and ``0`` for padding. When provided,
                padded positions are excluded from the mean pool so that
                padding never dilutes the embedding.

        Returns:
            Float tensor of shape ``(batch, d_model)`` — one pooled embedding
            per input sequence. Not L2-normalised; callers that want cosine
            similarity should normalise the result.
        """
        x = self.embedding(input_ids)
        for block in self.blocks:
            x, _ = block(
                x,
                attention_mask=attention_mask,
                past_kv=None,
                use_cache=False,
            )
        x = self.final_norm(x)  # (batch, seq_len, d_model)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(x.dtype)  # (batch, seq, 1)
            summed = (x * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            return summed / counts
        return x.mean(dim=1)

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Count the total number of (trainable) parameters.

        The weight-tied output head shares its tensor with the embedding,
        so it is counted only once — PyTorch's ``parameters()`` iterator
        de-duplicates shared tensors automatically.

        Args:
            trainable_only: If ``True`` (default), count only parameters
                with ``requires_grad=True``. Set to ``False`` to include
                frozen parameters as well.

        Returns:
            Total parameter count as an integer.
        """
        params = (
            self.parameters() if not trainable_only
            else (p for p in self.parameters() if p.requires_grad)
        )
        # Use a set of data_ptr values to avoid double-counting tied weights.
        seen: set[int] = set()
        total = 0
        for p in params:
            ptr = p.data_ptr()
            if ptr not in seen:
                seen.add(ptr)
                total += p.numel()
        return total
