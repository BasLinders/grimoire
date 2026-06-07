"""Token embedding layer for the GrimoireTransformer.

In this architecture, positional information is injected inside the attention
layer by RoPE (rotating query and key vectors), not by adding a positional
vector to the token embedding.  This module therefore only performs the
token-id → vector lookup; ``grimoire.llm.model.attention`` handles position.

The embedding weights are weight-tied to the output projection head in
``GrimoireTransformer``, so the model uses the same matrix to look up input
token representations and to project hidden states back into vocabulary
logits.  Weight tying reduces the parameter count by ``vocab_size × d_model``
and empirically improves generalisation for small language models
(Press & Wolf, 2017).
"""

import torch
import torch.nn as nn

from grimoire_ai.llm.model.config import TransformerConfig


class TokenEmbedding(nn.Module):
    """Maps integer token ids to dense embedding vectors.

    A thin wrapper around ``nn.Embedding`` plus dropout.  Unlike the original
    "Attention is All You Need" embedding, the output is deliberately *not*
    scaled by ``sqrt(d_model)``.  That scaling exists to balance token
    embeddings against an *additive* positional signal — but this model uses
    RoPE (applied inside attention) and pre-norm RMSNorm as the first
    operation of every block, so any constant scale on the embeddings is
    immediately normalised away and has no effect on the residual stream.

    Attributes:
        weight: The underlying ``nn.Embedding`` weight tensor of shape
            ``(vocab_size, d_model)``.  Exposed directly so that
            ``GrimoireTransformer`` can tie it to the output head.
        _embed: The ``nn.Embedding`` module.
        _dropout: Dropout applied after embedding lookup.
    """

    def __init__(self, config: TransformerConfig) -> None:
        """Initialise the embedding table.

        Args:
            config: Model configuration.  Uses ``vocab_size``, ``d_model``,
                and ``dropout``.
        """
        super().__init__()
        self._embed = nn.Embedding(config.vocab_size, config.d_model)
        self._dropout = nn.Dropout(config.dropout)

    @property
    def weight(self) -> torch.Tensor:
        """The raw embedding weight matrix of shape ``(vocab_size, d_model)``.

        Exposed for weight tying: ``GrimoireTransformer`` sets
        ``output_head.weight = embedding.weight`` after construction so
        both modules share the same parameter tensor.

        Returns:
            The ``nn.Embedding`` weight tensor.
        """
        return self._embed.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings and apply dropout.

        Args:
            input_ids: Integer tensor of shape ``(batch_size, seq_len)``
                containing token ids in the range ``[0, vocab_size)``.

        Returns:
            Float tensor of shape ``(batch_size, seq_len, d_model)``
            containing the dropout-regularised embedding vectors.
        """
        return self._dropout(self._embed(input_ids))
