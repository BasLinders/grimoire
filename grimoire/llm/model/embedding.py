"""Token embedding layer for the GrimoireTransformer.

In a RoPE-based model, positional information is injected inside the
attention layer by rotating query and key vectors, not by adding a
positional vector to the token embedding.  This module therefore only
handles token-to-vector lookup; ``grimoire.llm.model.attention`` handles
position encoding.

The embedding weights are later weight-tied to the output projection head
in ``GrimoireTransformer``, so the model uses the same matrix to both
look up input token representations and to project hidden states back into
vocabulary logits.  Weight tying reduces the parameter count by
``vocab_size × d_model`` and empirically improves generalisation for small
language models (Press & Wolf, 2017).
"""

import torch
import torch.nn as nn

from grimoire.llm.model.config import TransformerConfig


class TokenEmbedding(nn.Module):
    """Maps integer token ids to dense embedding vectors.

    A thin wrapper around ``nn.Embedding`` that scales the output by
    ``sqrt(d_model)`` following the convention in "Attention is All You
    Need".  This scaling keeps the embedding magnitudes comparable to the
    sinusoidal (or rotary) position signals added downstream, preventing
    the position signal from being drowned out by large embedding values.

    Attributes:
        d_model: Embedding dimension, copied from the config.
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
        self.d_model = config.d_model
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
        """Look up token embeddings and apply scaling and dropout.

        Args:
            input_ids: Integer tensor of shape ``(batch_size, seq_len)``
                containing token ids in the range ``[0, vocab_size)``.

        Returns:
            Float tensor of shape ``(batch_size, seq_len, d_model)``
            containing scaled, dropout-regularised embedding vectors.
        """
        x = self._embed(input_ids)
        # Scale by sqrt(d_model) to keep embedding magnitudes stable relative
        # to the position encoding signal injected later in the attention layer.
        x = x * (self.d_model ** 0.5)
        return self._dropout(x)
