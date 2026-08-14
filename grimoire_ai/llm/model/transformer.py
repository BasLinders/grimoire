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

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint

from grimoire_ai.llm.model.block import RMSNorm, TransformerBlock
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.embedding import TokenEmbedding

_ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
_FFN_PROJS  = ("gate_proj", "up_proj", "down_proj")


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

        self._gradient_checkpointing = False
        # Initialise before weight tying so named_parameters() only visits the
        # shared tensor once; initialising after would double-init it (once as
        # embedding._embed.weight, once as output_head.weight).
        self._init_weights()

        # Weight tying: share the embedding matrix with the output projection.
        # After this assignment both modules hold a reference to the same
        # nn.Parameter tensor; updating one updates the other automatically.
        self.output_head.weight = self.embedding.weight

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
            if self._gradient_checkpointing and self.training:
                # Recompute block activations during backward instead of
                # storing them.  KV-cache is disabled (past_kv=None,
                # use_cache=False) — caching is an inference-only feature.
                # use_reentrant=False: modern path, compatible with autocast
                # and torch.compile; None tensors (attention_mask) are safe.
                def _block_fn(blk, x_in, mask_in):
                    return blk(x_in, attention_mask=mask_in, past_kv=None, use_cache=False)[0]
                x = torch.utils.checkpoint.checkpoint(
                    _block_fn, block, x, attention_mask,
                    use_reentrant=False,
                )
            else:
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

    def _embed_pooled(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the pooled sentence embedding (gradient-enabled core of ``embed``).

        Runs the same trunk as ``forward`` (token embedding → blocks →
        final RMSNorm) but stops *before* the output projection head, then
        mean-pools the final hidden states across the sequence dimension.
        Unlike ``forward``, this never uses a KV cache and always runs the
        full bidirectional-within-window trunk in a single pass.

        This method carries no ``torch.no_grad``, unlike ``embed()`` — it is
        the entry point for embedding training (e.g. contrastive fine-tuning),
        where gradients must flow back through the pooled vector to the trunk
        weights. ``embed()`` remains the no-grad entry point for inference.

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

    @torch.no_grad()
    def embed(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce a dense sentence embedding for each input sequence.

        The result is the model's own learned representation of the text —
        suitable for semantic similarity search (cosine retrieval) using
        exactly the domain knowledge the model was trained on. Decorated with
        ``torch.no_grad`` because this entry point is for inference only; see
        ``_embed_pooled`` for the gradient-enabled core used during training.

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
        return self._embed_pooled(input_ids, attention_mask)

    @torch.no_grad()
    def merged_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict with LoRA adapters merged into base weights.

        Returns a dict with the same key structure as a plain (non-LoRA)
        GrimoireTransformer, suitable for use as a base checkpoint for a
        subsequent fine-tune run.  Computes merged weights tensor-by-tensor
        without copying the model.  When no adapters are present, equivalent
        to ``state_dict()``.
        """
        from grimoire_ai.llm.model.lora import LoRALinear

        lora_modules = {
            name: mod
            for name, mod in self.named_modules()
            if isinstance(mod, LoRALinear)
        }
        if not lora_modules:
            return self.state_dict()

        lora_internal = frozenset({"base_weight", "base_bias", "lora_A", "lora_B"})
        out: dict[str, torch.Tensor] = {}
        for key, tensor in self.state_dict().items():
            for path, lora_mod in lora_modules.items():
                if key.startswith(f"{path}."):
                    suffix = key[len(path) + 1:]
                    if suffix in lora_internal:
                        if suffix == "base_weight":
                            delta = (lora_mod.lora_B @ lora_mod.lora_A) * lora_mod.scale
                            out[f"{path}.weight"] = (lora_mod.base_weight + delta).clone()
                        elif suffix == "base_bias":
                            out[f"{path}.bias"] = tensor.clone()
                        # lora_A / lora_B are dropped
                        break
            else:
                out[key] = tensor
        return out

    def enable_gradient_checkpointing(self) -> None:
        """Trade activation memory for compute during training.

        When enabled, intermediate activations inside each ``TransformerBlock``
        are discarded after the forward pass and recomputed on demand during
        backward.  This halves peak VRAM at a cost of roughly 20 % extra
        compute (one extra forward pass per backward).

        Has no effect during inference (``model.eval()``).
        """
        self._gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        """Restore the standard forward pass that stores all activations."""
        self._gradient_checkpointing = False

    def add_lora_adapters(
        self,
        rank: int = 8,
        alpha: float = 16.0,
        targets: Optional[Iterable[str]] = None,
    ) -> None:
        """Replace target Linear layers with LoRALinear and freeze base weights.

        After this call only ``lora_A`` and ``lora_B`` parameters have
        ``requires_grad=True``; the Trainer's optimizer automatically picks
        up only those parameters (two 2-D matrices per adapted layer).

        Args:
            rank: LoRA rank r — number of columns in A and rows in B.
                Higher rank = more capacity but more parameters.
                Typical values: 4, 8, 16.
            alpha: Scaling constant; effective scale = alpha / rank.
                Setting alpha == rank keeps the scale at 1.0.
                Default 16.0 gives a scale of 2.0 for rank=8.
            targets: Names of ``nn.Linear`` submodules to wrap with LoRA.
                Valid attention names: ``q_proj``, ``k_proj``, ``v_proj``,
                ``o_proj``.  Valid FFN names: ``gate_proj``, ``up_proj``,
                ``down_proj``.  Defaults to ``("q_proj", "v_proj")`` — the
                two attention projections that give the best quality /
                parameter trade-off for instruction tuning.

        Raises:
            NotImplementedError: If ``config.attention_type != "gqa"``.
                ``MultiHeadLatentAttention`` has no ``q_proj``/``k_proj``/
                ``v_proj`` submodules to wrap — checked up front, before any
                parameter is frozen, so a rejected call leaves the model
                untouched rather than partially mutated.
        """
        if self.config.attention_type != "gqa":
            raise NotImplementedError(
                f"add_lora_adapters() only supports attention_type='gqa'; "
                f"this model uses attention_type={self.config.attention_type!r}. "
                "MultiHeadLatentAttention does not expose q_proj/k_proj/v_proj "
                "targets."
            )

        from grimoire_ai.llm.model.lora import LoRALinear

        target_set = set(targets) if targets is not None else {"q_proj", "v_proj"}

        for param in self.parameters():
            param.requires_grad_(False)

        for block in self.blocks:
            for name in _ATTN_PROJS:
                if name in target_set:
                    setattr(block.attn, name,
                            LoRALinear(getattr(block.attn, name), rank, alpha))
            for name in _FFN_PROJS:
                if name in target_set:
                    setattr(block.ffn, name,
                            LoRALinear(getattr(block.ffn, name), rank, alpha))

    def merge_and_unload(self) -> None:
        """Bake LoRA deltas into base weights and restore plain nn.Linear layers.

        After this call the model is functionally equivalent to one that was
        fully fine-tuned.  All parameters regain ``requires_grad=True``.

        Raises:
            NotImplementedError: If ``config.attention_type != "gqa"`` — see
                ``add_lora_adapters``.
        """
        if self.config.attention_type != "gqa":
            raise NotImplementedError(
                f"merge_and_unload() only supports attention_type='gqa'; "
                f"this model uses attention_type={self.config.attention_type!r}."
            )

        from grimoire_ai.llm.model.lora import LoRALinear

        for block in self.blocks:
            for parent, name in [
                (block.attn, "q_proj"), (block.attn, "k_proj"),
                (block.attn, "v_proj"), (block.attn, "o_proj"),
                (block.ffn,  "gate_proj"), (block.ffn, "up_proj"),
                (block.ffn,  "down_proj"),
            ]:
                mod = getattr(parent, name)
                if isinstance(mod, LoRALinear):
                    setattr(parent, name, mod.merge())

        for param in self.parameters():
            param.requires_grad_(True)

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
