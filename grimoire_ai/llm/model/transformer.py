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
from grimoire_ai.llm.model.feedforward import SwiGLUFeedForward

_ATTN_PROJS = ("q_proj", "k_proj", "v_proj", "o_proj")
# MultiHeadLatentAttention's projection names (see mla_attention.py) — a
# different set from GQA's, since MLA has no single q_proj/k_proj/v_proj.
_MLA_ATTN_PROJS = ("w_dkv", "w_uk", "w_uv", "w_qc", "w_qr", "w_kr", "o_proj")
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
        mtp_transforms: ``nn.ModuleList`` of ``config.n_predict`` extra
            ``SwiGLUFeedForward`` modules — one per auxiliary Multi-Token
            Prediction head (see ``forward``'s ``return_mtp_logits``).
            Empty when ``config.n_predict == 0`` (the default).
        mtp_norms: Matching ``RMSNorm`` applied before each MTP head's
            (tied) unembedding, mirroring ``final_norm``'s role for the
            primary output.
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
            [TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        # Import RMSNorm from block to avoid re-defining it.
        from grimoire_ai.llm.model.block import RMSNorm
        self.final_norm = RMSNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Multi-Token Prediction (docs/architecture_optimization.md item #2):
        # config.n_predict extra heads, each a SwiGLUFeedForward + RMSNorm
        # applied to the SAME shared final hidden state as the primary
        # prediction, then unembedded through the same tied output_head.
        # Reusing SwiGLUFeedForward (rather than a bespoke module) keeps this
        # proportionate to the model's existing capacity budget and gives
        # each head genuine extra parameters to learn a distinct
        # further-ahead prediction from — without extra parameters here,
        # every head would just recompute the primary next-token logits.
        # Built only when n_predict > 0, so a default-config model has
        # neither the modules nor their parameters — zero footprint when
        # unused.
        self.mtp_transforms = nn.ModuleList(
            [SwiGLUFeedForward(config) for _ in range(config.n_predict)]
        )
        self.mtp_norms = nn.ModuleList(
            [RMSNorm(config.d_model) for _ in range(config.n_predict)]
        )

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
        neighbor_ids: Optional[torch.Tensor] = None,
        neighbor_mask: Optional[torch.Tensor] = None,
        return_mtp_logits: bool = False,
    ) -> (
        torch.Tensor
        | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]
        | tuple[torch.Tensor, Optional[list[torch.Tensor]]]
        | tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]], Optional[list[torch.Tensor]]]
    ):
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
            neighbor_ids: Optional retrieved-neighbor token ids for Chunked
                Cross-Attention (see ``docs/architecture_optimization.md``
                item #3 and ``chunked_cross_attention.py``), shape
                ``(batch, n_neighbors, neighbor_len)``. Only consulted by
                blocks in ``config.retro_layers``; ignored entirely when
                ``config.retro_layers`` is ``None`` (the default) or when
                this is left ``None`` — every existing caller leaves it
                ``None``, so behaviour is unaffected unless a caller opts
                in explicitly.
            neighbor_mask: Optional padding mask for ``neighbor_ids``, shape
                ``(batch, n_neighbors, neighbor_len)``, ``1`` for real
                tokens. ``None`` when every neighbor chunk is exactly
                ``neighbor_len`` tokens.
            return_mtp_logits: When ``True`` and ``config.n_predict > 0``,
                also compute and return the auxiliary Multi-Token
                Prediction logits — see ``mtp_transforms``. Defaults to
                ``False``, matching every existing caller (the sampler and
                ``InferenceEngine`` never set this); only ``Trainer`` sets
                it, and only when the model was built with ``n_predict>0``.

        Returns:
            When both flags are ``False`` (the default / most callers): a
            float tensor of shape ``(batch, seq_len, vocab_size)`` —
            unchanged from before MTP existed.

            When ``use_cache=True`` and ``return_mtp_logits=False``
            (inference): a tuple ``(logits, present_kvs)``, where ``logits``
            is ``(batch, 1, vocab_size)`` — only the final position's
            distribution, regardless of input ``seq_len`` — since a KV-cache
            forward pass with no MTP request is only ever used to sample the
            next token. When ``return_mtp_logits`` is also ``True``,
            ``logits`` stays full-sequence ``(batch, seq_len, vocab_size)``
            — that combination is reserved for self-speculative decoding,
            which needs every position's primary logits alongside the MTP
            heads'.

            When ``return_mtp_logits=True``: ``mtp_logits`` is appended as
            the last element — ``(logits, mtp_logits)`` or
            ``(logits, present_kvs, mtp_logits)`` depending on
            ``use_cache``. ``mtp_logits`` is a list of ``config.n_predict``
            tensors, each ``(batch, seq_len, vocab_size)``, where element
            ``i`` predicts the token ``i + 2`` positions ahead (element 0
            is the first *extra* head — the primary ``logits`` already
            covers the ``+1`` offset). ``None`` when ``config.n_predict``
            is ``0``.
        """
        x = self.embedding(input_ids)
        present_kvs: list[tuple[torch.Tensor, torch.Tensor]] = []

        # Embed retrieved neighbors ONCE here (not per-block) with the same
        # token embedding table the trunk uses -- see
        # chunked_cross_attention.py's module docstring for why this reuses
        # an existing representation instead of a separate neighbor encoder.
        neighbor_emb: Optional[torch.Tensor] = None
        if neighbor_ids is not None:
            batch, n_neighbors, neighbor_len = neighbor_ids.shape
            neighbor_emb = self.embedding(neighbor_ids.reshape(batch, -1)).reshape(
                batch, n_neighbors, neighbor_len, self.config.d_model
            )

        for i, block in enumerate(self.blocks):
            if self._gradient_checkpointing and self.training:
                # Recompute block activations during backward instead of
                # storing them.  KV-cache is disabled (past_kv=None,
                # use_cache=False) — caching is an inference-only feature.
                # use_reentrant=False: modern path, compatible with autocast
                # and torch.compile; None tensors (attention_mask,
                # neighbor_emb, neighbor_mask) are safe.
                def _block_fn(blk, x_in, mask_in, nbr_emb_in, nbr_mask_in):
                    return blk(
                        x_in, attention_mask=mask_in, past_kv=None, use_cache=False,
                        neighbor_emb=nbr_emb_in, neighbor_mask=nbr_mask_in,
                    )[0]
                x = torch.utils.checkpoint.checkpoint(
                    _block_fn, block, x, attention_mask, neighbor_emb, neighbor_mask,
                    use_reentrant=False,
                )
            else:
                layer_past = past_kvs[i] if past_kvs is not None else None
                x, present_kv = block(
                    x,
                    attention_mask=attention_mask,
                    past_kv=layer_past,
                    use_cache=use_cache,
                    neighbor_emb=neighbor_emb,
                    neighbor_mask=neighbor_mask,
                )
                if use_cache and present_kv is not None:
                    present_kvs.append(present_kv)

        x = self.final_norm(x)
        if use_cache and not return_mtp_logits:
            # Plain-generation inference path (use_cache is never set during
            # training — see the arg docstring above): the sampler only ever
            # wants the next-token distribution for the final position
            # (sampler.py always indexes logits[:, -1, :]), so skip
            # projecting the discarded earlier positions through
            # output_head. Matters most on long, RAG-grounded prefill passes
            # where seq_len can be in the hundreds. Left full-sequence when
            # return_mtp_logits is also set — that combination is reserved
            # for self-speculative decoding (see
            # docs/architecture_optimization.md item #2), which needs every
            # position's primary logits alongside the MTP heads' logits.
            logits = self.output_head(x[:, -1:, :])
        else:
            logits = self.output_head(x)

        mtp_logits: Optional[list[torch.Tensor]] = None
        if return_mtp_logits and self.config.n_predict > 0:
            # Each head transforms the SAME shared trunk output x
            # independently (parallel heads off one representation, not a
            # sequential chain) then unembeds through the tied output_head —
            # gradients from every head flow back into the shared trunk,
            # which is the actual point of the auxiliary objective.
            mtp_logits = [
                self.output_head(self.mtp_norms[i](self.mtp_transforms[i](x)))
                for i in range(self.config.n_predict)
            ]

        if use_cache and return_mtp_logits:
            return logits, present_kvs, mtp_logits
        if use_cache:
            return logits, present_kvs
        if return_mtp_logits:
            return logits, mtp_logits
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
                GQA models (``attention_type="gqa"``): ``q_proj``,
                ``k_proj``, ``v_proj``, ``o_proj``. MLA models
                (``attention_type="mla"``): ``w_dkv``, ``w_uk``, ``w_uv``,
                ``w_qc``, ``w_qr``, ``w_kr``, ``o_proj`` — see
                ``mla_attention.py``. FFN names (``gate_proj``, ``up_proj``,
                ``down_proj``) are valid for either. Defaults to
                ``("q_proj", "v_proj")`` for GQA or ``("w_qc", "w_uv")`` for
                MLA — in both cases the query- and value-generating
                projections, the best quality/parameter trade-off for
                instruction tuning.
        """
        from grimoire_ai.llm.model.lora import LoRALinear

        is_mla = self.config.attention_type == "mla"
        attn_proj_names = _MLA_ATTN_PROJS if is_mla else _ATTN_PROJS
        default_targets = {"w_qc", "w_uv"} if is_mla else {"q_proj", "v_proj"}
        target_set = set(targets) if targets is not None else default_targets

        for param in self.parameters():
            param.requires_grad_(False)

        for block in self.blocks:
            for name in attn_proj_names:
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
        """
        from grimoire_ai.llm.model.lora import LoRALinear

        attn_proj_names = (
            _MLA_ATTN_PROJS if self.config.attention_type == "mla" else _ATTN_PROJS
        )

        for block in self.blocks:
            for parent, name in (
                [(block.attn, n) for n in attn_proj_names]
                + [(block.ffn, n) for n in _FFN_PROJS]
            ):
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
