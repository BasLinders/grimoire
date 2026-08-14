"""Configuration dataclass for the GrimoireTransformer.

All architectural hyperparameters live here so that a model can be fully
reconstructed from a single serialised dict — important for checkpointing
and for ensuring that the tokenizer vocabulary size stays in sync with the
embedding table.

Architecture overview
---------------------
The model is a decoder-only transformer using four modern improvements over
the original GPT-2 design:

1. **RMSNorm** instead of LayerNorm — removes the mean-centering step,
   slightly faster and equally stable (Zhang & Sennrich, 2019).
2. **RoPE** (Rotary Position Embedding) instead of learned absolute
   positional embeddings — encodes relative position via rotation, no
   extra parameters, generalises better within the training length
   (Su et al., 2021).
3. **SwiGLU** feed-forward activation instead of GELU — a gated linear
   unit that consistently outperforms GELU at the same parameter budget
   (Shazeer, 2020; used in PaLM and Llama).
4. **Grouped Query Attention (GQA)** instead of standard multi-head
   attention — ``n_kv_heads`` key/value heads are shared across groups
   of query heads, reducing KV-cache memory at inference time without
   measurable quality loss (Ainslie et al., 2023).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TransformerConfig:
    """Hyperparameters that fully specify a GrimoireTransformer.

    Attributes:
        vocab_size: Number of token ids in the BPE vocabulary, including
            the 6 reserved special-token ids.  Must match the trained
            ``BytePairEncoder`` vocabulary exactly.
        d_model: Embedding dimension.  Every token, position, and hidden
            state is represented as a vector of this size.
        n_layers: Number of stacked ``TransformerBlock`` layers.
        n_heads: Number of query attention heads.  Each head operates on
            a slice of ``d_model`` of size ``head_dim = d_model // n_heads``.
        n_kv_heads: Number of key/value heads for Grouped Query Attention.
            Must divide ``n_heads`` evenly.  Setting ``n_kv_heads == n_heads``
            recovers standard multi-head attention; ``n_kv_heads == 1``
            gives multi-query attention.
        d_ff: Hidden dimension of the SwiGLU feed-forward network.  The
            default ``1408`` keeps the parameter count equivalent to a
            standard ``4 × d_model`` FFN because SwiGLU uses three
            projection matrices instead of two
            (``d_ff ≈ int(2/3 × 4 × d_model)`` rounded to a multiple of 64).
        max_seq_len: Maximum number of tokens the model can process in one
            forward pass.  RoPE frequency tables are precomputed up to
            this length.
        dropout: Dropout probability applied after embeddings and within
            each attention and feed-forward sublayer.  Set to 0.0 at
            inference time.
        rope_theta: Base frequency for RoPE sinusoidal curves.  The
            default ``10000.0`` matches the original RoPE paper and Llama.
        mla_kv_latent_dim: Latent dimension for Multi-Head Latent Attention's
            compressed K/V bottleneck (``MultiHeadLatentAttention`` in
            ``grimoire_ai/llm/model/mla_attention.py``).  Only consulted when
            that module is used in place of ``GroupedQueryAttention`` — has
            no effect otherwise.  ``None`` (the default) lets the module pick
            ``2 * head_dim``.
        mla_rope_head_dim: Dimension of MLA's decoupled RoPE key/query
            channel — must be even and smaller than ``head_dim``.  ``None``
            lets the module pick ``head_dim // 2``.
    """

    vocab_size: int = 16384
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1408
    max_seq_len: int = 1024
    dropout: float = 0.1
    rope_theta: float = 10000.0
    mla_kv_latent_dim: Optional[int] = None
    mla_rope_head_dim: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate internal consistency of the configuration.

        Raises:
            ValueError: If ``d_model`` is not divisible by ``n_heads``, or
                if ``n_heads`` is not divisible by ``n_kv_heads``.
        """
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each individual attention head.

        Returns:
            ``d_model // n_heads``.
        """
        return self.d_model // self.n_heads

    @property
    def n_groups(self) -> int:
        """Number of query heads that share each key/value head in GQA.

        Returns:
            ``n_heads // n_kv_heads``.
        """
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> dict:
        """Serialise the config to a plain Python dict.

        Returns:
            A dict with string keys and JSON-serialisable values.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TransformerConfig":
        """Reconstruct a config from a plain dict (e.g. loaded from JSON).

        Args:
            data: A dict as produced by ``to_dict``.

        Returns:
            A fully validated ``TransformerConfig`` instance.
        """
        return cls(**data)

    def save(self, path: str) -> None:
        """Write the config to a JSON file.

        Args:
            path: Destination file path.  Parent directories are created
                if they do not exist.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "TransformerConfig":
        """Read a config from a JSON file written by ``save``.

        Args:
            path: Path to the JSON file.

        Returns:
            A ``TransformerConfig`` instance.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Named size presets
# ---------------------------------------------------------------------------

#: Ready-made configurations for common model sizes.
#: All presets share the same vocab_size, max_seq_len, dropout, and
#: rope_theta defaults; only the architectural dimensions differ.
#:
#: Approximate parameter counts assume vocab_size=16384 with weight tying.
MODEL_PRESETS: dict[str, TransformerConfig] = {
    # ~25M params — fast to train, good for small corpora or quick experiments.
    "small-25M": TransformerConfig(
        d_model=512, n_layers=6, n_heads=8, n_kv_heads=2, d_ff=1408,
    ),
    # ~85M params — meaningful quality jump, still trainable overnight on a
    # consumer GPU.  Recommended once corpus exceeds ~100M tokens.
    "medium-85M": TransformerConfig(
        d_model=768, n_layers=12, n_heads=12, n_kv_heads=3, d_ff=2048,
    ),
    # ~250M params — approaching small open-weight LLM territory.  Requires
    # a substantial corpus (500M+ tokens) and a GPU with ≥8 GB VRAM.
    "large-250M": TransformerConfig(
        d_model=1024, n_layers=20, n_heads=16, n_kv_heads=4, d_ff=2816,
    ),
}
