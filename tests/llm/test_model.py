"""Tests for the Phase 2 model architecture.

Gate criteria (all must pass before moving to Phase 3):
- Instantiate GrimoireTransformer and verify parameter count is in range.
- Forward pass produces the correct output shape.
- Causal masking is enforced: position i cannot attend to position j > i.
- Weight tying: embedding and output_head share the same tensor.
- Config round-trips through to_dict / from_dict without loss.
- Config save/load round-trips through JSON.
- Model serialises and deserialises via torch.save / torch.load.
- RMSNorm normalises correctly.
- RoPE tables have the expected shape and are stable across forward passes.
- SwiGLU output shape is correct.
- GQA output shape is correct with padding mask.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from grimoire.llm.model.attention import GroupedQueryAttention, _apply_rope, _precompute_rope_tables
from grimoire.llm.model.block import RMSNorm, TransformerBlock
from grimoire.llm.model.config import TransformerConfig
from grimoire.llm.model.feedforward import SwiGLUFeedForward
from grimoire.llm.model.transformer import GrimoireTransformer


# ---------------------------------------------------------------------------
# Shared tiny config — keeps tests fast on CPU
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> TransformerConfig:
    """A minimal TransformerConfig suitable for fast CPU tests."""
    return TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
    )


@pytest.fixture
def model(cfg: TransformerConfig) -> GrimoireTransformer:
    """A GrimoireTransformer built from the tiny config."""
    return GrimoireTransformer(cfg)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_default_values() -> None:
    """Default TransformerConfig should match the documented architecture."""
    cfg = TransformerConfig()
    assert cfg.vocab_size == 16384
    assert cfg.d_model == 512
    assert cfg.n_layers == 6
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 2
    assert cfg.d_ff == 1408
    assert cfg.max_seq_len == 1024


def test_config_head_dim(cfg: TransformerConfig) -> None:
    """head_dim property must equal d_model // n_heads."""
    assert cfg.head_dim == cfg.d_model // cfg.n_heads


def test_config_n_groups(cfg: TransformerConfig) -> None:
    """n_groups property must equal n_heads // n_kv_heads."""
    assert cfg.n_groups == cfg.n_heads // cfg.n_kv_heads


def test_config_rejects_mismatched_d_model() -> None:
    """d_model not divisible by n_heads must raise ValueError."""
    with pytest.raises(ValueError, match="d_model"):
        TransformerConfig(d_model=65, n_heads=8)


def test_config_rejects_mismatched_kv_heads() -> None:
    """n_heads not divisible by n_kv_heads must raise ValueError."""
    with pytest.raises(ValueError, match="n_heads"):
        TransformerConfig(n_heads=8, n_kv_heads=3)


def test_config_dict_round_trip(cfg: TransformerConfig) -> None:
    """to_dict / from_dict must reconstruct an identical config."""
    assert TransformerConfig.from_dict(cfg.to_dict()) == cfg


def test_config_json_round_trip(cfg: TransformerConfig) -> None:
    """save / load must round-trip through a JSON file without loss."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "config.json")
        cfg.save(path)
        loaded = TransformerConfig.load(path)
    assert loaded == cfg


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

def test_rmsnorm_output_shape() -> None:
    """RMSNorm must not change the tensor shape."""
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    assert norm(x).shape == x.shape


def test_rmsnorm_unit_rms() -> None:
    """After RMSNorm with unit weights the RMS of each vector should be ~1."""
    norm = RMSNorm(64)
    x = torch.randn(4, 8, 64) * 10.0      # large magnitude to make the test meaningful
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()   # (4, 8)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def test_rope_tables_shape() -> None:
    """Precomputed RoPE tables must have shape (max_seq_len, head_dim // 2)."""
    cos, sin = _precompute_rope_tables(head_dim=64, max_seq_len=32)
    assert cos.shape == (32, 32)
    assert sin.shape == (32, 32)


def test_rope_apply_preserves_shape() -> None:
    """_apply_rope must not change the shape of the input tensor."""
    cos, sin = _precompute_rope_tables(head_dim=16, max_seq_len=8)
    x = torch.randn(2, 4, 8, 16)   # (batch, n_heads, seq_len, head_dim)
    out = _apply_rope(x, cos, sin)
    assert out.shape == x.shape


def test_rope_tables_not_all_zero() -> None:
    """RoPE tables must contain non-trivial values."""
    cos, sin = _precompute_rope_tables(head_dim=64, max_seq_len=32)
    assert not torch.all(cos == 1.0)
    assert not torch.all(sin == 0.0)


# ---------------------------------------------------------------------------
# SwiGLU feed-forward
# ---------------------------------------------------------------------------

def test_swiglu_output_shape(cfg: TransformerConfig) -> None:
    """SwiGLUFeedForward must return a tensor of shape (batch, seq_len, d_model)."""
    ffn = SwiGLUFeedForward(cfg)
    x = torch.randn(2, 10, cfg.d_model)
    assert ffn(x).shape == x.shape


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------

def test_gqa_output_shape(cfg: TransformerConfig) -> None:
    """GQA must return a tensor of shape (batch, seq_len, d_model)."""
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out = attn(x)
    assert out.shape == (2, 8, cfg.d_model)


def test_gqa_output_shape_with_mask(cfg: TransformerConfig) -> None:
    """GQA with a padding mask must still return the correct shape."""
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    mask = torch.ones(2, 8)
    mask[0, 6:] = 0   # last two positions are padding in the first sequence
    out = attn(x, attention_mask=mask)
    assert out.shape == (2, 8, cfg.d_model)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

def test_block_output_shape(cfg: TransformerConfig) -> None:
    """TransformerBlock must return a tensor of shape (batch, seq_len, d_model)."""
    block = TransformerBlock(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    assert block(x).shape == x.shape


# ---------------------------------------------------------------------------
# GrimoireTransformer
# ---------------------------------------------------------------------------

def test_model_parameter_count_in_range(model: GrimoireTransformer) -> None:
    """Model parameter count should be positive and reasonable for the tiny config."""
    n = model.num_parameters()
    assert n > 0
    # Sanity: tiny config should be well under 1M params
    assert n < 1_000_000


def test_full_model_parameter_count() -> None:
    """Default config model should be in the 20–30M parameter range."""
    model = GrimoireTransformer(TransformerConfig())
    n = model.num_parameters()
    assert 15_000_000 < n < 35_000_000, f"Unexpected parameter count: {n:,}"


def test_forward_output_shape(model: GrimoireTransformer, cfg: TransformerConfig) -> None:
    """Forward pass must return logits of shape (batch, seq_len, vocab_size)."""
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_forward_with_attention_mask(model: GrimoireTransformer, cfg: TransformerConfig) -> None:
    """Forward pass with an attention mask must return the correct shape."""
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))
    mask = torch.ones(2, 16)
    mask[1, 12:] = 0
    logits = model(input_ids, attention_mask=mask)
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_causal_masking(model: GrimoireTransformer, cfg: TransformerConfig) -> None:
    """Position i must not influence the logits at position j < i.

    Strategy: run a forward pass with two sequences that are identical
    up to position k and differ after.  The logits at positions 0..k-1
    must be identical; those at k and beyond may differ.
    """
    model.eval()
    seq_len = 8
    ids_a = torch.randint(1, cfg.vocab_size, (1, seq_len))
    ids_b = ids_a.clone()
    ids_b[0, 4:] = torch.randint(1, cfg.vocab_size, (4,))  # differ from position 4

    with torch.no_grad():
        logits_a = model(ids_a)
        logits_b = model(ids_b)

    # Logits at positions 0–3 must be identical (causality).
    assert torch.allclose(logits_a[0, :4], logits_b[0, :4], atol=1e-5), (
        "Causal masking violated: future tokens affected earlier positions."
    )
    # Logits at position 4 and beyond may differ (different input).
    assert not torch.allclose(logits_a[0, 4:], logits_b[0, 4:], atol=1e-5), (
        "Sequences that differ from position 4 produced identical logits everywhere."
    )


def test_weight_tying(model: GrimoireTransformer) -> None:
    """Embedding matrix and output head must share the same tensor (weight tying)."""
    assert model.embedding.weight.data_ptr() == model.output_head.weight.data_ptr(), (
        "Weight tying broken: embedding and output_head use different tensors."
    )


def test_weight_tying_survives_update(model: GrimoireTransformer, cfg: TransformerConfig) -> None:
    """Modifying the embedding weight must be reflected in the output head."""
    model.embedding.weight.data.fill_(1.0)
    assert torch.all(model.output_head.weight == 1.0), (
        "Weight tying broken: output_head did not reflect the embedding update."
    )


def test_model_serialisation(model: GrimoireTransformer, cfg: TransformerConfig) -> None:
    """Model must serialise and deserialise via torch.save / torch.load."""
    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_before = model(input_ids)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, path)

        checkpoint = torch.load(path, weights_only=True)
        loaded_cfg = TransformerConfig.from_dict(checkpoint["config"])
        loaded_model = GrimoireTransformer(loaded_cfg)
        loaded_model.load_state_dict(checkpoint["state_dict"])

    with torch.no_grad():
        logits_after = loaded_model(input_ids)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Logits changed after serialisation/deserialisation."
    )
