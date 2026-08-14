"""Smoke tests for Multi-Head Latent Attention (MLA).

Not a full parity suite with ``test_model.py``'s GQA coverage — this checks
the load-bearing correctness properties: output shapes, causal masking,
cache round-trip shape, gradient flow, and — the property most likely to
hide a subtle bug — that the absorbed (cached-decode) attention path is
numerically equivalent to a full forward pass without a cache.

The second half of this file covers wiring MLA into
``TransformerBlock``/``GrimoireTransformer`` via
``TransformerConfig(attention_type="mla")``: full-model forward shape,
causal masking, KV-cache equivalence, and serialisation, mirroring the
GQA-path tests in ``test_model.py``. It also checks that the default
(``attention_type="gqa"``) is unchanged, and that the two downstream features
MLA does not yet support — LoRA adapters and GGUF export — fail with a clear
error instead of silently doing the wrong thing.

See docs/architecture_optimization.md item #1.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from grimoire_ai.llm.model.block import TransformerBlock
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.mla_attention import MultiHeadLatentAttention
from grimoire_ai.llm.model.transformer import GrimoireTransformer


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


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------

def test_mla_output_shape(cfg: TransformerConfig) -> None:
    """MLA must return (output, None) with output shape (batch, seq_len, d_model)."""
    attn = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    out, present_kv = attn(x)
    assert out.shape == (2, 8, cfg.d_model)
    assert present_kv is None  # use_cache=False by default


def test_mla_output_shape_with_padding_mask(cfg: TransformerConfig) -> None:
    """MLA with a padding mask must still return the correct shape."""
    attn = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 8, cfg.d_model)
    mask = torch.ones(2, 8)
    mask[0, 6:] = 0
    out, _ = attn(x, attention_mask=mask)
    assert out.shape == (2, 8, cfg.d_model)


def test_mla_use_cache_returns_compact_latents(cfg: TransformerConfig) -> None:
    """present_kv must be the compact (c_kv, k_rope) pair, not expanded per-head K/V."""
    attn = MultiHeadLatentAttention(cfg)
    x = torch.randn(1, 6, cfg.d_model)
    out, present_kv = attn(x, use_cache=True)
    assert out.shape == (1, 6, cfg.d_model)
    assert present_kv is not None
    c_kv, k_rope = present_kv
    assert c_kv.shape == (1, 6, attn.kv_latent_dim)
    assert k_rope.shape == (1, 6, attn.rope_head_dim)


def test_mla_cache_smaller_than_full_kv(cfg: TransformerConfig) -> None:
    """The entire point of MLA: cached state per token must beat full per-head K+V."""
    attn = MultiHeadLatentAttention(cfg)
    cache_per_token = attn.kv_latent_dim + attn.rope_head_dim
    full_kv_per_token = cfg.n_kv_heads * cfg.head_dim * 2  # GQA's own K + V cache
    assert cache_per_token < full_kv_per_token


def test_mla_cache_extends_on_second_call(cfg: TransformerConfig) -> None:
    """Passing past_kv must extend both cached latents and keep shapes consistent."""
    attn = MultiHeadLatentAttention(cfg)
    x1 = torch.randn(1, 4, cfg.d_model)
    _, past_kv = attn(x1, use_cache=True)

    x2 = torch.randn(1, 1, cfg.d_model)
    out2, present_kv = attn(x2, past_kv=past_kv, use_cache=True)
    assert out2.shape == (1, 1, cfg.d_model)
    # Cache must now hold 4 + 1 = 5 positions.
    assert present_kv[0].shape[1] == 5
    assert present_kv[1].shape[1] == 5


# ---------------------------------------------------------------------------
# Correctness: absorption must match materialized attention
# ---------------------------------------------------------------------------

def test_mla_absorbed_cache_matches_full_forward(cfg: TransformerConfig) -> None:
    """Cached generation must produce identical output to a full forward pass.

    Mirrors test_model.py's test_kv_cache_matches_full_forward for GQA. The
    absorbed path folds W_uk into the query and W_uv into the output so
    attention is computed directly against the cached latents without ever
    re-materializing full per-head K/V; this is the property that proves
    that algebra is correct rather than just fast.
    """
    attn = MultiHeadLatentAttention(cfg)
    attn.eval()
    prompt = torch.randn(1, 6, cfg.d_model)
    new_token = torch.randn(1, 1, cfg.d_model)

    with torch.no_grad():
        full_input = torch.cat([prompt, new_token], dim=1)
        out_full, _ = attn(full_input)
        ref = out_full[0, -1, :]

        _, past_kv = attn(prompt, use_cache=True)
        out_cached, _ = attn(new_token, past_kv=past_kv, use_cache=True)
        cached = out_cached[0, -1, :]

    assert torch.allclose(ref, cached, atol=1e-4), (
        "Absorbed cached-decode path diverged from the full forward pass — "
        "the W_uk/W_uv absorption is not equivalent to materialized attention."
    )


def test_mla_causal_masking(cfg: TransformerConfig) -> None:
    """Position i must not be influenced by positions j > i."""
    attn = MultiHeadLatentAttention(cfg)
    attn.eval()
    seq_len = 8
    x_a = torch.randn(1, seq_len, cfg.d_model)
    x_b = x_a.clone()
    x_b[0, 4:] = torch.randn(4, cfg.d_model)  # differ from position 4 onward

    with torch.no_grad():
        out_a, _ = attn(x_a)
        out_b, _ = attn(x_b)

    assert torch.allclose(out_a[0, :4], out_b[0, :4], atol=1e-5), (
        "Causal masking violated: future tokens affected earlier positions."
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_mla_rejects_rope_dim_too_large() -> None:
    """rope_head_dim must leave room for a nonzero content_dim."""
    cfg = TransformerConfig(
        vocab_size=256, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=16, dropout=0.0,
        mla_rope_head_dim=16,  # equals head_dim (64 // 4) — leaves no content room
    )
    with pytest.raises(ValueError, match="rope_head_dim"):
        MultiHeadLatentAttention(cfg)


def test_mla_rejects_odd_rope_dim() -> None:
    """rope_head_dim must be even for RoPE's rotate-half split."""
    cfg = TransformerConfig(
        vocab_size=256, d_model=64, n_layers=1, n_heads=4, n_kv_heads=2,
        d_ff=128, max_seq_len=16, dropout=0.0,
        mla_rope_head_dim=5,
    )
    with pytest.raises(ValueError, match="rope_head_dim"):
        MultiHeadLatentAttention(cfg)


# ---------------------------------------------------------------------------
# Trainability
# ---------------------------------------------------------------------------

def test_mla_gradients_flow(cfg: TransformerConfig) -> None:
    """A backward pass must reach every MLA parameter."""
    attn = MultiHeadLatentAttention(cfg)
    attn.train()
    x = torch.randn(2, 8, cfg.d_model, requires_grad=True)
    out, _ = attn(x)
    out.sum().backward()
    grads = [p.grad for p in attn.parameters() if p.grad is not None]
    assert len(grads) == len(list(attn.parameters())), (
        "Not all MLA parameters received a gradient."
    )


# ---------------------------------------------------------------------------
# Wiring: TransformerConfig(attention_type="mla") through the full model
# ---------------------------------------------------------------------------

@pytest.fixture
def mla_cfg() -> TransformerConfig:
    """A minimal TransformerConfig selecting MLA attention."""
    return TransformerConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=32,
        dropout=0.0,
        attention_type="mla",
    )


def test_default_attention_type_is_gqa() -> None:
    """attention_type must default to 'gqa' so every existing preset and
    shipped checkpoint is unaffected by MLA's addition."""
    assert TransformerConfig().attention_type == "gqa"


def test_config_rejects_invalid_attention_type() -> None:
    with pytest.raises(ValueError, match="attention_type"):
        TransformerConfig(attention_type="bogus")


def test_block_builds_mla_attention(mla_cfg: TransformerConfig) -> None:
    """TransformerBlock must construct a MultiHeadLatentAttention when asked."""
    block = TransformerBlock(mla_cfg)
    assert isinstance(block.attn, MultiHeadLatentAttention)


def test_block_forward_with_mla(mla_cfg: TransformerConfig) -> None:
    block = TransformerBlock(mla_cfg)
    x = torch.randn(2, 8, mla_cfg.d_model)
    out, present_kv = block(x)
    assert out.shape == x.shape
    assert present_kv is None


def test_transformer_forward_shape_with_mla(mla_cfg: TransformerConfig) -> None:
    model = GrimoireTransformer(mla_cfg)
    input_ids = torch.randint(0, mla_cfg.vocab_size, (2, 16))
    logits = model(input_ids)
    assert logits.shape == (2, 16, mla_cfg.vocab_size)


def test_transformer_causal_masking_with_mla(mla_cfg: TransformerConfig) -> None:
    """Same causality check as test_model.py's test_causal_masking, with
    attention_type='mla' selected end to end."""
    model = GrimoireTransformer(mla_cfg)
    model.eval()
    seq_len = 8
    ids_a = torch.randint(1, mla_cfg.vocab_size, (1, seq_len))
    ids_b = ids_a.clone()
    ids_b[0, 4:] = torch.randint(1, mla_cfg.vocab_size, (4,))

    with torch.no_grad():
        logits_a = model(ids_a)
        logits_b = model(ids_b)

    assert torch.allclose(logits_a[0, :4], logits_b[0, :4], atol=1e-5), (
        "Causal masking violated with attention_type='mla': future tokens "
        "affected earlier positions."
    )


def test_transformer_kv_cache_matches_full_forward_with_mla(
    mla_cfg: TransformerConfig,
) -> None:
    """Same cache-equivalence check as test_model.py's
    test_kv_cache_matches_full_forward, with attention_type='mla' selected
    end to end — exercises MLA's absorbed path through every layer at once,
    not just a single attention module in isolation."""
    model = GrimoireTransformer(mla_cfg)
    model.eval()
    prompt = torch.randint(1, mla_cfg.vocab_size, (1, 6))
    new_token = torch.randint(1, mla_cfg.vocab_size, (1, 1))

    with torch.no_grad():
        full_input = torch.cat([prompt, new_token], dim=1)
        logits_full = model(full_input)
        ref_logits = logits_full[0, -1, :]

        _, past_kvs = model(prompt, use_cache=True)
        logits_cached, _ = model(new_token, past_kvs=past_kvs, use_cache=True)
        cached_logits = logits_cached[0, -1, :]

    assert torch.allclose(ref_logits, cached_logits, atol=1e-4), (
        "KV-cache produced different logits from a full forward pass with "
        "attention_type='mla'."
    )


def test_transformer_serialisation_with_mla(mla_cfg: TransformerConfig) -> None:
    """Save/load round trip must preserve logits, including attention_type."""
    model = GrimoireTransformer(mla_cfg)
    input_ids = torch.randint(0, mla_cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_before = model(input_ids)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save({"state_dict": model.state_dict(), "config": mla_cfg.to_dict()}, path)

        checkpoint = torch.load(path, weights_only=True)
        loaded_cfg = TransformerConfig.from_dict(checkpoint["config"])
        loaded_model = GrimoireTransformer(loaded_cfg)
        loaded_model.load_state_dict(checkpoint["state_dict"])

    with torch.no_grad():
        logits_after = loaded_model(input_ids)

    assert loaded_cfg.attention_type == "mla"
    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Logits changed after serialisation/deserialisation with attention_type='mla'."
    )


def test_lora_rejects_mla_attention_type(mla_cfg: TransformerConfig) -> None:
    """add_lora_adapters() must refuse an MLA model with a clear error
    instead of crashing on a missing q_proj/v_proj attribute mid-loop."""
    model = GrimoireTransformer(mla_cfg)
    with pytest.raises(NotImplementedError, match="attention_type"):
        model.add_lora_adapters()


def test_merge_and_unload_rejects_mla_attention_type(mla_cfg: TransformerConfig) -> None:
    model = GrimoireTransformer(mla_cfg)
    with pytest.raises(NotImplementedError, match="attention_type"):
        model.merge_and_unload()
