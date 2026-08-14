"""Tests for Chunked Cross-Attention / RETRO wiring (docs/architecture_optimization.md item #3).

Three layers, mirroring the coverage style used for MLA:
- ``TransformerConfig.retro_layers``: default/validation.
- ``ChunkedCrossAttention``: the standalone attention module in isolation.
- ``TransformerBlock``/``GrimoireTransformer`` wiring: CCA sublayers are
  built only on the configured layers, are a no-op when neighbor_emb isn't
  supplied (even on a CCA-enabled block), actually change output when it
  is, compose correctly with gradient checkpointing, and don't break
  causal masking on the self-attention path.

This module and its wiring do not yet retrieve anything themselves —
neighbor ids are supplied directly by the caller (or the test). Precomputing
real per-window neighbor ids from a corpus and wiring that into Trainer is
tracked as separate follow-up work; see chunked_cross_attention.py's module
docstring.
"""

import pytest
import torch

from grimoire_ai.llm.model.block import TransformerBlock
from grimoire_ai.llm.model.chunked_cross_attention import ChunkedCrossAttention
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------

def test_retro_layers_defaults_to_none() -> None:
    assert TransformerConfig().retro_layers is None


def test_retro_layers_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="retro_layers"):
        TransformerConfig(retro_layers=[])


def test_retro_layers_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="retro_layers"):
        TransformerConfig(n_layers=4, retro_layers=[1, 1])


def test_retro_layers_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="retro_layers"):
        TransformerConfig(n_layers=4, retro_layers=[4])


def test_retro_layers_accepts_valid_indices() -> None:
    cfg = TransformerConfig(n_layers=4, retro_layers=[0, 2])
    assert cfg.retro_layers == [0, 2]


# ---------------------------------------------------------------------------
# ChunkedCrossAttention — standalone module
# ---------------------------------------------------------------------------

def _cca_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=64, max_seq_len=16, dropout=0.0,
    )


def test_cca_output_shape() -> None:
    cfg = _cca_config()
    cca = ChunkedCrossAttention(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    neighbor_emb = torch.randn(2, 3, 8, cfg.d_model)  # 3 neighbors, 8 tokens each
    out = cca(x, neighbor_emb)
    assert out.shape == (2, 6, cfg.d_model)


def test_cca_output_shape_with_mask() -> None:
    cfg = _cca_config()
    cca = ChunkedCrossAttention(cfg)
    x = torch.randn(2, 6, cfg.d_model)
    neighbor_emb = torch.randn(2, 3, 8, cfg.d_model)
    neighbor_mask = torch.ones(2, 3, 8)
    neighbor_mask[:, :, 6:] = 0  # last 2 tokens of every neighbor are padding
    out = cca(x, neighbor_emb, neighbor_mask)
    assert out.shape == (2, 6, cfg.d_model)


def test_cca_fully_masked_neighbor_row_no_nan() -> None:
    """An entirely-padded neighbor set (all-zero mask) must not produce NaN
    (all -inf scores -> 0/0 in softmax without the nan_to_num guard)."""
    cfg = _cca_config()
    cca = ChunkedCrossAttention(cfg)
    x = torch.randn(1, 4, cfg.d_model)
    neighbor_emb = torch.randn(1, 2, 5, cfg.d_model)
    neighbor_mask = torch.zeros(1, 2, 5)
    out = cca(x, neighbor_emb, neighbor_mask)
    assert not torch.any(torch.isnan(out))


def test_cca_gradients_reach_all_projections() -> None:
    cfg = _cca_config()
    cca = ChunkedCrossAttention(cfg)
    cca.train()
    x = torch.randn(2, 6, cfg.d_model, requires_grad=True)
    neighbor_emb = torch.randn(2, 3, 8, cfg.d_model, requires_grad=True)
    out = cca(x, neighbor_emb)
    out.sum().backward()
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        grad = getattr(cca, name).weight.grad
        assert grad is not None and torch.any(grad != 0), f"{name} got no gradient"


def test_cca_position_independence() -> None:
    """Each query position's output must depend only on its own row of x and
    the shared neighbor set -- not on other query positions. Changing x at
    position 0 must not change the output at position 1."""
    cfg = _cca_config()
    cca = ChunkedCrossAttention(cfg)
    cca.eval()
    x_a = torch.randn(1, 4, cfg.d_model)
    x_b = x_a.clone()
    x_b[0, 0] = torch.randn(cfg.d_model)  # only position 0 differs
    neighbor_emb = torch.randn(1, 2, 5, cfg.d_model)

    with torch.no_grad():
        out_a = cca(x_a, neighbor_emb)
        out_b = cca(x_b, neighbor_emb)

    assert torch.allclose(out_a[0, 1:], out_b[0, 1:], atol=1e-6), (
        "CCA output at unrelated positions changed — cross-attention must "
        "be position-wise, not mixing information across query positions."
    )


# ---------------------------------------------------------------------------
# TransformerBlock wiring
# ---------------------------------------------------------------------------

def _block_config(retro_layers=None) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=64, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2,
        d_ff=64, max_seq_len=16, dropout=0.0, retro_layers=retro_layers,
    )


def test_block_has_no_cca_by_default() -> None:
    block = TransformerBlock(_block_config(), layer_idx=0)
    assert block.use_cca is False
    assert not hasattr(block, "cca")


def test_block_has_cca_when_its_layer_idx_is_selected() -> None:
    block = TransformerBlock(_block_config(retro_layers=[0]), layer_idx=0)
    assert block.use_cca is True
    assert hasattr(block, "cca")


def test_block_no_cca_when_a_different_layer_idx_is_selected() -> None:
    block = TransformerBlock(_block_config(retro_layers=[1]), layer_idx=0)
    assert block.use_cca is False
    assert not hasattr(block, "cca")


def test_cca_block_is_noop_without_neighbor_emb() -> None:
    """A CCA-enabled block called without neighbor_emb must produce the
    exact same output as if CCA weren't there at all -- same seed, compare
    a CCA-enabled block to a plain one over identical attn/ffn weights."""
    torch.manual_seed(0)
    plain = TransformerBlock(_block_config(retro_layers=None), layer_idx=0)
    torch.manual_seed(0)
    cca_block = TransformerBlock(_block_config(retro_layers=[0]), layer_idx=0)
    # Same seed reproduces identical attn/ffn weights up to the point CCA's
    # extra parameters are drawn; copy shared submodules explicitly to be
    # exact rather than relying on RNG draw order staying aligned.
    cca_block.attn.load_state_dict(plain.attn.state_dict())
    cca_block.ffn.load_state_dict(plain.ffn.state_dict())
    cca_block.attn_norm.load_state_dict(plain.attn_norm.state_dict())
    cca_block.ffn_norm.load_state_dict(plain.ffn_norm.state_dict())

    x = torch.randn(2, 6, 32)
    plain.eval()
    cca_block.eval()
    with torch.no_grad():
        out_plain, _ = plain(x)
        out_cca_noop, _ = cca_block(x, neighbor_emb=None)

    assert torch.allclose(out_plain, out_cca_noop, atol=1e-6)


def test_cca_block_changes_output_when_neighbor_emb_given() -> None:
    block = TransformerBlock(_block_config(retro_layers=[0]), layer_idx=0)
    block.eval()
    x = torch.randn(2, 6, 32)
    neighbor_emb = torch.randn(2, 2, 4, 32)
    with torch.no_grad():
        out_noop, _ = block(x, neighbor_emb=None)
        out_with_neighbors, _ = block(x, neighbor_emb=neighbor_emb)
    assert not torch.allclose(out_noop, out_with_neighbors)


# ---------------------------------------------------------------------------
# GrimoireTransformer integration
# ---------------------------------------------------------------------------

def test_transformer_ignores_neighbor_ids_when_retro_disabled() -> None:
    """retro_layers=None means no block ever has use_cca=True, so
    neighbor_ids -- even if accidentally passed -- must not affect output."""
    cfg = _block_config(retro_layers=None)
    model = GrimoireTransformer(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    neighbor_ids = torch.randint(0, cfg.vocab_size, (1, 2, 4))
    with torch.no_grad():
        logits_no_neighbors = model(input_ids)
        logits_with_neighbors = model(input_ids, neighbor_ids=neighbor_ids)
    assert torch.allclose(logits_no_neighbors, logits_with_neighbors, atol=1e-6)


def test_transformer_neighbor_ids_change_output_when_retro_enabled() -> None:
    cfg = _block_config(retro_layers=[1])
    model = GrimoireTransformer(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    neighbor_ids = torch.randint(0, cfg.vocab_size, (1, 2, 4))
    with torch.no_grad():
        logits_no_neighbors = model(input_ids)
        logits_with_neighbors = model(input_ids, neighbor_ids=neighbor_ids)
    assert not torch.allclose(logits_no_neighbors, logits_with_neighbors)


def test_transformer_causal_masking_preserved_with_retro_enabled() -> None:
    """CCA cross-attends position-wise to a shared external neighbor set —
    it must not introduce leakage between sequence positions. Same check as
    test_model.py's test_causal_masking, with RETRO active end to end."""
    cfg = _block_config(retro_layers=[0, 1])
    model = GrimoireTransformer(cfg)
    model.eval()
    seq_len = 8
    ids_a = torch.randint(1, cfg.vocab_size, (1, seq_len))
    ids_b = ids_a.clone()
    ids_b[0, 4:] = torch.randint(1, cfg.vocab_size, (4,))
    neighbor_ids = torch.randint(0, cfg.vocab_size, (1, 2, 4))

    with torch.no_grad():
        logits_a = model(ids_a, neighbor_ids=neighbor_ids)
        logits_b = model(ids_b, neighbor_ids=neighbor_ids)

    assert torch.allclose(logits_a[0, :4], logits_b[0, :4], atol=1e-5), (
        "Causal masking violated with RETRO enabled: future tokens affected "
        "earlier positions."
    )


def test_transformer_retro_composes_with_gradient_checkpointing() -> None:
    """Gradients must reach the CCA parameters even when gradient
    checkpointing recomputes block activations during backward."""
    cfg = _block_config(retro_layers=[0])
    model = GrimoireTransformer(cfg)
    model.enable_gradient_checkpointing()
    model.train()
    input_ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    neighbor_ids = torch.randint(0, cfg.vocab_size, (2, 2, 4))

    logits = model(input_ids, neighbor_ids=neighbor_ids)
    logits.sum().backward()

    cca_grad = model.blocks[0].cca.q_proj.weight.grad
    assert cca_grad is not None and torch.any(cca_grad != 0)


def test_transformer_serialisation_with_retro(tmp_path) -> None:
    """Save/load round trip must preserve logits, including CCA weights."""
    cfg = _block_config(retro_layers=[0])
    model = GrimoireTransformer(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    neighbor_ids = torch.randint(0, cfg.vocab_size, (1, 2, 4))
    with torch.no_grad():
        logits_before = model(input_ids, neighbor_ids=neighbor_ids)

    path = tmp_path / "model.pt"
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    loaded_cfg = TransformerConfig.from_dict(checkpoint["config"])
    loaded_model = GrimoireTransformer(loaded_cfg)
    loaded_model.load_state_dict(checkpoint["state_dict"])
    loaded_model.eval()

    with torch.no_grad():
        logits_after = loaded_model(input_ids, neighbor_ids=neighbor_ids)

    assert loaded_cfg.retro_layers == [0]
    assert torch.allclose(logits_before, logits_after, atol=1e-6)
