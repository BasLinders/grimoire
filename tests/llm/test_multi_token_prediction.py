"""Tests for Multi-Token Prediction (docs/architecture_optimization.md item #2).

Three layers, mirroring the coverage style used for MLA and constrained
decoding:
- ``TransformerConfig``'s ``n_predict`` field: default/validation.
- ``GrimoireTransformer``: MTP heads are built only when requested, are
  opt-in per forward() call even when built, and gradients from the
  auxiliary heads reach both the heads themselves and the shared trunk.
- ``trainer.py``'s ``_mtp_target`` helper (pure token-shifting logic) and a
  full ``Trainer.train()`` run with MTP enabled, proving the auxiliary loss
  is actually wired into a real training step end to end.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.trainer import Trainer, _mtp_target


# ---------------------------------------------------------------------------
# TransformerConfig
# ---------------------------------------------------------------------------

def test_n_predict_defaults_to_zero() -> None:
    assert TransformerConfig().n_predict == 0


def test_n_predict_rejects_negative() -> None:
    with pytest.raises(ValueError, match="n_predict"):
        TransformerConfig(n_predict=-1)


# ---------------------------------------------------------------------------
# GrimoireTransformer — helpers
# ---------------------------------------------------------------------------

def _tiny_config(n_predict: int = 0) -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        n_predict=n_predict,
    )


# ---------------------------------------------------------------------------
# GrimoireTransformer — head construction
# ---------------------------------------------------------------------------

def test_no_mtp_modules_when_n_predict_zero() -> None:
    model = GrimoireTransformer(_tiny_config(n_predict=0))
    assert len(model.mtp_transforms) == 0
    assert len(model.mtp_norms) == 0


def test_mtp_modules_built_when_n_predict_positive() -> None:
    model = GrimoireTransformer(_tiny_config(n_predict=3))
    assert len(model.mtp_transforms) == 3
    assert len(model.mtp_norms) == 3


def test_mtp_heads_add_real_parameters() -> None:
    """MTP heads must have their own learnable parameters — without them,
    every head would just recompute the primary next-token logits from the
    same shared hidden state through the same tied unembedding."""
    base = GrimoireTransformer(_tiny_config(n_predict=0)).num_parameters()
    with_mtp = GrimoireTransformer(_tiny_config(n_predict=2)).num_parameters()
    assert with_mtp > base


# ---------------------------------------------------------------------------
# GrimoireTransformer — forward() opt-in behaviour
# ---------------------------------------------------------------------------

def test_forward_default_unaffected_even_with_mtp_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default forward() (return_mtp_logits=False) must still return a bare
    tensor, even on a model that has MTP heads — opt-in per call, not just
    per model."""
    cfg = _tiny_config(n_predict=2)
    model = GrimoireTransformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 8, cfg.vocab_size)


def test_forward_returns_mtp_logits_when_requested() -> None:
    cfg = _tiny_config(n_predict=2)
    model = GrimoireTransformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, mtp_logits = model(input_ids, return_mtp_logits=True)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert len(mtp_logits) == 2
    for head_logits in mtp_logits:
        assert head_logits.shape == (2, 8, cfg.vocab_size)


def test_forward_mtp_logits_none_when_n_predict_zero() -> None:
    cfg = _tiny_config(n_predict=0)
    model = GrimoireTransformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, mtp_logits = model(input_ids, return_mtp_logits=True)
    assert mtp_logits is None


def test_forward_use_cache_and_mtp_logits_together() -> None:
    cfg = _tiny_config(n_predict=1)
    model = GrimoireTransformer(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, present_kvs, mtp_logits = model(
        input_ids, use_cache=True, return_mtp_logits=True
    )
    assert logits.shape == (1, 4, cfg.vocab_size)
    assert len(present_kvs) == cfg.n_layers
    assert len(mtp_logits) == 1


# ---------------------------------------------------------------------------
# GrimoireTransformer — gradient flow
# ---------------------------------------------------------------------------

def test_mtp_gradients_reach_heads_and_shared_trunk() -> None:
    """Backward through the MTP logits must update both the head's own
    parameters and the shared trunk (embedding/blocks) — that shared-trunk
    signal is the entire point of the auxiliary objective."""
    cfg = _tiny_config(n_predict=1)
    model = GrimoireTransformer(cfg)
    model.train()
    input_ids = torch.randint(1, cfg.vocab_size, (2, 8))

    _, mtp_logits = model(input_ids, return_mtp_logits=True)
    mtp_logits[0].sum().backward()

    head_grad = model.mtp_transforms[0].down_proj.weight.grad
    trunk_grad = model.embedding.weight.grad
    assert head_grad is not None and torch.any(head_grad != 0)
    assert trunk_grad is not None and torch.any(trunk_grad != 0)


# ---------------------------------------------------------------------------
# _mtp_target — pure token-shifting logic
# ---------------------------------------------------------------------------

def test_mtp_target_basic_shift() -> None:
    """full_ids = [0,1,2,3,4,5] (seq_len=5, so input_ids=[0..4], the extra
    real token is 5). offset=2 -> target[t] = full_ids[t+2]."""
    full_ids = torch.arange(6).unsqueeze(0)  # (1, 6)
    target = _mtp_target(full_ids, offset=2, seq_len=5)
    assert target.shape == (1, 5)
    assert target.tolist() == [[2, 3, 4, 5, -100]]


def test_mtp_target_larger_offset_masks_more_tail() -> None:
    full_ids = torch.arange(6).unsqueeze(0)
    target = _mtp_target(full_ids, offset=3, seq_len=5)
    assert target.tolist() == [[3, 4, 5, -100, -100]]


def test_mtp_target_offset_beyond_range_is_fully_masked() -> None:
    full_ids = torch.arange(6).unsqueeze(0)
    target = _mtp_target(full_ids, offset=10, seq_len=5)
    assert target.tolist() == [[-100, -100, -100, -100, -100]]


def test_mtp_target_batched() -> None:
    full_ids = torch.stack([torch.arange(6), torch.arange(10, 16)])  # (2, 6)
    target = _mtp_target(full_ids, offset=2, seq_len=5)
    assert target.tolist() == [[2, 3, 4, 5, -100], [12, 13, 14, 15, -100]]


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

def _tiny_dataset(tmp_dir: str, cfg: TransformerConfig) -> TokenizedDataset:
    corpus = np.arange(cfg.vocab_size * 4, dtype=np.int32) % cfg.vocab_size
    path = str(Path(tmp_dir) / "corpus.bin")
    corpus.astype(np.int32).tofile(path)
    return TokenizedDataset(corpus_path=path, seq_len=cfg.max_seq_len)


def test_trainer_rejects_negative_mtp_loss_weight() -> None:
    with pytest.raises(ValueError, match="mtp_loss_weight"):
        Trainer(
            model=GrimoireTransformer(_tiny_config()),
            train_dataset=torch.utils.data.TensorDataset(torch.zeros(1)),
            mtp_loss_weight=-0.1,
            device="cpu",
        )


def test_training_run_with_mtp_enabled(tmp_path) -> None:
    """A short training run with n_predict > 0 must complete without error,
    produce a finite loss, and persist the MTP head weights in the
    checkpoint (proving they were actually part of the trained model, not
    silently dropped by state_dict/save_checkpoint)."""
    cfg = _tiny_config(n_predict=2)
    model = GrimoireTransformer(cfg)
    dataset = _tiny_dataset(str(tmp_path), cfg)
    logs: list[float] = []

    Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=3,
        batch_size=2,
        accumulate_steps=1,
        log_every=1,
        save_every=3,
        checkpoint_dir=str(tmp_path),
        mtp_loss_weight=0.3,
        device="cpu",
        on_log=lambda step, loss, lr, elapsed: logs.append(loss),
    ).train()

    assert len(logs) == 3
    assert all(torch.isfinite(torch.tensor(loss)) for loss in logs)

    ckpt = torch.load(str(tmp_path / "step_0000003.pt"), map_location="cpu", weights_only=True)
    state_dict = ckpt["model"]
    assert any(key.startswith("mtp_transforms.0.") for key in state_dict)
    assert any(key.startswith("mtp_transforms.1.") for key in state_dict)


def test_training_run_without_mtp_unaffected(tmp_path) -> None:
    """n_predict=0 (default) must train exactly as before — no MTP keys,
    no crash from the new code path being present but unused."""
    cfg = _tiny_config(n_predict=0)
    model = GrimoireTransformer(cfg)
    dataset = _tiny_dataset(str(tmp_path), cfg)
    logs: list[float] = []

    Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=2,
        batch_size=2,
        accumulate_steps=1,
        log_every=1,
        save_every=2,
        checkpoint_dir=str(tmp_path),
        device="cpu",
        on_log=lambda step, loss, lr, elapsed: logs.append(loss),
    ).train()

    assert len(logs) == 2
    ckpt = torch.load(str(tmp_path / "step_0000002.pt"), map_location="cpu", weights_only=True)
    assert not any(key.startswith("mtp_transforms.") for key in ckpt["model"])
