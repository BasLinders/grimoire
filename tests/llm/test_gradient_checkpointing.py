"""Tests for gradient checkpointing in GrimoireTransformer and Trainer.

Verifies:
- enable/disable_gradient_checkpointing() toggle the flag correctly.
- Forward pass in training mode with checkpointing produces the same output
  shape as the standard path.
- Gradients flow correctly through a checkpointed forward pass.
- Trainer initialises with gradient_checkpointing=True and sets the flag.
- A short training run completes without error when checkpointing is enabled.
- Checkpointing is inactive during eval (model.eval()) — KV-cache still works.
"""

import tempfile
from pathlib import Path

import torch
import pytest

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.training.trainer import Trainer
from grimoire_ai.llm.training.checkpoint import save_checkpoint
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=256,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


def _tiny_model() -> GrimoireTransformer:
    return GrimoireTransformer(_tiny_config())


def _tiny_dataset(tmp_dir: str, cfg: TransformerConfig) -> TokenizedDataset:
    """Write a tiny corpus.bin and return a TokenizedDataset."""
    import numpy as np
    corpus = np.arange(cfg.vocab_size * 4, dtype=np.int32) % cfg.vocab_size
    path = str(Path(tmp_dir) / "corpus.bin")
    corpus.astype(np.int32).tofile(path)
    return TokenizedDataset(corpus_path=path, seq_len=cfg.max_seq_len)


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------

def test_gradient_checkpointing_disabled_by_default() -> None:
    model = _tiny_model()
    assert model._gradient_checkpointing is False


def test_enable_gradient_checkpointing() -> None:
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    assert model._gradient_checkpointing is True


def test_disable_gradient_checkpointing() -> None:
    model = _tiny_model()
    model.enable_gradient_checkpointing()
    model.disable_gradient_checkpointing()
    assert model._gradient_checkpointing is False


def test_forward_shape_with_checkpointing() -> None:
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    model.enable_gradient_checkpointing()
    model.train()

    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    logits = model(ids)
    assert logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)


def test_gradients_flow_through_checkpointed_forward() -> None:
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    model.enable_gradient_checkpointing()
    model.train()

    ids = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len))
    logits = model(ids)
    loss = logits.mean()
    loss.backward()

    # At least one parameter should have a non-None gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward through checkpointed model"


def test_checkpointing_inactive_during_eval() -> None:
    """In eval mode _gradient_checkpointing has no effect; KV-cache must work."""
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    model.enable_gradient_checkpointing()
    model.eval()

    ids = torch.randint(0, cfg.vocab_size, (1, 4))
    logits, kvs = model(ids, use_cache=True)
    # use_cache=True without return_mtp_logits only returns the final
    # position's logits (see transformer.py's forward() docstring) — the
    # KV cache itself still covers all 4 prefilled positions.
    assert logits.shape == (1, 1, cfg.vocab_size)
    assert len(kvs) == cfg.n_layers


def test_checkpointed_output_matches_standard() -> None:
    """Checkpointing should not change the forward pass output values."""
    cfg = _tiny_config()

    torch.manual_seed(42)
    model_std = GrimoireTransformer(cfg)
    model_std.train()

    torch.manual_seed(42)
    model_ckpt = GrimoireTransformer(cfg)
    model_ckpt.enable_gradient_checkpointing()
    model_ckpt.train()

    # Copy weights so comparison is fair.
    model_ckpt.load_state_dict(model_std.state_dict())

    ids = torch.randint(0, cfg.vocab_size, (2, cfg.max_seq_len))
    with torch.no_grad():
        logits_std  = model_std(ids)
        logits_ckpt = model_ckpt(ids)

    assert torch.allclose(logits_std, logits_ckpt, atol=1e-5), (
        "Checkpointed and standard forward pass produced different outputs"
    )


# ---------------------------------------------------------------------------
# Trainer-level tests
# ---------------------------------------------------------------------------

def test_trainer_sets_checkpointing_flag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config()
        model = GrimoireTransformer(cfg)
        dataset = _tiny_dataset(tmp, cfg)
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=1,
            gradient_checkpointing=True,
            device="cpu",
        )
        assert trainer.gradient_checkpointing is True
        assert trainer.model._gradient_checkpointing is True


def test_trainer_without_checkpointing_default() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config()
        model = GrimoireTransformer(cfg)
        dataset = _tiny_dataset(tmp, cfg)
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=1,
            device="cpu",
        )
        assert trainer.gradient_checkpointing is False
        assert trainer.model._gradient_checkpointing is False


def test_training_run_with_checkpointing() -> None:
    """A short training run must complete without error."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config()
        model = GrimoireTransformer(cfg)
        dataset = _tiny_dataset(tmp, cfg)
        logs: list[str] = []
        Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=3,
            batch_size=2,
            accumulate_steps=1,
            log_every=1,
            save_every=10,
            checkpoint_dir=tmp,
            gradient_checkpointing=True,
            device="cpu",
            on_log=lambda step, loss, lr, elapsed: logs.append(f"{step}:{loss:.4f}"),
        ).train()
        assert len(logs) == 3, f"Expected 3 log entries, got {len(logs)}: {logs}"
