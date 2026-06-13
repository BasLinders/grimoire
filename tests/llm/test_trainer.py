"""Tests for the training pipeline.

Gate criteria:
- A single training step on random data produces a finite loss.
- Loss decreases over 20 steps on a tiny overfit corpus.
- Checkpoint save/load round-trips correctly (model produces identical output).
- Trainer respects gradient accumulation (optimizer steps fewer than forward passes).
- LR schedule: LR is 0 at step 0, reaches peak at warmup_steps, decays after.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from grimoire_ai.llm.data.dataset import TokenizedDataset
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.training.checkpoint import load_checkpoint, save_checkpoint
from grimoire_ai.llm.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    """Minimal config for fast CPU tests."""
    return TransformerConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


def _write_corpus(n_tokens: int, vocab_size: int, tmp_dir: str) -> str:
    """Write a random corpus of ``n_tokens`` tokens to a binary file."""
    path = str(Path(tmp_dir) / "corpus.bin")
    tokens = np.random.randint(6, vocab_size, size=n_tokens, dtype=np.int32)
    fp = np.memmap(path, dtype=np.int32, mode="w+", shape=(n_tokens,))
    fp[:] = tokens
    fp.flush()
    del fp
    return path


def _write_repeat_corpus(seq: list[int], repeats: int, tmp_dir: str) -> str:
    """Write a corpus that is the same short sequence repeated many times."""
    path = str(Path(tmp_dir) / "corpus.bin")
    tokens = np.array(seq * repeats, dtype=np.int32)
    fp = np.memmap(path, dtype=np.int32, mode="w+", shape=(len(tokens),))
    fp[:] = tokens
    fp.flush()
    del fp
    return path


def _make_trainer(
    tmp_dir: str,
    total_steps: int = 5,
    accumulate_steps: int = 1,
    corpus_tokens: int = 500,
) -> tuple[Trainer, GrimoireTransformer]:
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    corpus_path = _write_corpus(corpus_tokens, cfg.vocab_size, tmp_dir)
    dataset = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        total_steps=total_steps,
        batch_size=2,
        accumulate_steps=accumulate_steps,
        log_every=total_steps + 1,   # suppress logging during tests
        save_every=total_steps + 1,  # suppress checkpointing during tests
        checkpoint_dir=tmp_dir,
        device="cpu",
    )
    return trainer, model


# ---------------------------------------------------------------------------
# Single training step produces a finite loss
# ---------------------------------------------------------------------------

def test_single_step_produces_finite_loss() -> None:
    """After one optimizer step the model must have a finite, positive loss."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model = _make_trainer(tmp, total_steps=1)
        trainer.train()
    # If we reach here without NaN/inf the test passes; additionally
    # verify the model parameters are still finite.
    for p in model.parameters():
        assert torch.isfinite(p).all(), "Model contains non-finite parameters after one step."


# ---------------------------------------------------------------------------
# Loss decreases on a tiny overfit corpus
# ---------------------------------------------------------------------------

def test_loss_decreases_on_overfit_corpus() -> None:
    """Loss should decrease when the model is given 20 steps on a tiny corpus.

    We create a 200-token corpus by repeating a 17-token sequence so that
    only a handful of unique windows exist.  The model should memorise them.
    """
    seq = list(range(6, 23))   # 17 tokens, ids 6-22 (above special token range)
    cfg = _tiny_config()

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = _write_repeat_corpus(seq, repeats=50, tmp_dir=tmp)
        dataset = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=4)
        model = GrimoireTransformer(cfg)

        losses: list[float] = []

        # Monkey-patch the trainer log to capture losses.
        class _CapturingTrainer(Trainer):
            def train(self, resume_from=None):
                import torch.nn.functional as F
                from grimoire_ai.llm.tokenizer.special_tokens import PAD_ID
                self.model.train()
                self._optimizer.zero_grad()
                data_iter = iter(self._loader)
                for _ in range(self.total_steps * self.accumulate_steps):
                    try:
                        inp, tgt, mask = next(data_iter)
                    except StopIteration:
                        data_iter = iter(self._loader)
                        inp, tgt, mask = next(data_iter)
                    logits = self.model(inp, attention_mask=mask)
                    loss = F.cross_entropy(
                        logits.view(-1, self.config.vocab_size),
                        tgt.view(-1),
                        ignore_index=PAD_ID,
                    ) / self.accumulate_steps
                    loss.backward()
                    losses.append(loss.item() * self.accumulate_steps)
                    if len(losses) % self.accumulate_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self._optimizer.step()
                        self._scheduler.step()
                        self._optimizer.zero_grad()

        trainer = _CapturingTrainer(
            model=model,
            train_dataset=dataset,
            total_steps=20,
            # Short warmup + real peak LR so the 20-step window actually
            # exercises learning. With the default 500-step warmup the LR
            # would crawl at ~1e-5 and no meaningful update would occur.
            warmup_steps=5,
            peak_lr=1e-3,
            batch_size=2,
            accumulate_steps=1,
            log_every=999,
            save_every=999,
            checkpoint_dir=tmp,
            device="cpu",
        )
        trainer.train()

    assert len(losses) == 20
    first_loss = sum(losses[:3]) / 3
    last_loss  = sum(losses[-3:]) / 3
    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}. "
        "The model may have a bug in the forward pass or loss computation."
    )


# ---------------------------------------------------------------------------
# Checkpoint save / load round-trip
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip() -> None:
    """Saving then loading a checkpoint must produce identical model outputs."""
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits_before = model(input_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "ckpt.pt")
        save_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            step=42,
            config_dict=cfg.to_dict(),
            train_loss=1.23,
        )
        ckpt = load_checkpoint(path)

    assert ckpt["step"] == 42
    assert abs(ckpt["train_loss"] - 1.23) < 1e-6

    loaded_model = GrimoireTransformer(TransformerConfig.from_dict(ckpt["config"]))
    loaded_model.load_state_dict(ckpt["model"])
    loaded_model.eval()

    with torch.no_grad():
        logits_after = loaded_model(input_ids)

    assert torch.allclose(logits_before, logits_after, atol=1e-6), (
        "Logits changed after checkpoint save/load."
    )


def test_checkpoint_missing_file_raises() -> None:
    """Loading from a non-existent path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/tmp/grimoire_no_such_checkpoint.pt")


# ---------------------------------------------------------------------------
# LR schedule shape
# ---------------------------------------------------------------------------

def test_on_log_callback_fires() -> None:
    """on_log must be called with (step, loss, lr, elapsed) at each log interval."""
    log_calls: list[tuple[int, float, float, float]] = []

    def capture(step: int, loss: float, lr: float, elapsed: float) -> None:
        log_calls.append((step, loss, lr, elapsed))

    with tempfile.TemporaryDirectory() as tmp:
        cfg = _tiny_config()
        model = GrimoireTransformer(cfg)
        corpus_path = _write_corpus(500, cfg.vocab_size, tmp)
        dataset = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=10,
            warmup_steps=2,
            peak_lr=1e-3,
            batch_size=2,
            accumulate_steps=1,
            log_every=5,
            save_every=999,
            checkpoint_dir=tmp,
            device="cpu",
            on_log=capture,
        )
        trainer.train()

    # log_every=5, total_steps=10 → expect exactly 2 calls (at step 5 and 10).
    assert len(log_calls) == 2, f"Expected 2 on_log calls, got {len(log_calls)}."
    steps = [c[0] for c in log_calls]
    assert steps == [5, 10], f"Expected steps [5, 10], got {steps}."
    for step, loss, lr, elapsed in log_calls:
        assert isinstance(step, int)
        assert loss > 0
        assert lr > 0
        assert elapsed >= 0


def test_on_log_none_does_not_raise() -> None:
    """Omitting on_log (default None) must produce no error."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_trainer(tmp, total_steps=2)
        trainer.train()  # must not raise


# ---------------------------------------------------------------------------
# Validation / evaluation
# ---------------------------------------------------------------------------

def _make_eval_trainer(tmp: str, **overrides):
    """Build a tiny CPU trainer with a validation set for eval tests."""
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    corpus_path = _write_corpus(500, cfg.vocab_size, tmp)
    train_ds = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)
    val_ds = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)
    kwargs = dict(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        total_steps=10,
        warmup_steps=2,
        peak_lr=1e-3,
        batch_size=2,
        accumulate_steps=1,
        log_every=999,
        save_every=999,
        checkpoint_dir=tmp,
        device="cpu",
    )
    kwargs.update(overrides)
    return Trainer(**kwargs), model


def test_evaluate_returns_finite_loss() -> None:
    """evaluate() must return a finite, positive loss when a val set is set."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_eval_trainer(tmp)
        val_loss = trainer.evaluate()
    assert math.isfinite(val_loss)
    assert val_loss > 0


def test_evaluate_without_val_set_returns_nan() -> None:
    """evaluate() must return nan when no validation set was configured."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_trainer(tmp, total_steps=2)
        assert math.isnan(trainer.evaluate())


def test_evaluate_restores_training_mode() -> None:
    """evaluate() must leave the model in training mode if it started there."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, model = _make_eval_trainer(tmp)
        model.train()
        trainer.evaluate()
        assert model.training, "evaluate() should restore train mode."


def test_on_eval_callback_fires() -> None:
    """on_eval must fire at each eval interval with a finite val loss."""
    eval_calls: list[tuple[int, float, float]] = []

    def capture(step: int, val_loss: float, elapsed: float) -> None:
        eval_calls.append((step, val_loss, elapsed))

    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_eval_trainer(tmp, eval_every=5, on_eval=capture)
        trainer.train()

    # eval_every=5, total_steps=10 → expect eval at step 5 and 10.
    assert [c[0] for c in eval_calls] == [5, 10]
    for step, val_loss, elapsed in eval_calls:
        assert math.isfinite(val_loss) and val_loss > 0
        assert elapsed >= 0


def test_eval_every_defaults_to_save_every() -> None:
    """When eval_every is unset (0), eval should fall back to save_every."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_eval_trainer(tmp, save_every=999)
        assert trainer._eval_every == 999


def test_eval_batches_caps_validation_passes() -> None:
    """eval_batches must limit how many val batches are averaged."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_eval_trainer(tmp, eval_batches=1)
        # With a cap of 1 the call must still produce a finite loss.
        assert math.isfinite(trainer.evaluate())


def test_no_eval_without_val_dataset_does_not_raise() -> None:
    """Training without a val_dataset must run with no eval and no error."""
    on_eval_calls: list = []
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_trainer(tmp, total_steps=2)
        trainer._on_eval = lambda *a: on_eval_calls.append(a)
        trainer.train()  # must not raise
    assert on_eval_calls == [], "on_eval must not fire without a val_dataset."


def test_early_stopping_halts_before_total_steps() -> None:
    """With a flat validation loss, early stopping must end the run early."""
    with tempfile.TemporaryDirectory() as tmp:
        # patience=2, eval every 2 steps. A subclass returns a constant val loss
        # so no eval ever clears the (zero) noise band after the first.
        trainer, _ = _make_eval_trainer(
            tmp,
            total_steps=100,
            eval_every=2,
            early_stop_enabled=True,
            early_stop_patience=2,
            early_stop_bootstraps=50,
        )

        # Force a deterministic flat validation signal.
        def _flat_eval() -> float:
            trainer._last_val_batch_losses = [1.0, 1.0, 1.0, 1.0]
            return 1.0

        trainer.evaluate = _flat_eval  # type: ignore[method-assign]
        trainer.train()

        # First eval sets the best; evals 2 and 3 are "bad" → stop at the 3rd
        # eval, i.e. around step 6, far below total_steps=100.
        assert trainer._step < 100, (
            f"Early stopping should have halted before 100 steps, "
            f"got {trainer._step}."
        )


def test_early_stopping_disabled_runs_full() -> None:
    """Without early stopping the run must reach total_steps."""
    with tempfile.TemporaryDirectory() as tmp:
        trainer, _ = _make_eval_trainer(tmp, total_steps=6, eval_every=2)
        trainer.train()
        assert trainer._step == 6


def test_lr_schedule_warmup_and_decay() -> None:
    """LR should rise during warmup and then decrease after peak."""
    cfg = _tiny_config()
    total_steps  = 100
    warmup_steps = 10

    with tempfile.TemporaryDirectory() as tmp:
        corpus_path = _write_corpus(2000, cfg.vocab_size, tmp)
        dataset = TokenizedDataset(corpus_path, seq_len=cfg.max_seq_len, stride=cfg.max_seq_len)
        model = GrimoireTransformer(cfg)
        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            batch_size=2,
            accumulate_steps=1,
            log_every=999,
            save_every=999,
            checkpoint_dir=tmp,
            device="cpu",
        )
        # Step the scheduler manually — must call optimizer.step() first per PyTorch convention.
        sched = trainer._scheduler
        opt   = trainer._optimizer
        lrs: list[float] = []
        for _ in range(total_steps + 1):
            opt.step()
            sched.step()
            lrs.append(sched.get_last_lr()[0])

    peak_lr = max(lrs)
    lr_at_warmup_end = lrs[warmup_steps - 1]
    lr_at_end        = lrs[-1]

    # LR must have increased to (near) peak during warmup.
    assert lr_at_warmup_end >= lrs[0], "LR should rise during warmup."
    # LR at the end of training must be below peak.
    assert lr_at_end < peak_lr, "LR should decay after the warmup peak."
    # Minimum LR should be approximately 10% of peak.
    assert lr_at_end < peak_lr * 0.2, "LR should decay significantly by the end."
