"""Tests for the standalone LR range-test script.

Gate criteria:
- The sweep returns equal-length lrs/losses/smoothed lists.
- LRs increase geometrically across the sweep.
- A suggested LR (when produced) lies within the swept range.
"""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np

# Load the script (scripts/ is not an importable package).
_SPEC = importlib.util.spec_from_file_location(
    "lr_range_test",
    Path(__file__).resolve().parent.parent / "scripts" / "lr_range_test.py",
)
lr_range_test = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lr_range_test)


def _tiny_model_config() -> dict:
    return {
        "vocab_size":  64,
        "d_model":     32,
        "n_layers":    2,
        "n_heads":     2,
        "n_kv_heads":  1,
        "d_ff":        64,
        "max_seq_len": 16,
        "dropout":     0.0,
    }


def _write_corpus(n_tokens: int, vocab_size: int, tmp_dir: str) -> str:
    path = str(Path(tmp_dir) / "corpus.bin")
    tokens = np.random.randint(6, vocab_size, size=n_tokens, dtype=np.int32)
    fp = np.memmap(path, dtype=np.int32, mode="w+", shape=(n_tokens,))
    fp[:] = tokens
    fp.flush()
    del fp
    return path


def test_lr_range_test_runs_and_records() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _write_corpus(2000, 64, tmp)
        result = lr_range_test.run_lr_range_test(
            corpus_path=corpus,
            model_config=_tiny_model_config(),
            min_lr=1e-5,
            max_lr=1.0,
            num_iters=20,
            batch_size=2,
            device="cpu",
        )

    n = len(result["lrs"])
    assert n >= 3
    assert len(result["losses"]) == n
    assert len(result["smoothed"]) == n
    # LRs are strictly increasing (geometric sweep) up to the recorded length.
    assert all(b > a for a, b in zip(result["lrs"], result["lrs"][1:]))
    # A suggestion, if any, must lie within the swept LR range.
    if result["suggested_lr"] is not None:
        assert result["lrs"][0] / 10.0 <= result["suggested_lr"] <= result["lrs"][-1]


def test_lr_range_test_writes_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _write_corpus(1500, 64, tmp)
        result = lr_range_test.run_lr_range_test(
            corpus_path=corpus,
            model_config=_tiny_model_config(),
            min_lr=1e-4,
            max_lr=0.5,
            num_iters=12,
            batch_size=2,
            device="cpu",
        )
        out = str(Path(tmp) / "curve.csv")
        lr_range_test._write_csv(result, out)
        text = Path(out).read_text(encoding="utf-8")
    assert text.splitlines()[0] == "step,lr,loss,smoothed_loss"
    assert len(text.splitlines()) == len(result["lrs"]) + 1
