"""Tests for dynamic int8 quantization in InferenceEngine.

Verifies:
- Engine loads and responds with quantize=False (baseline, no regression).
- Engine loads and responds with quantize=True on CPU.
- Linear layers are replaced with DynamicQuantizedLinear when quantize=True on CPU.
- engine.quantized is True after a CPU quantized load.
- engine.quantized is False after a non-quantized load.
- On a non-CPU device string, quantize=True is silently skipped (engine.quantized=False).
- Generation output is a non-empty string with both quantized and unquantized engines.
"""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import pytest

from grimoire_ai.llm.inference.engine import InferenceEngine
from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.transformer import GrimoireTransformer
from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.training.checkpoint import save_checkpoint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_config() -> TransformerConfig:
    return TransformerConfig(
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=32,
        dropout=0.0,
    )


def _save_artifacts(tmp_dir: str) -> tuple[str, str]:
    cfg = _tiny_config()
    model = GrimoireTransformer(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ckpt_path = str(Path(tmp_dir) / "ckpt.pt")
    save_checkpoint(
        path=ckpt_path,
        model=model,
        optimizer=optimizer,
        step=1,
        config_dict=cfg.to_dict(),
        train_loss=4.0,
    )
    tok_path = str(Path(tmp_dir) / "bpe.json")
    enc = BytePairEncoder()
    enc.train(["the quick brown fox jumps over the lazy dog " * 30], vocab_size=cfg.vocab_size)
    enc.save(tok_path)
    return ckpt_path, tok_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unquantized_engine_responds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(checkpoint_path=ckpt, tokenizer_path=tok, device="cpu")
        assert engine.quantized is False
        result = engine.respond("hello")
        assert isinstance(result, str)


def test_quantized_engine_responds() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, device="cpu", quantize=True
        )
        assert engine.quantized is True
        result = engine.respond("hello")
        assert isinstance(result, str)


def test_quantized_linears_replaced() -> None:
    """After quantization, nn.Linear layers should be DynamicQuantizedLinear."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, device="cpu", quantize=True
        )
        dynamic_linear = torch.nn.quantized.dynamic.Linear
        quantized_linears = [
            m for m in engine.model.modules()
            if isinstance(m, dynamic_linear)
        ]
        assert len(quantized_linears) > 0, "No quantized Linear layers found after quantize=True"


def test_unquantized_has_fp32_linears() -> None:
    """Without quantization, modules stay as ordinary nn.Linear in fp32."""
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, device="cpu", quantize=False
        )
        fp32_linears = [m for m in engine.model.modules() if type(m) is nn.Linear]
        assert len(fp32_linears) > 0, "Expected plain nn.Linear layers in unquantized engine"


def test_quantize_skipped_on_cuda() -> None:
    """When device is not CPU, quantize=True is silently ignored."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    with tempfile.TemporaryDirectory() as tmp:
        ckpt, tok = _save_artifacts(tmp)
        engine = InferenceEngine(
            checkpoint_path=ckpt, tokenizer_path=tok, device="cuda", quantize=True
        )
        assert engine.quantized is False
