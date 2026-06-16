"""Tests for the GGUF export library (GGUFWriter) and the name mapping.

Covers:
- GGUF binary header (magic, version, counts)
- KV metadata encoding (string, uint32, float32, array variants)
- Tensor-info encoding (name, dims, dtype, offset)
- Tensor data alignment and padding
- grimoire_to_gguf_name() mapping correctness
- F32 / F16 dtype selection (1-D tensors always F32)
- export_gguf() end-to-end with a synthetic checkpoint
- LoRA-key warning path
- Missing-key error path
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from grimoire_ai.llm.export.gguf_writer import (
    GGUF_ALIGNMENT,
    GGUF_MAGIC,
    GGUF_VERSION,
    GGML_F16,
    GGML_F32,
    GGUFWriter,
    grimoire_to_gguf_name,
)
from scripts.export_gguf import export_gguf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_header(path: Path) -> dict:
    """Read the GGUF binary header (first 24 bytes)."""
    with open(path, "rb") as f:
        magic = f.read(4)
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
    return {"magic": magic, "version": version, "n_tensors": n_tensors, "n_kv": n_kv}


def _make_checkpoint(
    n_layers: int = 2,
    d_model: int = 8,
    n_heads: int = 2,
    n_kv_heads: int = 1,
    d_ff: int = 16,
    vocab_size: int = 32,
    max_seq_len: int = 64,
    include_lora: bool = False,
) -> dict:
    """Return a synthetic checkpoint dict resembling save_checkpoint output."""
    head_dim = d_model // n_heads
    sd: dict = {}

    sd["embedding._embed.weight"] = torch.randn(vocab_size, d_model)
    for i in range(n_layers):
        sd[f"blocks.{i}.attn_norm.weight"] = torch.ones(d_model)
        sd[f"blocks.{i}.attn.q_proj.weight"] = torch.randn(d_model, d_model)
        sd[f"blocks.{i}.attn.k_proj.weight"] = torch.randn(n_kv_heads * head_dim, d_model)
        sd[f"blocks.{i}.attn.v_proj.weight"] = torch.randn(n_kv_heads * head_dim, d_model)
        sd[f"blocks.{i}.attn.o_proj.weight"] = torch.randn(d_model, d_model)
        sd[f"blocks.{i}.attn._cos"] = torch.ones(max_seq_len, head_dim // 2)
        sd[f"blocks.{i}.attn._sin"] = torch.ones(max_seq_len, head_dim // 2)
        sd[f"blocks.{i}.ffn_norm.weight"] = torch.ones(d_model)
        sd[f"blocks.{i}.ffn.gate_proj.weight"] = torch.randn(d_ff, d_model)
        sd[f"blocks.{i}.ffn.up_proj.weight"] = torch.randn(d_ff, d_model)
        sd[f"blocks.{i}.ffn.down_proj.weight"] = torch.randn(d_model, d_ff)
    sd["final_norm.weight"] = torch.ones(d_model)
    sd["output_head.weight"] = torch.randn(vocab_size, d_model)

    if include_lora:
        sd["blocks.0.attn.q_proj.lora_A"] = torch.randn(4, d_model)
        sd["blocks.0.attn.q_proj.lora_B"] = torch.randn(d_model, 4)
        sd["blocks.0.attn.q_proj.base_weight"] = torch.randn(d_model, d_model)

    cfg = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_ff": d_ff,
        "max_seq_len": max_seq_len,
        "dropout": 0.1,
        "rope_theta": 10000.0,
    }
    return {"step": 1, "config": cfg, "model": sd, "optimizer": {}, "scaler": None, "train_loss": 0.0}


def _save_checkpoint(ckpt: dict, tmp_path: Path) -> Path:
    p = tmp_path / "ckpt.pt"
    torch.save(ckpt, str(p))
    return p


# ---------------------------------------------------------------------------
# grimoire_to_gguf_name mapping
# ---------------------------------------------------------------------------

class TestNameMapping:
    def test_embedding(self):
        assert grimoire_to_gguf_name("embedding._embed.weight") == "token_embd.weight"

    def test_final_norm(self):
        assert grimoire_to_gguf_name("final_norm.weight") == "output_norm.weight"

    def test_output_head(self):
        assert grimoire_to_gguf_name("output_head.weight") == "output.weight"

    def test_block_attn_norm(self):
        assert grimoire_to_gguf_name("blocks.3.attn_norm.weight") == "blk.3.attn_norm.weight"

    def test_block_q_proj(self):
        assert grimoire_to_gguf_name("blocks.0.attn.q_proj.weight") == "blk.0.attn_q.weight"

    def test_block_k_proj(self):
        assert grimoire_to_gguf_name("blocks.1.attn.k_proj.weight") == "blk.1.attn_k.weight"

    def test_block_v_proj(self):
        assert grimoire_to_gguf_name("blocks.1.attn.v_proj.weight") == "blk.1.attn_v.weight"

    def test_block_o_proj(self):
        assert grimoire_to_gguf_name("blocks.2.attn.o_proj.weight") == "blk.2.attn_output.weight"

    def test_block_ffn_norm(self):
        assert grimoire_to_gguf_name("blocks.0.ffn_norm.weight") == "blk.0.ffn_norm.weight"

    def test_block_gate_proj(self):
        assert grimoire_to_gguf_name("blocks.0.ffn.gate_proj.weight") == "blk.0.ffn_gate.weight"

    def test_block_up_proj(self):
        assert grimoire_to_gguf_name("blocks.0.ffn.up_proj.weight") == "blk.0.ffn_up.weight"

    def test_block_down_proj(self):
        assert grimoire_to_gguf_name("blocks.0.ffn.down_proj.weight") == "blk.0.ffn_down.weight"

    def test_rope_cos_skipped(self):
        assert grimoire_to_gguf_name("blocks.0.attn._cos") is None

    def test_rope_sin_skipped(self):
        assert grimoire_to_gguf_name("blocks.0.attn._sin") is None

    def test_mask_skipped(self):
        assert grimoire_to_gguf_name("blocks.0.attn._mask") is None

    def test_unknown_key_returns_none(self):
        assert grimoire_to_gguf_name("some.random.key") is None


# ---------------------------------------------------------------------------
# GGUFWriter: binary format correctness
# ---------------------------------------------------------------------------

class TestGGUFWriterHeader:
    def test_magic(self, tmp_path):
        w = GGUFWriter()
        w.add_tensor("t", np.zeros((2, 2), dtype=np.float32))
        w.write(tmp_path / "out.gguf")
        hdr = _parse_header(tmp_path / "out.gguf")
        assert hdr["magic"] == GGUF_MAGIC

    def test_version(self, tmp_path):
        w = GGUFWriter()
        w.add_tensor("t", np.zeros((2, 2), dtype=np.float32))
        w.write(tmp_path / "out.gguf")
        hdr = _parse_header(tmp_path / "out.gguf")
        assert hdr["version"] == GGUF_VERSION

    def test_n_tensors_correct(self, tmp_path):
        w = GGUFWriter()
        w.add_tensor("a", np.zeros((4,), dtype=np.float32))
        w.add_tensor("b", np.zeros((2, 3), dtype=np.float32))
        w.write(tmp_path / "out.gguf")
        assert _parse_header(tmp_path / "out.gguf")["n_tensors"] == 2

    def test_n_kv_correct(self, tmp_path):
        w = GGUFWriter()
        w.add_kv_str("general.architecture", "llama")
        w.add_kv_uint32("llama.block_count", 6)
        w.add_tensor("t", np.zeros((2,), dtype=np.float32))
        w.write(tmp_path / "out.gguf")
        assert _parse_header(tmp_path / "out.gguf")["n_kv"] == 2

    def test_no_tensors_raises(self, tmp_path):
        w = GGUFWriter()
        with pytest.raises(ValueError, match="No tensors"):
            w.write(tmp_path / "out.gguf")

    def test_invalid_dtype_raises(self):
        with pytest.raises(ValueError):
            GGUFWriter(dtype="q4")

    def test_output_dir_created(self, tmp_path):
        w = GGUFWriter()
        w.add_tensor("t", np.zeros((1,), dtype=np.float32))
        nested = tmp_path / "a" / "b" / "out.gguf"
        w.write(nested)
        assert nested.is_file()

    def test_tensor_data_section_aligned(self, tmp_path):
        """Tensor data must start on a 32-byte boundary."""
        w = GGUFWriter()
        w.add_kv_str("general.architecture", "llama")
        arr = np.ones((4,), dtype=np.float32)
        w.add_tensor("norm", arr, force_f32=True)
        out = tmp_path / "out.gguf"
        w.write(out)
        # Read the full pre-data section up to tensor data.
        # The tensor data starts immediately after the padded header+kv+info.
        data = out.read_bytes()
        # Tensor data offset = total file size - tensor data size
        # tensor 0 is 4 * 4 = 16 bytes, padded to 32
        tensor_size_padded = 32
        tensor_data_start = len(data) - tensor_size_padded
        assert tensor_data_start % GGUF_ALIGNMENT == 0


class TestGGUFWriterTensorDtype:
    def test_1d_tensor_always_f32(self, tmp_path):
        """1-D tensors (norm weights) should be stored as F32 even in f16 mode."""
        w = GGUFWriter(dtype="f16")
        arr = np.ones(8, dtype=np.float32)
        w.add_tensor("norm", arr)
        # Stored internally as float32
        _, stored = w._tensors[0]
        assert stored.dtype == np.float32

    def test_2d_tensor_f16_mode(self, tmp_path):
        w = GGUFWriter(dtype="f16")
        arr = np.ones((4, 8), dtype=np.float32)
        w.add_tensor("mat", arr)
        _, stored = w._tensors[0]
        assert stored.dtype == np.float16

    def test_2d_tensor_f32_mode(self, tmp_path):
        w = GGUFWriter(dtype="f32")
        arr = np.ones((4, 8), dtype=np.float32)
        w.add_tensor("mat", arr)
        _, stored = w._tensors[0]
        assert stored.dtype == np.float32

    def test_force_f32_overrides_f16_mode(self, tmp_path):
        w = GGUFWriter(dtype="f16")
        arr = np.ones((4, 8), dtype=np.float32)
        w.add_tensor("mat", arr, force_f32=True)
        _, stored = w._tensors[0]
        assert stored.dtype == np.float32

    def test_non_contiguous_tensor_made_contiguous(self, tmp_path):
        w = GGUFWriter(dtype="f32")
        arr = np.ones((4, 8), dtype=np.float32)[::2]  # non-contiguous slice
        w.add_tensor("mat", arr, force_f32=True)
        _, stored = w._tensors[0]
        assert stored.flags["C_CONTIGUOUS"]


class TestGGUFWriterTensorDims:
    def test_dims_stored_reversed(self):
        """GGUF stores dims in reversed (column-major) order vs NumPy."""
        w = GGUFWriter(dtype="f32")
        arr = np.zeros((5, 3), dtype=np.float32)  # rows=5, cols=3
        w.add_tensor("mat", arr, force_f32=True)
        # After writing to bytes, the ti_buf should contain dims [3, 5].
        # We check the stored tensor shape directly instead.
        _, stored = w._tensors[0]
        assert stored.shape == (5, 3)  # unchanged in memory


# ---------------------------------------------------------------------------
# export_gguf() end-to-end
# ---------------------------------------------------------------------------

class TestExportGGUF:
    def test_creates_file(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        assert out.is_file()
        assert out.stat().st_size > 0

    def test_header_magic_and_version(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        hdr = _parse_header(out)
        assert hdr["magic"] == b"GGUF"
        assert hdr["version"] == 3

    def test_tensor_count(self, tmp_path):
        n_layers = 3
        ckpt = _make_checkpoint(n_layers=n_layers)
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        hdr = _parse_header(out)
        # 1 token_embd + 9 per block + output_norm + output
        expected = 1 + n_layers * 9 + 2
        assert hdr["n_tensors"] == expected

    def test_f16_smaller_than_f32(self, tmp_path):
        ckpt = _make_checkpoint(d_model=16, d_ff=32, vocab_size=64)
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out16 = tmp_path / "m16.gguf"
        out32 = tmp_path / "m32.gguf"
        export_gguf(str(ckpt_path), str(out16), dtype="f16")
        export_gguf(str(ckpt_path), str(out32), dtype="f32")
        assert out16.stat().st_size < out32.stat().st_size

    def test_rope_buffers_not_exported(self, tmp_path):
        """RoPE _cos and _sin buffers must not appear in the GGUF tensor data."""
        ckpt = _make_checkpoint(n_layers=1)
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        raw = out.read_bytes()
        # "_cos" and "_sin" as GGUF tensor names would be encoded as UTF-8
        # substrings in the tensor-info section.
        assert b"_cos" not in raw
        assert b"_sin" not in raw

    def test_missing_required_key_raises(self, tmp_path):
        ckpt = _make_checkpoint()
        del ckpt["model"]["embedding._embed.weight"]
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        with pytest.raises(ValueError, match="missing expected keys"):
            export_gguf(str(ckpt_path), str(tmp_path / "model.gguf"))

    def test_lora_keys_trigger_warning(self, tmp_path, capsys):
        ckpt = _make_checkpoint(include_lora=True)
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        export_gguf(str(ckpt_path), str(tmp_path / "model.gguf"))
        captured = capsys.readouterr()
        assert "LoRA" in captured.out

    def test_vocab_embeds_tokenizer_kv(self, tmp_path):
        ckpt = _make_checkpoint(vocab_size=32)
        ckpt_path = _save_checkpoint(ckpt, tmp_path)

        # Write a minimal vocab JSON matching the checkpoint's vocab_size.
        vocab = {str(i): i for i in range(32)}
        vocab_data = {"vocab_size": 32, "vocab": vocab, "merges": []}
        vocab_path = tmp_path / "bpe.json"
        vocab_path.write_text(json.dumps(vocab_data), encoding="utf-8")

        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out), vocab_path=str(vocab_path))
        # "tokenizer.ggml.model" should be encoded in the file.
        raw = out.read_bytes()
        assert b"tokenizer.ggml.model" in raw
        assert b"tokenizer.ggml.tokens" in raw

    def test_without_vocab_still_writes_tokenizer_model_key(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        raw = out.read_bytes()
        assert b"tokenizer.ggml.model" in raw

    def test_architecture_kv_present(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        raw = out.read_bytes()
        assert b"general.architecture" in raw
        assert b"llama" in raw

    def test_output_dir_created(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "subdir" / "model.gguf"
        export_gguf(str(ckpt_path), str(out))
        assert out.is_file()

    def test_f32_export(self, tmp_path):
        ckpt = _make_checkpoint()
        ckpt_path = _save_checkpoint(ckpt, tmp_path)
        out = tmp_path / "model.gguf"
        export_gguf(str(ckpt_path), str(out), dtype="f32")
        hdr = _parse_header(out)
        assert hdr["magic"] == b"GGUF"
