"""GGUF binary writer for GrimoireTransformer checkpoints.

GGUF (version 3) is the portable format used by llama.cpp.  This module
implements the full binary layout — header, KV metadata, tensor info, and
tensor data — so that exported files can be loaded by llama-cli, llama-server,
and any other llama.cpp-compatible runtime without a Python dependency.

On-disk layout
--------------
Bytes 0–23  : Header (magic 4B, version 4B, n_tensors 8B, n_kv 8B)
After header: n_kv key-value metadata pairs (variable length)
After KV    : n_tensors tensor-info records (name, dims, dtype, offset)
Padding     : zeros to the next 32-byte boundary
Tensor data : each tensor's raw bytes, each padded to 32 bytes

All integers are little-endian.  Strings are uint64-length-prefixed UTF-8
(no null terminator).  Tensor dims are stored in GGUF order — the reverse
of NumPy (row-major) order — so a PyTorch ``(out, in)`` matrix is listed
as dims ``[in, out]``.  The actual data bytes are unchanged.

Architecture mapping (GrimoireTransformer → GGUF "llama" architecture)
-----------------------------------------------------------------------
embedding._embed.weight          → token_embd.weight
blocks.{i}.attn_norm.weight      → blk.{i}.attn_norm.weight
blocks.{i}.attn.q_proj.weight    → blk.{i}.attn_q.weight
blocks.{i}.attn.k_proj.weight    → blk.{i}.attn_k.weight
blocks.{i}.attn.v_proj.weight    → blk.{i}.attn_v.weight
blocks.{i}.attn.o_proj.weight    → blk.{i}.attn_output.weight
blocks.{i}.ffn_norm.weight       → blk.{i}.ffn_norm.weight
blocks.{i}.ffn.gate_proj.weight  → blk.{i}.ffn_gate.weight
blocks.{i}.ffn.up_proj.weight    → blk.{i}.ffn_up.weight
blocks.{i}.ffn.down_proj.weight  → blk.{i}.ffn_down.weight
final_norm.weight                → output_norm.weight
output_head.weight               → output.weight  (weight-tied; exported once)

RoPE buffers (_cos, _sin) and attention mask (_mask) are not exported:
llama.cpp recomputes them at load time from ``rope_theta`` and context length.
"""

import struct
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# GGUF binary constants
# ---------------------------------------------------------------------------

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGNMENT = 32  # default tensor-data alignment in bytes

# GGUF metadata value-type codes
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

# GGML tensor dtype codes (subset used here)
GGML_F32 = 0
GGML_F16 = 1

# RoPE buffers and attention mask are not exported.
_SKIP_SUFFIXES: frozenset[str] = frozenset({"._cos", "._sin", "._mask"})

# Grimoire state_dict key → GGUF tensor name, for per-block tensors.
_BLOCK_SUFFIX_MAP: dict[str, str] = {
    "attn_norm.weight":     "attn_norm.weight",
    "ffn_norm.weight":      "ffn_norm.weight",
    "attn.q_proj.weight":   "attn_q.weight",
    "attn.k_proj.weight":   "attn_k.weight",
    "attn.v_proj.weight":   "attn_v.weight",
    "attn.o_proj.weight":   "attn_output.weight",
    "ffn.gate_proj.weight": "ffn_gate.weight",
    "ffn.up_proj.weight":   "ffn_up.weight",
    "ffn.down_proj.weight": "ffn_down.weight",
}


def grimoire_to_gguf_name(sd_key: str) -> Optional[str]:
    """Map a GrimoireTransformer state_dict key to its GGUF tensor name.

    Args:
        sd_key: A key from ``model.state_dict()``.

    Returns:
        The GGUF tensor name, or ``None`` if the tensor should not be exported
        (RoPE buffers, attention mask, or any unrecognised key).
    """
    for suffix in _SKIP_SUFFIXES:
        if suffix in sd_key:
            return None

    if sd_key == "embedding._embed.weight":
        return "token_embd.weight"
    if sd_key == "final_norm.weight":
        return "output_norm.weight"
    if sd_key == "output_head.weight":
        return "output.weight"

    parts = sd_key.split(".", 2)
    if parts[0] == "blocks" and len(parts) == 3:
        blk_idx, sub = parts[1], parts[2]
        gguf_sub = _BLOCK_SUFFIX_MAP.get(sub)
        if gguf_sub:
            return f"blk.{blk_idx}.{gguf_sub}"

    return None


# ---------------------------------------------------------------------------
# Low-level binary packing helpers
# ---------------------------------------------------------------------------

def _enc_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pad_to_align(n: int, alignment: int = GGUF_ALIGNMENT) -> int:
    """Return the number of padding bytes needed to align *n* to *alignment*."""
    rem = n % alignment
    return (alignment - rem) % alignment


# ---------------------------------------------------------------------------
# GGUFWriter
# ---------------------------------------------------------------------------

class GGUFWriter:
    """Accumulates GGUF metadata and tensors, then serialises to disk.

    Typical usage::

        writer = GGUFWriter(dtype="f16")
        writer.add_kv_str("general.architecture", "llama")
        writer.add_kv_uint32("llama.context_length", 1024)
        writer.add_tensor("token_embd.weight", embedding_np_array)
        writer.write("models/grimoire-f16.gguf")
    """

    def __init__(self, dtype: str = "f16") -> None:
        """Initialise an empty writer.

        Args:
            dtype: Default tensor dtype — ``"f16"`` (recommended) or ``"f32"``.
                One-dimensional tensors (norms) are always stored as F32
                regardless of this setting.
        """
        if dtype not in ("f16", "f32"):
            raise ValueError(f"dtype must be 'f16' or 'f32', got {dtype!r}")
        self._dtype = dtype
        self._np_dtype = np.float16 if dtype == "f16" else np.float32
        self._ggml_dtype = GGML_F16 if dtype == "f16" else GGML_F32
        self._kv_buf = bytearray()
        self._kv_count = 0
        # (name, numpy_array) in insertion order
        self._tensors: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------ #
    # KV metadata                                                          #
    # ------------------------------------------------------------------ #

    def add_kv_str(self, key: str, value: str) -> None:
        """Add a string-valued KV metadata entry."""
        self._kv_buf += _enc_str(key) + struct.pack("<I", _STRING) + _enc_str(value)
        self._kv_count += 1

    def add_kv_uint32(self, key: str, value: int) -> None:
        """Add a uint32-valued KV metadata entry."""
        self._kv_buf += _enc_str(key) + struct.pack("<II", _UINT32, value)
        self._kv_count += 1

    def add_kv_float32(self, key: str, value: float) -> None:
        """Add a float32-valued KV metadata entry."""
        self._kv_buf += _enc_str(key) + struct.pack("<If", _FLOAT32, value)
        self._kv_count += 1

    def add_kv_bool(self, key: str, value: bool) -> None:
        """Add a bool-valued KV metadata entry."""
        self._kv_buf += _enc_str(key) + struct.pack("<I?", _BOOL, value)
        self._kv_count += 1

    def add_kv_array_str(self, key: str, values: list[str]) -> None:
        """Add a string-array KV metadata entry."""
        buf = _enc_str(key)
        buf += struct.pack("<I", _ARRAY)
        buf += struct.pack("<IQ", _STRING, len(values))
        for v in values:
            buf += _enc_str(v)
        self._kv_buf += buf
        self._kv_count += 1

    def add_kv_array_int32(self, key: str, values: list[int]) -> None:
        """Add an int32-array KV metadata entry."""
        buf = _enc_str(key)
        buf += struct.pack("<I", _ARRAY)
        buf += struct.pack("<IQ", _INT32, len(values))
        buf += struct.pack(f"<{len(values)}i", *values)
        self._kv_buf += buf
        self._kv_count += 1

    def add_kv_array_float32(self, key: str, values: list[float]) -> None:
        """Add a float32-array KV metadata entry."""
        buf = _enc_str(key)
        buf += struct.pack("<I", _ARRAY)
        buf += struct.pack("<IQ", _FLOAT32, len(values))
        buf += struct.pack(f"<{len(values)}f", *values)
        self._kv_buf += buf
        self._kv_count += 1

    # ------------------------------------------------------------------ #
    # Tensors                                                              #
    # ------------------------------------------------------------------ #

    def add_tensor(
        self,
        name: str,
        array: np.ndarray,
        force_f32: bool = False,
    ) -> None:
        """Queue a tensor for export.

        Args:
            name: GGUF tensor name (e.g. ``"blk.0.attn_q.weight"``).
            array: Float numpy array.  Will be cast to the writer's dtype
                (or F32 when *force_f32* is ``True`` or the array is 1-D).
            force_f32: Store as F32 regardless of the writer's dtype.
        """
        arr = array.astype(np.float32, copy=False)
        if not force_f32 and arr.ndim > 1:
            arr = arr.astype(self._np_dtype, copy=False)
        # Ensure C-contiguous for correct byte serialisation.
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        self._tensors.append((name, arr))

    # ------------------------------------------------------------------ #
    # Serialisation                                                        #
    # ------------------------------------------------------------------ #

    def write(self, path: "str | Path") -> None:
        """Write the accumulated GGUF data to *path*.

        Args:
            path: Destination ``.gguf`` file.  Parent directories are
                created if absent.

        Raises:
            ValueError: If no tensors have been added.
        """
        if not self._tensors:
            raise ValueError("No tensors added — call add_tensor() before write().")

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        n_tensors = len(self._tensors)
        n_kv = self._kv_count

        # -- Tensor data: compute offsets first --
        tensor_blobs: list[bytes] = []
        offsets: list[int] = []
        cur_offset = 0
        for _, arr in self._tensors:
            offsets.append(cur_offset)
            raw = arr.tobytes()
            pad = _pad_to_align(len(raw))
            blob = raw + b"\x00" * pad
            tensor_blobs.append(blob)
            cur_offset += len(blob)

        # -- Tensor info records --
        ti_buf = bytearray()
        for (name, arr), offset in zip(self._tensors, offsets):
            ti_buf += _enc_str(name)
            ti_buf += struct.pack("<I", arr.ndim)
            # GGUF dim order is the reverse of NumPy shape order.
            for d in reversed(arr.shape):
                ti_buf += struct.pack("<Q", d)
            ggml_t = GGML_F32 if arr.dtype == np.float32 else GGML_F16
            ti_buf += struct.pack("<I", ggml_t)
            ti_buf += struct.pack("<Q", offset)

        # -- Header --
        header = (
            GGUF_MAGIC
            + struct.pack("<I", GGUF_VERSION)
            + struct.pack("<Q", n_tensors)
            + struct.pack("<Q", n_kv)
        )

        # Pad header + KV + tensor_info to GGUF_ALIGNMENT so tensor data
        # starts on an aligned boundary.
        pre_data = header + bytes(self._kv_buf) + bytes(ti_buf)
        pre_pad = _pad_to_align(len(pre_data))

        with open(p, "wb") as f:
            f.write(pre_data)
            f.write(b"\x00" * pre_pad)
            for blob in tensor_blobs:
                f.write(blob)
