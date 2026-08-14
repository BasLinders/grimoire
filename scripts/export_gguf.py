"""Export a GrimoireTransformer checkpoint to GGUF format for llama.cpp.

The exported file can be loaded by llama-cli, llama-server, and any runtime
that understands GGUF v3.  F16 halves the file size with negligible quality
loss; use llama-quantize for Q4_K_M compression after export:

    llama-quantize grimoire-f16.gguf grimoire-q4km.gguf Q4_K_M

Usage
-----
# F16 export (recommended):
python scripts/export_gguf.py \\
    --checkpoint checkpoints/pretrain/step_0010000.pt \\
    --output     models/grimoire-f16.gguf \\
    --vocab      data/tokenizer/bpe.json

# F32 export:
python scripts/export_gguf.py \\
    --checkpoint checkpoints/pretrain/step_0010000.pt \\
    --output     models/grimoire-f32.gguf \\
    --dtype      f32

# Run with llama.cpp after export:
#   llama-cli -m models/grimoire-f16.gguf -p "You are a D&D assistant." -n 256
# Or after quantization:
#   llama-cli -m grimoire-q4km.gguf -p "You are a D&D assistant." -n 256
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from grimoire_ai.llm.export.gguf_writer import GGUFWriter, grimoire_to_gguf_name

# RMSNorm epsilon used by GrimoireTransformer (block.py default).
_NORM_EPS = 1e-6

# GGUF general.file_type codes: ALL_F32=0, MOSTLY_F16=1.
_FILE_TYPE = {"f32": 0, "f16": 1}

# Keys that must be present in the state_dict for a valid export.
# Checked at startup so errors surface before writing any output.
_REQUIRED_TOPLEVEL = {"embedding._embed.weight", "final_norm.weight", "output_head.weight"}


def _load_vocab(vocab_path: str) -> dict:
    """Load a BPE vocab JSON and return parsed data."""
    return json.loads(Path(vocab_path).read_text(encoding="utf-8"))


def _add_tokenizer_kv(writer: GGUFWriter, vocab_data: dict) -> None:
    """Write tokenizer KV entries from a loaded BPE vocab dict."""
    vocab: dict[str, int] = vocab_data["vocab"]
    # Use the saved vocab_size (not len(vocab)) so that gaps or out-of-range
    # indices in a hand-edited file are caught as IndexError rather than
    # silently producing empty-string token entries.
    n: int = vocab_data["vocab_size"]
    id_to_token = [""] * n
    for tok, idx in vocab.items():
        id_to_token[idx] = tok

    writer.add_kv_str("tokenizer.ggml.model", "gpt2")
    writer.add_kv_array_str("tokenizer.ggml.tokens", id_to_token)

    if "merges" in vocab_data:
        # merges are stored as [[a, b], ...] pairs in bpe.json.
        merges = [f"{a} {b}" for a, b in vocab_data["merges"]]
        writer.add_kv_array_str("tokenizer.ggml.merges", merges)

    # Token types: 3=control/special, 1=normal.
    # _N_SPECIAL_TOKENS must equal _N_SPECIAL in grimoire_ai/llm/tokenizer/bpe.py.
    _N_SPECIAL_TOKENS = 6
    token_types = [3 if i < _N_SPECIAL_TOKENS else 1 for i in range(n)]
    writer.add_kv_array_int32("tokenizer.ggml.token_type", token_types)


def export_gguf(
    checkpoint_path: str,
    output_path: str,
    dtype: str = "f16",
    vocab_path: str | None = None,
) -> None:
    """Export a Grimoire checkpoint to a GGUF file.

    Args:
        checkpoint_path: Path to a ``.pt`` checkpoint written by
            ``save_checkpoint``.
        output_path: Destination ``.gguf`` file (created with parent dirs).
        dtype: Export dtype — ``"f16"`` (recommended) or ``"f32"``.
        vocab_path: Optional path to a BPE ``.json`` vocab file.  When
            provided, tokenizer metadata is embedded so llama.cpp can
            decode generated text.
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state_dict = ckpt["model"]

    # Validate that this looks like a Grimoire checkpoint.
    missing = _REQUIRED_TOPLEVEL - set(state_dict.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing expected keys: {sorted(missing)}. "
            "Is this a GrimoireTransformer checkpoint?"
        )

    # Warn about LoRA parameters — they are not merged into the export.
    lora_keys = [k for k in state_dict if "lora_" in k or "base_weight" in k]
    if lora_keys:
        print(
            f"WARNING: {len(lora_keys)} LoRA parameter(s) found in checkpoint "
            "(e.g. lora_A, lora_B, base_weight). These are NOT merged into the "
            "export. Run InferenceEngine.unload_lora() before exporting to bake "
            "the adapter into the base weights."
        )

    # GGUF export assumes GQA's q_proj/k_proj/v_proj state_dict keys (see
    # _per_block_suffixes below); MLA uses a different key structure
    # entirely. Checked explicitly here for a clear message — the
    # missing-block-keys check further down would also catch this, but with
    # a much less obvious error.
    attention_type = cfg.get("attention_type", "gqa")
    if attention_type != "gqa":
        raise NotImplementedError(
            f"GGUF export only supports attention_type='gqa' checkpoints; "
            f"this checkpoint uses attention_type={attention_type!r}. "
            "MultiHeadLatentAttention is not yet supported by "
            "scripts/export_gguf.py."
        )

    vocab_size: int = cfg["vocab_size"]
    d_model: int = cfg["d_model"]
    n_layers: int = cfg["n_layers"]
    n_heads: int = cfg["n_heads"]
    n_kv_heads: int = cfg["n_kv_heads"]
    d_ff: int = cfg["d_ff"]
    max_seq_len: int = cfg["max_seq_len"]
    rope_theta: float = cfg["rope_theta"]
    head_dim: int = d_model // n_heads

    print(
        f"Architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, "
        f"n_kv_heads={n_kv_heads}, d_ff={d_ff}, vocab={vocab_size}, "
        f"max_seq_len={max_seq_len}"
    )

    # ------------------------------------------------------------------ #
    # Build the GGUF file                                                  #
    # ------------------------------------------------------------------ #
    writer = GGUFWriter(dtype=dtype)

    # Architecture metadata
    writer.add_kv_str("general.architecture", "llama")
    writer.add_kv_str("general.name", "grimoire")
    writer.add_kv_uint32("general.file_type", _FILE_TYPE[dtype])
    writer.add_kv_uint32("llama.context_length", max_seq_len)
    writer.add_kv_uint32("llama.embedding_length", d_model)
    writer.add_kv_uint32("llama.block_count", n_layers)
    writer.add_kv_uint32("llama.feed_forward_length", d_ff)
    writer.add_kv_uint32("llama.rope.dimension_count", head_dim)
    writer.add_kv_float32("llama.rope.freq_base", rope_theta)
    writer.add_kv_uint32("llama.attention.head_count", n_heads)
    writer.add_kv_uint32("llama.attention.head_count_kv", n_kv_heads)
    writer.add_kv_float32("llama.attention.layer_norm_rms_epsilon", _NORM_EPS)

    # Tokenizer metadata
    if vocab_path:
        if Path(vocab_path).is_file():
            print(f"Embedding tokenizer metadata from: {vocab_path}")
            _add_tokenizer_kv(writer, _load_vocab(vocab_path))
        else:
            print(f"WARNING: --vocab path not found: {vocab_path!r} — "
                  "tokenizer metadata not embedded; llama.cpp will not be able to decode output")
            writer.add_kv_str("tokenizer.ggml.model", "gpt2")
    else:
        writer.add_kv_str("tokenizer.ggml.model", "gpt2")

    # Validate that cfg["n_layers"] matches the actual block count in the checkpoint.
    _per_block_suffixes = [
        "attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
        "attn.v_proj.weight", "attn.o_proj.weight", "ffn_norm.weight",
        "ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight",
    ]
    missing_block_keys = [
        f"blocks.{i}.{s}"
        for i in range(n_layers)
        for s in _per_block_suffixes
        if f"blocks.{i}.{s}" not in state_dict
    ]
    if missing_block_keys:
        raise ValueError(
            f"Checkpoint config says n_layers={n_layers} but "
            f"{len(missing_block_keys)} expected block key(s) are absent, e.g. "
            f"{missing_block_keys[0]!r}. Is this the right checkpoint?"
        )

    # Validate that cfg["vocab_size"] matches the actual embedding weight shape.
    actual_vocab = state_dict["embedding._embed.weight"].shape[0]
    if actual_vocab != vocab_size:
        raise ValueError(
            f"cfg['vocab_size']={vocab_size} does not match "
            f"embedding._embed.weight.shape[0]={actual_vocab}. "
            "The checkpoint config is inconsistent with its weights."
        )

    # Tensors — exported in canonical llama.cpp order.
    def sd_np(key: str) -> np.ndarray:
        return state_dict[key].float().numpy()

    writer.add_tensor("token_embd.weight", sd_np("embedding._embed.weight"))

    for i in range(n_layers):
        writer.add_tensor(f"blk.{i}.attn_norm.weight",
                          sd_np(f"blocks.{i}.attn_norm.weight"), force_f32=True)
        writer.add_tensor(f"blk.{i}.attn_q.weight",
                          sd_np(f"blocks.{i}.attn.q_proj.weight"))
        writer.add_tensor(f"blk.{i}.attn_k.weight",
                          sd_np(f"blocks.{i}.attn.k_proj.weight"))
        writer.add_tensor(f"blk.{i}.attn_v.weight",
                          sd_np(f"blocks.{i}.attn.v_proj.weight"))
        writer.add_tensor(f"blk.{i}.attn_output.weight",
                          sd_np(f"blocks.{i}.attn.o_proj.weight"))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight",
                          sd_np(f"blocks.{i}.ffn_norm.weight"), force_f32=True)
        writer.add_tensor(f"blk.{i}.ffn_gate.weight",
                          sd_np(f"blocks.{i}.ffn.gate_proj.weight"))
        writer.add_tensor(f"blk.{i}.ffn_up.weight",
                          sd_np(f"blocks.{i}.ffn.up_proj.weight"))
        writer.add_tensor(f"blk.{i}.ffn_down.weight",
                          sd_np(f"blocks.{i}.ffn.down_proj.weight"))

    writer.add_tensor("output_norm.weight", sd_np("final_norm.weight"), force_f32=True)
    # output_head.weight is weight-tied to embedding._embed.weight; export once.
    writer.add_tensor("output.weight", sd_np("output_head.weight"))

    # Log any state_dict keys that map to a GGUF name but were not exported.
    exported = {
        "embedding._embed.weight", "final_norm.weight", "output_head.weight",
        *(f"blocks.{i}.{s}"
          for i in range(n_layers)
          for s in [
              "attn_norm.weight", "attn.q_proj.weight", "attn.k_proj.weight",
              "attn.v_proj.weight", "attn.o_proj.weight", "ffn_norm.weight",
              "ffn.gate_proj.weight", "ffn.up_proj.weight", "ffn.down_proj.weight",
          ]),
    }
    for key in sorted(state_dict.keys()):
        if key in exported:
            continue
        gguf_name = grimoire_to_gguf_name(key)
        if gguf_name is not None:
            print(f"  WARNING: unmapped key not exported: {key!r} → {gguf_name!r}")

    n_tensors = 1 + n_layers * 9 + 2  # embd + 9 per block + output_norm + output
    print(f"Writing {n_tensors} tensors ({dtype.upper()}) → {output_path}")
    writer.write(output_path)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB")
    print()
    print("Next steps:")
    print(f"  Quantize:  llama-quantize {output_path} "
          f"{Path(output_path).stem}-q4km.gguf Q4_K_M")
    print(f"  Run:       llama-cli -m {output_path} "
          f"-p \"You are a D&D assistant.\" -n 256")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a GrimoireTransformer checkpoint to GGUF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to a .pt checkpoint file.",
    )
    parser.add_argument(
        "--output", required=True, metavar="PATH",
        help="Destination .gguf output file.",
    )
    parser.add_argument(
        "--dtype", choices=["f16", "f32"], default="f16",
        help="Export dtype: f16 (default, recommended) or f32.",
    )
    parser.add_argument(
        "--vocab", metavar="PATH", default=None,
        help="Path to BPE vocab JSON (embeds tokenizer metadata).",
    )
    args = parser.parse_args()
    export_gguf(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        dtype=args.dtype,
        vocab_path=args.vocab,
    )


if __name__ == "__main__":
    main()
