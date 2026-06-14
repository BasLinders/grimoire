"""LoRA: Low-Rank Adaptation for parameter-efficient fine-tuning.

LoRALinear wraps a frozen nn.Linear and adds two trainable low-rank
matrices A (r × in) and B (out × r) whose product approximates the
weight update:

    h = W x + scale * B A x   (scale = alpha / rank)

A is initialised with Kaiming-uniform, B with zeros — so the initial
LoRA delta is exactly zero and the model starts identical to the base
checkpoint.

Usage
-----
    model.add_lora_adapters(rank=8, alpha=16.0, targets=["q_proj", "v_proj"])
    # … fine-tune; only lora_A / lora_B are updated …
    model.merge_and_unload()   # bake adapters into base weights

Serialisation
-------------
    from grimoire_ai.llm.model.lora import save_lora, load_lora

    save_lora(model, rank=8, alpha=16.0, targets=[...], path="saga.lora")
    load_lora(model, "saga.lora")   # re-applies adapters if not present
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """nn.Linear with frozen base weight and trainable low-rank bypass.

    Args:
        linear: The original (to-be-frozen) nn.Linear to wrap.
        rank: Low-rank dimension r (columns of A, rows of B).
        alpha: LoRA scaling constant; effective scale = alpha / rank.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        out_features, in_features = linear.weight.shape
        self.in_features  = in_features
        self.out_features = out_features
        self.rank  = rank
        self.scale = alpha / rank

        # Frozen base weight registered as a buffer so it moves with .to()
        # but is excluded from any optimizer param group.
        self.register_buffer("base_weight", linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer("base_bias", linear.bias.data.clone())
        else:
            self.register_buffer("base_bias", None)

        # Trainable low-rank matrices.
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.base_weight, self.base_bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with the LoRA delta baked into the weight."""
        merged = self.base_weight + (self.lora_B @ self.lora_A) * self.scale
        out = nn.Linear(
            self.in_features, self.out_features,
            bias=self.base_bias is not None,
            device=merged.device,
            dtype=merged.dtype,
        )
        out.weight = nn.Parameter(merged)
        if self.base_bias is not None:
            out.bias = nn.Parameter(self.base_bias.clone())
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"rank={self.rank}, scale={self.scale:.3g}"
        )


def save_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: list[str],
    path: str,
) -> None:
    """Save LoRA adapter weights to *path*.

    Only lora_A and lora_B tensors are written — the frozen base weights
    are omitted, keeping the file to 2–5 MB for rank=8.

    Args:
        model: GrimoireTransformer with LoRA adapters in place.
        rank: LoRA rank used when the adapters were created.
        alpha: LoRA alpha used when the adapters were created.
        targets: Names of the Linear layers that were wrapped.
        path: Destination path (``.lora`` extension recommended).
    """
    lora_sd = {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
    payload = {
        "rank":       rank,
        "alpha":      alpha,
        "targets":    list(targets),
        "state_dict": lora_sd,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(p))


def load_lora(model: nn.Module, path: str) -> dict:
    """Load LoRA adapter weights from *path* into *model*.

    If the model has no adapters yet, ``model.add_lora_adapters()`` is
    called automatically using the metadata embedded in the file.

    Args:
        model: GrimoireTransformer (with or without adapters already applied).
        path: Path to a ``.lora`` file written by ``save_lora``.

    Returns:
        The raw payload dict (keys: "rank", "alpha", "targets", "state_dict").
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)

    has_lora = any(isinstance(m, LoRALinear) for m in model.modules())
    if not has_lora:
        model.add_lora_adapters(
            rank=payload["rank"],
            alpha=payload["alpha"],
            targets=payload["targets"],
        )

    _, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected LoRA keys in state_dict: {unexpected}")
    return payload
