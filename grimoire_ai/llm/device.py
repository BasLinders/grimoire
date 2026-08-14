"""Compute device auto-detection shared by training, inference, and eval.

Preference order: CUDA (NVIDIA) > MPS (Apple Silicon Metal) > CPU. This keeps
"runs on consumer hardware" as broad as possible while still picking up free
GPU acceleration on Apple Silicon Macs, which have no CUDA path at all.

CUDA-specific optimizations (AMP autocast, ``GradScaler``, ``torch.compile``,
cuDNN benchmark mode, pinned-memory DataLoaders) are intentionally NOT
extended to MPS — several of those (notably ``GradScaler``) are CUDA-only in
PyTorch, and MPS autocast support is comparatively immature. MPS still gets
its main win, which is running the forward/backward pass on the GPU instead
of the CPU; callers needing device-specific branching should keep comparing
against ``"cuda"`` explicitly rather than "not cpu".
"""

from typing import Optional

import torch


def select_device(preferred: Optional[str] = None) -> str:
    """Return the best available torch device string.

    Args:
        preferred: Explicit device override (``"cuda"``, ``"mps"``,
            ``"cpu"``, etc.) from a user/CLI flag. Returned as-is when set,
            so an explicit choice always wins over auto-detection.

    Returns:
        ``"cuda"`` if a CUDA GPU is available, otherwise ``"mps"`` if
        running on Apple Silicon with Metal support, otherwise ``"cpu"``.
    """
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def torch_has_triton() -> bool:
    """Best-effort check for a working Triton backend (private torch API).

    Shared by ``Trainer`` and ``EmbedTuner`` to decide whether to silence
    ``torch.compile``'s dynamo/inductor fallback warnings -- expected and
    harmless when Triton itself isn't installed (e.g. Windows), but worth
    leaving visible on a working Triton setup where a warning would signal a
    genuine compile regression.
    """
    try:
        from torch.utils._triton import has_triton

        return bool(has_triton())
    except Exception:
        return False
