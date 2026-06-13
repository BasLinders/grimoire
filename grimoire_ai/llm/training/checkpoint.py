"""Checkpoint save and load utilities for the GrimoireTransformer.

A checkpoint is a single ``.pt`` file containing everything needed to
resume training or run inference:

    {
        "step":        int,                     current global step
        "config":      dict,                    TransformerConfig as dict
        "model":       OrderedDict,             model.state_dict()
        "optimizer":   dict,                    optimizer.state_dict()
        "scheduler":   dict,                    lr_scheduler.state_dict()
        "scaler":      dict | None,             GradScaler state (CUDA only)
        "train_loss":  float,                   most recent logged loss
    }

Embedding the config inside the checkpoint prevents version-mismatch
bugs: when you load a checkpoint you always rebuild the model with the
exact architecture it was trained with, regardless of what
``TransformerConfig`` defaults look like at load time.

Scaler state is stored when present so that mixed-precision training
resumes without a warm-up period for the loss scale.  It is ``None``
when training on CPU.
"""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optimizer,
    step: int,
    config_dict: dict,
    scheduler: Optional[LRScheduler] = None,
    train_loss: float = 0.0,
    scaler: Optional[Any] = None,
) -> None:
    """Save a training checkpoint to disk.

    Args:
        path: Destination ``.pt`` file path.  Parent directories are
            created automatically.
        model: The ``GrimoireTransformer`` being trained.
        optimizer: The ``AdamW`` optimizer instance.
        scheduler: The LR scheduler instance.
        step: Current global training step (number of optimizer updates).
        config_dict: The ``TransformerConfig`` serialised with
            ``config.to_dict()``.  Embedded so the model can be
            reconstructed from the checkpoint alone.
        train_loss: Most recent training loss value, stored for reference.
        scaler: Optional ``torch.cuda.amp.GradScaler`` instance.  Its
            state is saved when provided so that mixed-precision training
            can resume with the same loss scale.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict = {
        "step":       step,
        "config":     config_dict,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler is not None else None,
        "scaler":     scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
    }
    torch.save(checkpoint, str(p))


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint from disk.

    Args:
        path: Path to a ``.pt`` file written by ``save_checkpoint``.

    Returns:
        The raw checkpoint dict with keys ``"step"``, ``"config"``,
        ``"model"``, ``"optimizer"``, ``"scaler"``, and ``"train_loss"``.
        Use this dict to rebuild the model and optimizer before calling
        ``model.load_state_dict(ckpt["model"])`` etc.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    p = Path(path)
    if not path or not p.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False is required because checkpoints contain optimizer
    # state dicts which include Python objects beyond plain tensors.
    return torch.load(str(p), map_location="cpu", weights_only=False)
