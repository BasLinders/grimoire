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
    model_state_dict: Optional[dict] = None,
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
        "model":      model_state_dict if model_state_dict is not None else model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler is not None else None,
        "scaler":     scaler.state_dict() if scaler is not None else None,
        "train_loss": train_loss,
    }
    torch.save(checkpoint, str(p))


def resize_checkpoint_vocab(ckpt: dict, new_vocab_size: int) -> dict:
    """Grow a checkpoint's embedding / output_head weight to a larger vocab.

    Lets ``--resume`` work after the BPE vocabulary has been extended (see
    ``BytePairEncoder.extend``) instead of requiring a full retrain. Existing
    rows are copied verbatim — the ids they represent keep their learned
    meaning exactly — and the newly added rows are randomly initialised with
    the same ``std=0.02`` scheme ``GrimoireTransformer._init_weights`` uses,
    so they look like any other freshly-initialised weight to the optimizer.

    ``embedding._embed.weight`` and ``output_head.weight`` are weight-tied,
    so both keys are present in ``state_dict()`` and both are resized
    identically here (using the same new rows) to preserve that tie.

    Args:
        ckpt: A checkpoint dict as returned by ``load_checkpoint``.  Mutated
            in place and also returned for convenience.
        new_vocab_size: Target vocabulary size.  Must be greater than or
            equal to the checkpoint's current vocabulary size.

    Returns:
        ``ckpt``, with ``ckpt["model"]`` and ``ckpt["config"]`` updated to
        the new vocabulary size.  Returned unchanged if the sizes already
        match.

    Raises:
        ValueError: If ``new_vocab_size`` is smaller than the checkpoint's
            current vocabulary size — shrinking would discard learned rows
            and is not supported.
    """
    model_sd = ckpt["model"]
    vocab_keys = [k for k in ("embedding._embed.weight", "output_head.weight") if k in model_sd]
    if not vocab_keys:
        return ckpt

    old_weight = model_sd[vocab_keys[0]]
    old_vocab_size, d_model = old_weight.shape
    if new_vocab_size == old_vocab_size:
        return ckpt
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"Cannot resize checkpoint vocabulary from {old_vocab_size} down "
            f"to {new_vocab_size} — shrinking would discard learned rows. "
            "Use a vocab_size >= the checkpoint's existing size."
        )

    extra = torch.empty(new_vocab_size - old_vocab_size, d_model, dtype=old_weight.dtype)
    nn.init.normal_(extra, mean=0.0, std=0.02)
    new_weight = torch.cat([old_weight, extra], dim=0)
    for key in vocab_keys:
        model_sd[key] = new_weight.clone()

    cfg = ckpt.get("config")
    if isinstance(cfg, dict) and "vocab_size" in cfg:
        cfg["vocab_size"] = new_vocab_size

    return ckpt


def resize_optimizer_vocab_state(
    optimizer: Optimizer,
    embedding_weight: torch.Tensor,
    new_vocab_size: int,
) -> None:
    """Pad an AdamW optimizer's momentum buffers for a grown embedding.

    ``Optimizer.load_state_dict`` does not validate that restored state
    tensors (``exp_avg``, ``exp_avg_sq``) match the current parameter shape
    — the mismatch only surfaces as a runtime error on the first ``step()``.
    Call this right after loading optimizer state for a resumed checkpoint
    whose embedding was grown by ``resize_checkpoint_vocab``: it pads each
    momentum buffer with zero rows for the newly added vocabulary ids, which
    is the standard "no momentum yet" initial state for a fresh parameter.

    Args:
        optimizer: The optimizer, after ``load_state_dict`` has already run.
        embedding_weight: The model's current (already-resized) embedding
            weight tensor — used only as the dict key under which AdamW
            stores this parameter's state.
        new_vocab_size: The embedding's current row count.  Rows are padded
            up to this size; if the existing state already has this many
            rows (e.g. the parameter was never resized), this is a no-op.
    """
    state = optimizer.state.get(embedding_weight)
    if not state:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        buf = state.get(key)
        if buf is None or buf.shape[0] >= new_vocab_size:
            continue
        pad = torch.zeros(new_vocab_size - buf.shape[0], *buf.shape[1:], dtype=buf.dtype)
        state[key] = torch.cat([buf, pad], dim=0)


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
