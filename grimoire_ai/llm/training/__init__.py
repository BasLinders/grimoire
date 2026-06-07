"""Training utilities for the GrimoireTransformer.

Public surface
--------------
Trainer
    Owns the training loop, optimizer, scheduler, and gradient scaler.
save_checkpoint / load_checkpoint
    Persist and restore model + optimizer state with config embedded.
"""

from grimoire_ai.llm.training.checkpoint import load_checkpoint, save_checkpoint
from grimoire_ai.llm.training.trainer import Trainer

__all__ = ["Trainer", "save_checkpoint", "load_checkpoint"]
