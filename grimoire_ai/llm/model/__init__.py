"""Grimoire transformer model package.

Public surface
--------------
GrimoireTransformer
    The full decoder-only language model.

TransformerConfig
    Hyperparameter dataclass.  Construct one and pass it to
    ``GrimoireTransformer`` to build a model.
"""

from grimoire_ai.llm.model.config import TransformerConfig
from grimoire_ai.llm.model.lora import LoRALinear, load_lora, save_lora
from grimoire_ai.llm.model.transformer import GrimoireTransformer

__all__ = [
    "GrimoireTransformer",
    "TransformerConfig",
    "LoRALinear",
    "save_lora",
    "load_lora",
]
