"""Grimoire transformer model package.

Public surface
--------------
GrimoireTransformer
    The full decoder-only language model.

TransformerConfig
    Hyperparameter dataclass.  Construct one and pass it to
    ``GrimoireTransformer`` to build a model.
"""

from grimoire.llm.model.config import TransformerConfig
from grimoire.llm.model.transformer import GrimoireTransformer

__all__ = ["GrimoireTransformer", "TransformerConfig"]
