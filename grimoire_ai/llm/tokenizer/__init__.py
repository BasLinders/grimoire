"""Grimoire LLM tokenizer package.

Public surface
--------------
BytePairEncoder
    The BPE tokenizer.  Train it on domain corpora, persist it with
    ``save``, reload it with ``load``, then use ``encode`` / ``decode``.

Special token constants (ids and string forms)
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN, USR_TOKEN, AST_TOKEN
    PAD_ID, BOS_ID, EOS_ID, SEP_ID, USR_ID, AST_ID
    ALL_SPECIAL_TOKENS, SPECIAL_TOKEN_TO_ID
"""

from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
from grimoire_ai.llm.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    AST_ID,
    AST_TOKEN,
    BOS_ID,
    BOS_TOKEN,
    EOS_ID,
    EOS_TOKEN,
    PAD_ID,
    PAD_TOKEN,
    SEP_ID,
    SEP_TOKEN,
    SPECIAL_TOKEN_TO_ID,
    USR_ID,
    USR_TOKEN,
)

__all__ = [
    "BytePairEncoder",
    "ALL_SPECIAL_TOKENS",
    "SPECIAL_TOKEN_TO_ID",
    "PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "SEP_TOKEN", "USR_TOKEN", "AST_TOKEN",
    "PAD_ID", "BOS_ID", "EOS_ID", "SEP_ID", "USR_ID", "AST_ID",
]
