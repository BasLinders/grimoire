"""Reserved special tokens for the Grimoire LLM tokenizer.

Special tokens occupy the first six ids in the vocabulary so they can never
be produced by BPE merges (which start numbering base characters after id 5).
They are defined here as module-level constants so every component that needs
to inject or detect them imports from one authoritative place.

Token roles
-----------
PAD   Padding token.  Used by the collator to make all sequences in a batch
      the same length.  Ignored by the loss function.
BOS   Beginning-of-sequence.  Prepended to every prompt fed to the model.
EOS   End-of-sequence.  The model learns to emit this when it finishes a
      response; the sampler stops generation when it sees it.
SEP   Separator.  Delimits the corpus-context block from the conversation
      block inside a prompt (see ``grimoire.llm.inference.prompt``).
USR   User-turn marker.  Placed before every user utterance in the
      conversation history so the model knows who is speaking.
AST   Assistant-turn marker.  Placed before every assistant response;
      generation is conditioned to follow this token.
"""

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SEP_TOKEN = "<SEP>"
USR_TOKEN = "<USR>"
AST_TOKEN = "<AST>"

# Ordered list — position in this list equals the reserved token id.
ALL_SPECIAL_TOKENS: list[str] = [
    PAD_TOKEN,  # id 0
    BOS_TOKEN,  # id 1
    EOS_TOKEN,  # id 2
    SEP_TOKEN,  # id 3
    USR_TOKEN,  # id 4
    AST_TOKEN,  # id 5
]

PAD_ID: int = 0
BOS_ID: int = 1
EOS_ID: int = 2
SEP_ID: int = 3
USR_ID: int = 4
AST_ID: int = 5

SPECIAL_TOKEN_TO_ID: dict[str, int] = {
    tok: idx for idx, tok in enumerate(ALL_SPECIAL_TOKENS)
}
