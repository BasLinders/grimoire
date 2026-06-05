"""PromptBuilder: format a user query and optional corpus results into token ids.

This is the seam between the corpus retrieval layer and the LLM.  The output
of ``GrimoireCorpus.query`` — a list of ``QueryResult`` objects — is converted
into a context block that is prepended to the user's query before the model
generates a response.

Prompt format
-------------
The assembled prompt uses special tokens to mark the boundaries between the
context block, the user utterance, and the assistant turn:

    <BOS> <SEP> {context} <SEP> <USR> {query} <AST>

- ``<BOS>``  — standard sequence start.
- ``<SEP>``  — opens and closes the context block.
- ``<USR>``  — marks the start of the user utterance.
- ``<AST>``  — marks where the assistant should begin generating.

The model is then expected to generate tokens that follow ``<AST>`` until it
emits ``<EOS>`` or the sampler's ``max_new_tokens`` budget is exhausted.

Context encoding (Phase 4)
---------------------------
``QueryResult.next_token`` values are stemmed words (e.g. ``"grappl"``).
They are joined with spaces and tokenized as plain text.  This is a
lightweight but functional bridge: the retrieval signal informs generation
even though the surface form is slightly normalised.

Phase 5 enhancement: when the corpus stores original unstemmed excerpts in
``QueryResult``, pass them to ``PromptBuilder`` instead for richer context.

Budget management
-----------------
The context block is trimmed so that the full prompt (context + query +
special tokens) never exceeds ``max_context_tokens``.  Excess context tokens
are dropped from the right (lowest-scoring results are added last, so the
highest-scoring ones are retained when the budget is tight).
"""

from grimoire.corpus.corpus import QueryResult
from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    SEP_ID,
    USR_ID,
)

# Number of special tokens in a context-bearing prompt:
#   <BOS> <SEP> {context} <SEP> <USR> {query} <AST>
_CONTEXT_PROMPT_OVERHEAD = 5


class PromptBuilder:
    """Assemble a model-ready token-id sequence from a query and corpus results.

    Attributes:
        _tokenizer: Trained ``BytePairEncoder`` used to convert text to ids.
        max_context_tokens: Maximum number of token ids allowed for the entire
            prompt (context + query + special tokens).  Prompts that exceed
            this limit have their context block trimmed.
    """

    def __init__(
        self,
        tokenizer: BytePairEncoder,
        max_context_tokens: int = 512,
    ) -> None:
        """Initialise the builder.

        Args:
            tokenizer: A trained ``BytePairEncoder``.  Must have a vocabulary
                loaded (i.e. ``train`` or ``load`` has been called).
            max_context_tokens: Token budget for the full prompt.  Defaults to
                512, which fits comfortably inside the model's 1024-token
                ``max_seq_len`` and leaves room for the generated response.
        """
        self._tokenizer = tokenizer
        self.max_context_tokens = max_context_tokens

    def build(
        self,
        query: str,
        results: list[QueryResult] | None = None,
    ) -> list[int]:
        """Build the prompt token-id sequence.

        The prompt layout is::

            [BOS, SEP, *context_ids, SEP, USR, *query_ids, AST]

        When no corpus results are provided (or none contain a ``next_token``),
        the context block is omitted and the prompt becomes::

            [BOS, USR, *query_ids, AST]

        Args:
            query: Raw user query string.
            results: Optional list of ``QueryResult`` objects from
                ``GrimoireCorpus.query``.  Results are processed in order, so
                callers should pass them sorted by descending score (which is
                already the default for ``GrimoireCorpus.query``).

        Returns:
            A list of integer token ids ready to be fed to
            ``GrimoireTransformer.forward``.
        """
        query_ids = self._tokenizer.encode(query)

        # Build context text from corpus results.
        # Prefer the unstemmed excerpt when available (Phase 5+); fall back
        # to the stemmed next_token for corpora built without excerpt support.
        context_ids: list[int] = []
        if results:
            context_parts: list[str] = []
            for r in results:
                if r.excerpt:
                    context_parts.append(r.excerpt)
                elif r.next_token:
                    context_parts.append(r.next_token)
            if context_parts:
                context_text = " ".join(context_parts)
                context_ids = self._tokenizer.encode(context_text)

        # Trim the context to fit the budget alongside the query and the
        # five framing special tokens. If nothing survives the trim, drop the
        # context block entirely rather than emit a hollow <SEP><SEP> pair the
        # model never saw during training.
        budget = self.max_context_tokens - len(query_ids) - _CONTEXT_PROMPT_OVERHEAD
        context_ids = context_ids[:max(0, budget)]

        if context_ids:
            prompt = [BOS_ID, SEP_ID] + context_ids + [SEP_ID, USR_ID] + query_ids + [AST_ID]
        else:
            prompt = [BOS_ID, USR_ID] + query_ids + [AST_ID]

        return prompt
