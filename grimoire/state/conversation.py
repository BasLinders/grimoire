"""ConversationState: rolling multi-turn history for Grimoire sessions.

Each call to ``InferenceEngine.chat()`` adds a ``Turn`` to the state.  On the
next call, the state injects all stored turns into the prompt so the model can
see the full conversation history rather than treating every query as isolated.

Prompt layout (history present)
--------------------------------
    <BOS> [<SEP> {context} <SEP>] <USR> q1 <AST> a1
                                  <USR> q2 <AST> a2
                                  …
                                  <USR> current_query <AST>

Prompt layout (first turn, no history)
---------------------------------------
    <BOS> [<SEP> {context} <SEP>] <USR> current_query <AST>

This is identical to the single-turn format used during fine-tuning, so the
model handles the first turn of every session without any distribution shift.

Budget management
-----------------
The total prompt must not exceed ``max_seq_len``.  The priority order when
space is tight is:

  1. Current query — always included (the model must see what was asked)
  2. Conversation history — most recent turns first; oldest are dropped if
     they do not fit
  3. Corpus context — fills whatever budget remains after history

This prioritisation keeps conversation coherence (history) over factual
grounding (corpus) under pressure — an acceptable trade-off because recent
turns already carry domain vocabulary that the corpus would have provided.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from grimoire.llm.tokenizer.bpe import BytePairEncoder
from grimoire.llm.tokenizer.special_tokens import (
    AST_ID,
    BOS_ID,
    SEP_ID,
    USR_ID,
)


@dataclass
class Turn:
    """A single completed exchange between the user and the model.

    Attributes:
        user: The raw user query string.
        assistant: The model's response string.
    """

    user: str
    assistant: str


class ConversationState:
    """Stores and manages the rolling multi-turn history for one session.

    Instantiate one ``ConversationState`` per conversation (or per user
    session in a multi-user context).  Pass the same instance to every
    ``InferenceEngine.chat()`` call to maintain continuity.

    Attributes:
        max_turns: Hard cap on stored turns.  Once reached, the oldest turn
            is evicted to make room for the newest.
        _turns: Internal list of completed ``Turn`` objects, oldest first.
    """

    def __init__(self, max_turns: int = 20) -> None:
        """Initialise an empty conversation.

        Args:
            max_turns: Maximum number of past turns to keep.  Older turns
                are evicted when this limit is reached.  20 is a generous
                default — the token budget in ``build_prompt_ids`` will
                typically trim the injected history long before 20 turns
                of tokens exhaust the model's context window.
        """
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a completed exchange and evict the oldest if over the cap.

        Args:
            user: The user's query text.
            assistant: The model's response text.
        """
        self._turns.append(Turn(user=user, assistant=assistant))
        if len(self._turns) > self.max_turns:
            self._turns.pop(0)

    def clear(self) -> None:
        """Reset the conversation to an empty state."""
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        """Number of completed turns in the current history."""
        return len(self._turns)

    @property
    def history(self) -> list[Turn]:
        """Read-only copy of all stored turns, oldest first."""
        return list(self._turns)

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_prompt_ids(
        self,
        query: str,
        tokenizer: BytePairEncoder,
        context_ids: Optional[list[int]] = None,
        max_seq_len: int = 1024,
    ) -> list[int]:
        """Assemble the full multi-turn prompt as a token-id list.

        Layout::

            [BOS, (SEP, *ctx, SEP,)? *history_turns, USR, *query, AST]

        History turns are packed newest-first; the oldest turns are silently
        dropped when they do not fit within ``max_seq_len``.  Any remaining
        budget after history is allocated to the optional context block.

        Args:
            query: Current user query (plain text).
            tokenizer: Trained ``BytePairEncoder`` used for encoding.
            context_ids: Pre-encoded corpus context token ids.  When provided
                they are injected between ``<SEP>`` markers after ``<BOS>``,
                trimmed to whatever budget survives after history is packed.
                Pass ``None`` when no corpus is attached.
            max_seq_len: Maximum total tokens for the assembled prompt.
                Should match ``GrimoireTransformer.config.max_seq_len``.

        Returns:
            A list of integer token ids ready for ``generate()``.
        """
        query_ids = tokenizer.encode(query)
        current_segment = [USR_ID] + query_ids + [AST_ID]

        # BOS is always 1 token.
        overhead = 1 + len(current_segment)
        history_budget = max_seq_len - overhead

        # If context is present, SEP + ctx + SEP costs 2 + len(ctx) tokens.
        # Reserve that cost from the history budget so context can at least
        # partially survive.  We will trim ctx to whatever is left after
        # history is packed.
        if context_ids:
            # Tentatively reserve space for SEP tokens only; we will shrink
            # context if history eats into it.
            history_budget -= 2  # for the two SEP tokens

        # Pack history segments newest-first into available budget.
        packed: list[list[int]] = []
        used = 0
        for turn in reversed(self._turns):
            u_ids = tokenizer.encode(turn.user)
            a_ids = tokenizer.encode(turn.assistant)
            seg = [USR_ID] + u_ids + [AST_ID] + a_ids
            if used + len(seg) > history_budget:
                break  # oldest remaining turn does not fit; stop
            packed.insert(0, seg)
            used += len(seg)

        history_ids: list[int] = [tok for seg in packed for tok in seg]

        # Trim context to whatever budget remains after history.
        if context_ids:
            ctx_budget = max_seq_len - overhead - 2 - len(history_ids)
            context_ids = context_ids[: max(0, ctx_budget)]

        # Assemble the final prompt.
        if context_ids:
            prefix = [BOS_ID, SEP_ID] + context_ids + [SEP_ID]
        else:
            prefix = [BOS_ID]

        return prefix + history_ids + current_segment
