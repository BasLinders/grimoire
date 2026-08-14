"""Adapter from ``ConversationState``'s ``Turn`` history to ``gr.Chatbot``'s
message format.

Deliberately has no Gradio import. The installed Gradio version (6.x) only
accepts the OpenAI-style messages format for ``gr.Chatbot`` --
``list[{"role": "user"/"assistant", "content": ...}]`` -- the older tuples
format (``list[tuple[str, str]]``) is fully removed. This module is the one
place that conversion happens, kept Gradio-free so it's testable in
isolation.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grimoire_ai.state.conversation import Turn


def history_to_messages(history: "list[Turn]") -> list[dict]:
    """Convert oldest-first ``Turn`` history into oldest-first Chatbot messages.

    Each ``Turn`` becomes two messages (user, then assistant) -- a 3-turn
    history produces 6 dicts. Content is passed through unmodified;
    ``gr.Chatbot``'s own ``render_markdown`` handles rendering.

    Args:
        history: ``ConversationState.history`` -- oldest-first list of
            ``Turn`` objects.

    Returns:
        Oldest-first list of ``{"role": "user"/"assistant", "content": str}``
        dicts, ready to pass as (or append to) a ``gr.Chatbot`` value.
    """
    messages: list[dict] = []
    for turn in history:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.assistant})
    return messages
