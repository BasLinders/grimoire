"""Tests for the ConversationState -> gr.Chatbot messages adapter.

No ``pytest.importorskip("gradio")`` needed here -- ``chat_adapter.py`` has
no Gradio import, deliberately, so it's testable without the ``ui`` extra
installed at all.
"""

from grimoire_ai.state.conversation import Turn
from grimoire_ai.ui.chat_adapter import history_to_messages


class TestHistoryToMessages:
    def test_empty_history_returns_empty_list(self):
        assert history_to_messages([]) == []

    def test_single_turn_produces_two_ordered_messages(self):
        history = [Turn(user="hello", assistant="hi there")]
        messages = history_to_messages(history)
        assert messages == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]

    def test_multi_turn_ordering_preserved_oldest_first(self):
        history = [
            Turn(user="q1", assistant="a1"),
            Turn(user="q2", assistant="a2"),
            Turn(user="q3", assistant="a3"),
        ]
        messages = history_to_messages(history)
        assert [m["content"] for m in messages] == ["q1", "a1", "q2", "a2", "q3", "a3"]
        assert [m["role"] for m in messages] == [
            "user", "assistant", "user", "assistant", "user", "assistant",
        ]

    def test_content_passed_through_unmodified(self):
        """No markdown escaping or transformation -- gr.Chatbot's own
        render_markdown handles rendering, this adapter just relays text."""
        history = [Turn(user="**bold** <script>", assistant="`code` & stuff")]
        messages = history_to_messages(history)
        assert messages[0]["content"] == "**bold** <script>"
        assert messages[1]["content"] == "`code` & stuff"

    def test_agent_key_and_routing_score_are_not_leaked_into_messages(self):
        """Turn carries extra routing metadata the Chatbot UI has no use
        for -- confirm the adapter only extracts user/assistant text."""
        history = [Turn(user="q", assistant="a", agent_key="rules-lawyer", routing_score=0.42)]
        messages = history_to_messages(history)
        assert messages == [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
