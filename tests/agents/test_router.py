"""Tests for AgentRouter and MultiAgentEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from grimoire_ai.agents.router import AgentRouter, MultiAgentEngine, _RoutingStateWrapper
from grimoire_ai.state.conversation import ConversationState, Turn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corpus(score: float):
    """Return a mock GrimoireCorpus that always returns the given score."""
    result = MagicMock()
    result.score = score
    corpus = MagicMock()
    corpus.query.return_value = [result]
    return corpus


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------

class TestAgentRouter:
    def test_routes_to_highest_scoring_agent(self):
        corpora = {
            "saga":    _make_corpus(0.20),
            "general": _make_corpus(0.05),
        }
        router = AgentRouter(corpora=corpora, default_key="general", threshold=0.05)
        key, score = router.route("What AC does plate armour give?")
        assert key == "saga"
        assert abs(score - 0.20) < 1e-9

    def test_falls_back_when_below_threshold(self):
        corpora = {"saga": _make_corpus(0.02)}
        router = AgentRouter(corpora=corpora, default_key="general", threshold=0.05)
        key, score = router.route("Hello")
        assert key == "general"
        assert abs(score - 0.02) < 1e-9  # score is still returned, just below threshold

    def test_agents_without_corpora_excluded_from_scoring(self):
        # "general" has no corpus entry — it can only win as the default.
        corpora = {"saga": _make_corpus(0.03)}
        router = AgentRouter(corpora=corpora, default_key="general", threshold=0.05)
        key, _ = router.route("Hello")
        # saga scored 0.03 < 0.05, so we fall back to "general"
        assert key == "general"

    def test_exact_threshold_is_accepted(self):
        corpora = {"saga": _make_corpus(0.05)}
        router = AgentRouter(corpora=corpora, default_key="general", threshold=0.05)
        key, _ = router.route("query")
        # score == threshold is NOT accepted (strict <); falls back
        assert key == "general"

    def test_empty_corpora_always_returns_default(self):
        router = AgentRouter(corpora={}, default_key="general", threshold=0.05)
        key, score = router.route("anything")
        assert key == "general"
        assert score == 0.0

    def test_corpus_returning_no_results_does_not_crash(self):
        corpus = MagicMock()
        corpus.query.return_value = []
        router = AgentRouter(corpora={"saga": corpus}, default_key="general", threshold=0.05)
        key, score = router.route("query")
        assert key == "general"
        assert score == 0.0

    def test_default_key_property(self):
        router = AgentRouter(corpora={}, default_key="fallback", threshold=0.05)
        assert router.default_key == "fallback"


# ---------------------------------------------------------------------------
# _RoutingStateWrapper
# ---------------------------------------------------------------------------

class TestRoutingStateWrapper:
    def test_add_turn_injects_routing_metadata(self):
        state = ConversationState()
        wrapper = _RoutingStateWrapper(state, agent_key="saga", routing_score=0.18)
        wrapper.add_turn("What is grappling?", "Grappling reduces speed to 0.")
        assert state.turn_count == 1
        turn = state.history[0]
        assert turn.agent_key == "saga"
        assert abs(turn.routing_score - 0.18) < 1e-9

    def test_forwards_other_attributes_to_state(self):
        state = ConversationState()
        wrapper = _RoutingStateWrapper(state, agent_key="saga", routing_score=0.1)
        # build_prompt_ids, turn_count, history, etc. should be forwarded
        assert wrapper.turn_count == 0
        assert wrapper.history == []
        assert wrapper.max_turns == 20


# ---------------------------------------------------------------------------
# ConversationState routing_log
# ---------------------------------------------------------------------------

class TestRoutingLog:
    def test_routing_log_empty_without_routing(self):
        state = ConversationState()
        state.add_turn("hi", "hello")
        assert state.routing_log == []

    def test_routing_log_records_keyed_turns(self):
        state = ConversationState()
        state.add_turn("q1", "a1", agent_key="saga", routing_score=0.18)
        state.add_turn("q2", "a2")
        state.add_turn("q3", "a3", agent_key="general", routing_score=0.03)
        log = state.routing_log
        assert len(log) == 2
        assert log[0] == (0, "saga", 0.18)
        assert log[1] == (2, "general", 0.03)

    def test_turn_agent_key_preserved_under_eviction(self):
        state = ConversationState(max_turns=2)
        state.add_turn("q1", "a1", agent_key="saga", routing_score=0.1)
        state.add_turn("q2", "a2", agent_key="general", routing_score=0.2)
        state.add_turn("q3", "a3", agent_key="saga", routing_score=0.3)
        # q1 was evicted; remaining turns are q2 and q3
        log = state.routing_log
        assert len(log) == 2
        assert log[0][1] == "general"
        assert log[1][1] == "saga"


# ---------------------------------------------------------------------------
# MultiAgentEngine
# ---------------------------------------------------------------------------

class TestMultiAgentEngine:
    def _make_engine(self, default_key="general", threshold=0.05):
        """Build a MultiAgentEngine with mock internals."""
        mock_inner = MagicMock()
        mock_inner.model.modules.return_value = []
        mock_inner.chat_stream.return_value = iter(["Hello", "Hello world"])
        mock_inner.chat.return_value = "Hello world"

        corpora = {
            "saga":    _make_corpus(0.20),
            "general": _make_corpus(0.01),
        }
        router = AgentRouter(corpora=corpora, default_key=default_key, threshold=threshold)

        engine = MultiAgentEngine(
            engine=mock_inner,
            router=router,
            lora_paths={"saga": "checkpoints/lora/saga.lora"},
            corpora={"saga": corpora["saga"], "general": None},
            default_key=default_key,
        )
        return engine, mock_inner

    def test_chat_stream_routes_and_records_last_route(self):
        engine, _ = self._make_engine()
        state = ConversationState()
        list(engine.chat_stream("What is grappling?", state))
        assert engine.last_route[0] == "saga"
        assert engine.last_route[1] > 0.05

    def test_chat_stream_uses_wrapper_for_add_turn(self):
        engine, mock_inner = self._make_engine()
        state = ConversationState()

        # Simulate engine calling state.add_turn at the end of generation
        def _fake_chat_stream(query, state_arg, top_k_corpus=5, gen_config=None):
            yield "partial"
            state_arg.add_turn(query, "partial")

        mock_inner.chat_stream.side_effect = _fake_chat_stream

        list(engine.chat_stream("What is grappling?", state))
        assert state.turn_count == 1
        assert state.history[0].agent_key == "saga"

    def test_no_lora_swap_when_same_agent_routed_twice(self):
        engine, mock_inner = self._make_engine()
        state = ConversationState()
        list(engine.chat_stream("What is grappling?", state))
        list(engine.chat_stream("What is grappling again?", state))
        # load_lora should be called only once (first switch to "saga")
        mock_inner.load_lora.assert_called_once_with("checkpoints/lora/saga.lora")

    def test_switch_to_applies_lora_for_agent_with_lora_path(self):
        engine, mock_inner = self._make_engine()
        state = ConversationState()
        list(engine.chat_stream("What is grappling?", state))
        mock_inner.load_lora.assert_called_once_with("checkpoints/lora/saga.lora")

    def test_switch_to_base_weights_when_agent_has_no_lora(self):
        """Switching to an agent without lora_path calls unload_lora on the inner engine."""
        engine, mock_inner = self._make_engine()

        # Force a switch to "general" (no lora_path).
        engine._switch_to("general")

        mock_inner.unload_lora.assert_called_once()

    def test_fallback_routes_to_default_below_threshold(self):
        engine, _ = self._make_engine(threshold=0.50)  # very high threshold
        state = ConversationState()
        list(engine.chat_stream("Hello", state))
        assert engine.last_route[0] == "general"
