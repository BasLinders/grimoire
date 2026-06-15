"""Agent routing: score queries against domain corpora and dispatch to the best agent.

``AgentRouter`` scores a query against each agent's lexical corpus using a
single top-1 Jaccard retrieval call per agent.  The highest-scoring agent
wins; if no agent clears the confidence threshold the default agent handles
the query.

``MultiAgentEngine`` wraps a single ``InferenceEngine`` and routes each turn
by swapping the active LoRA adapter and corpus in-place.  Because the base
model weights are shared across all agents, switching costs only the time to
copy a small LoRA weight tensor — typically sub-second.

Usage
-----
    registry = AgentRegistry("agents.json")
    engine = registry.build_multi_agent_engine(threshold=0.05)

    state = ConversationState()
    for partial in engine.chat_stream("What AC does plate armour give?", state):
        print(partial, end="", flush=True)

    key, score = engine.last_route   # e.g. ("saga", 0.18)
    print(state.routing_log)         # [(0, "saga", 0.18)]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from grimoire_ai.llm.training.checkpoint import load_checkpoint

if TYPE_CHECKING:
    from grimoire_ai.corpus.corpus import GrimoireCorpus
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.state.conversation import ConversationState


class AgentRouter:
    """Routes a query to the best-matching agent by lexical corpus score.

    Scores the query against each agent's ``GrimoireCorpus`` (top-1 Jaccard)
    and returns the key of the highest-scoring agent.  Agents without a corpus
    are excluded from scoring and can only win as the ``default_key`` fallback.

    Args:
        corpora: Mapping of agent key → ``GrimoireCorpus`` for every
            corpus-bearing agent.
        default_key: Agent key returned when no agent exceeds the threshold.
        threshold: Minimum score required to accept a routing decision.
            The score must strictly exceed this value; ties fall back to the default.
    """

    def __init__(
        self,
        corpora: dict[str, "GrimoireCorpus"],
        default_key: str,
        threshold: float = 0.05,
    ) -> None:
        self._corpora = corpora
        self._default_key = default_key
        self._threshold = threshold

    @property
    def default_key(self) -> str:
        return self._default_key

    def route(self, query: str) -> tuple[str, float]:
        """Return ``(agent_key, score)`` for the best-matching agent.

        Args:
            query: The user's query string.

        Returns:
            ``(agent_key, score)`` where ``score`` is the top-1 Jaccard
            similarity from the winning corpus, or ``0.0`` if the query fell
            back to the default.
        """
        best_key = self._default_key
        best_score = 0.0
        for key, corpus in self._corpora.items():
            results = corpus.query(query, top_k=1)
            if results and results[0].score > best_score:
                best_score = results[0].score
                best_key = key
        if best_score <= self._threshold:
            return self._default_key, best_score
        return best_key, best_score


class _RoutingStateWrapper:
    """Proxy that injects routing metadata when ``add_turn`` is called.

    Forwards all attribute access to the wrapped ``ConversationState`` except
    ``add_turn``, which is intercepted to attach the routing decision.
    """

    def __init__(
        self,
        state: "ConversationState",
        agent_key: str,
        routing_score: float,
    ) -> None:
        self._state = state
        self._agent_key = agent_key
        self._routing_score = routing_score

    def add_turn(self, user: str, assistant: str) -> None:
        self._state.add_turn(
            user,
            assistant,
            agent_key=self._agent_key,
            routing_score=self._routing_score,
        )

    def __getattr__(self, name: str):
        return getattr(self._state, name)


class MultiAgentEngine:
    """Single ``InferenceEngine`` that routes each turn to the best agent.

    On each ``chat_stream`` / ``chat`` call:
    1. ``AgentRouter.route(query)`` picks the best agent.
    2. If the agent changed, the active LoRA adapter and corpus are swapped.
    3. The query is forwarded to the underlying engine.

    The engine is always kept in a consistent state: ``model``, ``corpus``,
    and ``last_route`` are updated atomically before generation starts.

    Args:
        engine: Shared ``InferenceEngine`` built from the default agent's
            checkpoint with no LoRA active.
        router: ``AgentRouter`` instance for scoring queries.
        lora_paths: Mapping of agent key → resolved ``.lora`` file path.
            Agents not in this dict use base weights only.
        corpora: Mapping of agent key → ``GrimoireCorpus`` or ``None``.
        default_key: Registry key of the fallback agent.
    """

    def __init__(
        self,
        engine: "InferenceEngine",
        router: AgentRouter,
        lora_paths: dict[str, str],
        corpora: dict[str, Optional["GrimoireCorpus"]],
        default_key: str,
    ) -> None:
        self._engine = engine
        self._router = router
        self._lora_paths = lora_paths
        self._corpora = corpora
        self._default_key = default_key
        self._active_key: Optional[str] = None
        self.last_route: tuple[str, float] = ("", 0.0)

    def route(self, query: str) -> tuple[str, float]:
        """Score *query* and return ``(agent_key, score)``."""
        return self._router.route(query)

    def _switch_to(self, key: str) -> None:
        """Swap the active LoRA adapter and corpus to *key* if needed."""
        if key == self._active_key:
            return

        lora_path = self._lora_paths.get(key, "")
        if lora_path:
            # load_lora reloads base weights before applying the new adapter,
            # so it handles the LoRA-to-LoRA transition correctly.
            self._engine.load_lora(lora_path)
        else:
            # Target agent uses base weights only — unload any active LoRA.
            from grimoire_ai.llm.model.lora import LoRALinear
            if any(isinstance(m, LoRALinear) for m in self._engine.model.modules()):
                self._engine.model.merge_and_unload()
                ckpt = load_checkpoint(self._engine._checkpoint_path)
                self._engine.model.load_state_dict(ckpt["model"])
                self._engine.model.to(self._engine.device)
                self._engine.model.eval()

        self._engine.corpus = self._corpora.get(key)
        self._active_key = key

    def chat_stream(
        self,
        query: str,
        state: "ConversationState",
        gen_config: Optional["GenerationConfig"] = None,
    ):
        """Route *query*, switch agent if needed, then stream the response.

        Yields partial response strings exactly as ``InferenceEngine.chat_stream``
        does.  The routing decision is recorded on *state* when generation
        completes (via ``state.add_turn``).
        """
        key, score = self.route(query)
        self.last_route = (key, score)
        self._switch_to(key)
        yield from self._engine.chat_stream(
            query,
            _RoutingStateWrapper(state, key, score),
            gen_config=gen_config,
        )

    def chat(
        self,
        query: str,
        state: "ConversationState",
        gen_config: Optional["GenerationConfig"] = None,
    ) -> str:
        """Route *query* and return the complete response string."""
        key, score = self.route(query)
        self.last_route = (key, score)
        self._switch_to(key)
        return self._engine.chat(
            query,
            _RoutingStateWrapper(state, key, score),
            gen_config=gen_config,
        )
