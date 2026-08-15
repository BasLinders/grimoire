"""Grimoire chat UI.

A dedicated, single-purpose Gradio app for talking to a loaded model.
Deliberately separate from ``grimoire_ai.ui.train_app`` (Preprocess/Pre-train/
Fine-tune/Scale/Evaluate/Ingest/Corpus) — chat needs to be unambiguous: the
transcript scrolling in the upper part of the screen, the input pinned at
the bottom, nothing else competing for attention. All the loading and
generation settings live in a collapsible sidebar instead of being stacked
above the conversation.

Message format
---------------
``gr.Chatbot`` on the installed Gradio version only accepts the OpenAI-style
messages format (``list[{"role": "user"/"assistant", "content": ...}]``) —
the older tuples format is fully removed. ``ConversationState`` (in
``grimoire_ai.state.conversation``) has no notion of this shape; the
adapter in ``grimoire_ai.ui.chat_adapter`` bridges the two.

Usage
-----
    python -m grimoire_ai.ui.chat_app
    # then open http://localhost:7861
"""

import json
import os
from pathlib import Path
from typing import Generator, Optional

import gradio as gr

from grimoire_ai.ui.chat_adapter import history_to_messages
from grimoire_ai.ui.shared import (
    _CSS,
    _ENCODER_CHOICES,
    _RERANKER_CHOICES,
    _THEME,
    _detect_device_profile,
    _index_is_fresh,
    _scan_files,
    _scan_subdirs,
    _semantic_index_dir,
    _semantic_index_dir_external,
    add_header,
)

# ---------------------------------------------------------------------------
# Dropdown refresh helper (chat-only -- train_app.py has its own subset)
# ---------------------------------------------------------------------------

def _refresh_lora_choices():
    return gr.update(choices=_scan_files("checkpoints/", "*.lora", recursive=True))


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

_AGENTS_JSON = "agents.json"


def _load_agent_names() -> list[str]:
    """Return agent choices for the dropdown.

    Prepends 'Auto-route' when the registry contains agents, so the router
    option is always at the top.
    """
    try:
        from grimoire_ai.agents.registry import AgentRegistry
        names = AgentRegistry(_AGENTS_JSON).display_names()
        return (["Auto-route"] + names) if names else []
    except (FileNotFoundError, ValueError):
        return []


def _preview_agent_config(display_name: str) -> tuple:
    """Pre-fill generation sliders and path fields when an agent is selected.

    Returns updates for:
      chat_routing_threshold, chat_temp, chat_top_k, chat_top_p,
      chat_tokens, chat_corpus_dir, chat_lora, chat_ckpt, chat_adaptive_temp
    """
    _no_change = (
        gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(),
    )

    is_auto_route = display_name == "Auto-route"

    # For Auto-route there is no single agent config to preview.
    if is_auto_route or not display_name:
        return (gr.update(visible=is_auto_route),) + _no_change

    try:
        from grimoire_ai.agents.registry import AgentRegistry
        cfg = AgentRegistry(_AGENTS_JSON).get_by_display_name(display_name)
    except Exception:
        return (gr.update(visible=False),) + _no_change

    gc = cfg.gen_config  # may be empty dict

    def _from_gc(key):
        return gr.update(value=gc[key]) if key in gc else gr.update()

    return (
        gr.update(visible=False),                                   # chat_routing_threshold
        _from_gc("temperature"),                                    # chat_temp
        _from_gc("top_k"),                                         # chat_top_k
        _from_gc("top_p"),                                         # chat_top_p
        _from_gc("max_new_tokens"),                                 # chat_tokens
        gr.update(value=cfg.corpus_dirs[0]) if cfg.corpus_dirs     # chat_corpus_dir
            else gr.update(),
        gr.update(value=cfg.lora_path or ""),                      # chat_lora
        gr.update(value=cfg.checkpoint),                           # chat_ckpt
        _from_gc("adaptive_temperature"),                           # chat_adaptive_temp
    )


def load_agent(
    display_name: str,
    encoder: str = "Model (decoder embeddings)",
    retrieval_threshold: Optional[float] = None,
    quantize: bool = False,
    math_tool_enabled: bool = False,
    routing_threshold: float = 0.05,
    stat_block_constraint_enabled: bool = False,
    reranker: str = "None",
    rerank_candidates: int = 20,
) -> tuple[object, object, str, str, str]:
    """Load an agent by display name, applying the chosen retrieval backend.

    When *display_name* is ``"Auto-route"`` a ``MultiAgentEngine`` is built
    that scores each query against every agent's corpus and routes to the
    best match.

    Returns (engine, conv_state, status, checkpoint_path, vocab_path).
    The last two values are passed back so the manual path fields reflect what
    was actually loaded.
    """
    from grimoire_ai.agents.registry import AgentRegistry
    from grimoire_ai.llm.inference.semantic import EXTERNAL_ENCODERS, SemanticRetriever, make_external_embed_fn
    from grimoire_ai.state.conversation import ConversationState

    registry = AgentRegistry(_AGENTS_JSON)

    # ---- Auto-route: build MultiAgentEngine --------------------------------
    if display_name == "Auto-route":
        try:
            engine = registry.build_multi_agent_engine(threshold=routing_threshold, quantize=quantize)
        except Exception as exc:
            return None, None, f"Failed to build router: {exc}", "", ""
        engine._engine.retrieval_threshold = retrieval_threshold
        if math_tool_enabled:
            from grimoire_ai.tools.math_tool import MathTool
            engine._engine.math_tool = MathTool()
        if stat_block_constraint_enabled:
            from grimoire_ai.llm.inference.constrained_decoding import StatBlockConstraint
            engine._engine.stat_block_constraint = StatBlockConstraint(engine._engine.tokenizer)
        if reranker != "None":
            from grimoire_ai.llm.inference.reranker import CROSS_ENCODER_MODELS, Reranker, make_cross_encoder_score_fn
            try:
                # Pass the engine's own resolved device explicitly rather
                # than letting the reranker auto-detect independently -- an
                # explicit device the engine was pinned to (e.g. "cpu" for
                # quantization) must not be silently overridden by the
                # reranker picking CUDA/MPS on its own.
                score_fn = make_cross_encoder_score_fn(CROSS_ENCODER_MODELS[reranker], device=engine._engine.device)
            except ImportError as exc:
                return None, None, str(exc), "", ""
            engine._engine.reranker = Reranker(score_fn)
            engine._engine.rerank_candidates = rerank_candidates
        default_cfg = registry.get(registry.default_key)
        n_agents = len(registry.keys())
        return (
            engine,
            ConversationState(),
            f"Auto-routing across {n_agents} agent(s). Default: '{default_cfg.display_name}'.",
            default_cfg.checkpoint,
            default_cfg.vocab,
        )

    cfg = registry.get_by_display_name(display_name)

    use_lexical  = encoder == "Lexical (Jaccard)"
    use_external = encoder in EXTERNAL_ENCODERS

    # build_engine always loads the lexical corpus from corpus_dirs.
    # For semantic / external we replace it afterwards.
    engine = registry.build_engine(cfg.key, quantize=quantize)
    engine.retrieval_threshold = retrieval_threshold
    if math_tool_enabled:
        from grimoire_ai.tools.math_tool import MathTool
        engine.math_tool = MathTool()
    if stat_block_constraint_enabled:
        from grimoire_ai.llm.inference.constrained_decoding import StatBlockConstraint
        engine.stat_block_constraint = StatBlockConstraint(engine.tokenizer)
    if reranker != "None":
        from grimoire_ai.llm.inference.reranker import CROSS_ENCODER_MODELS, Reranker, make_cross_encoder_score_fn
        try:
            # Pin the reranker to the engine's own resolved device -- see
            # the Auto-route branch above for why this can't be left to
            # independent auto-detection.
            score_fn = make_cross_encoder_score_fn(CROSS_ENCODER_MODELS[reranker], device=engine.device)
        except ImportError as exc:
            return None, None, str(exc), cfg.checkpoint, cfg.vocab
        engine.reranker = Reranker(score_fn)
        engine.rerank_candidates = rerank_candidates

    if not use_lexical and engine.corpus is not None:
        # Resolve via the registry so paths are correct regardless of cwd.
        resolved_dirs = [str(registry._resolve(d)) for d in cfg.corpus_dirs or []]
        documents: list[tuple[str, str]] = []
        for resolved_dir in resolved_dirs:
            for txt_file in sorted(Path(resolved_dir).glob("*.txt")):
                documents.append((txt_file.read_text(encoding="utf-8"), txt_file.stem))

        if documents:
            if use_external:
                try:
                    embed_fn = make_external_embed_fn(EXTERNAL_ENCODERS[encoder])
                except ImportError as exc:
                    return None, None, str(exc), cfg.checkpoint, cfg.vocab
                # Persistent index cache, same idiom as the model-embeddings
                # branch below but keyed by encoder (not checkpoint/LoRA --
                # an external encoder's embeddings don't depend on either).
                index_dir = _semantic_index_dir_external(resolved_dirs, EXTERNAL_ENCODERS[encoder])
                loaded_ok = False
                if index_dir and _index_is_fresh(index_dir, resolved_dirs, ""):
                    try:
                        retriever = SemanticRetriever.from_index(index_dir, embed_fn=embed_fn)
                        loaded_ok = retriever.size > 0
                    except Exception:
                        loaded_ok = False
                if not loaded_ok:
                    retriever = SemanticRetriever(embed_fn=embed_fn)
                    for text, source in documents:
                        retriever.add_text(text, source=source)
                    retriever.index()
                    if index_dir:
                        try:
                            from grimoire_ai.llm.inference.rag_index import RagIndex
                            hashes = RagIndex.compute_source_hashes(resolved_dirs, "", cache_dir=index_dir)
                            retriever.save_index(index_dir, source_hashes=hashes)
                        except Exception:
                            pass
                engine.corpus = retriever
            else:
                resolved_ckpt = str(registry._resolve(cfg.checkpoint))
                resolved_lora = str(registry._resolve(cfg.lora_path)) if cfg.lora_path else ""
                index_dir = _semantic_index_dir(resolved_dirs)
                loaded_ok = False
                if index_dir and _index_is_fresh(index_dir, resolved_dirs, resolved_ckpt, resolved_lora):
                    try:
                        engine.corpus = SemanticRetriever.from_index(index_dir, embed_fn=engine.embed)
                        loaded_ok = engine.corpus.size > 0
                    except Exception:
                        pass
                if not loaded_ok:
                    from grimoire_ai.llm.inference.rag_index import RagIndex
                    retriever = engine.build_semantic_corpus(documents)
                    if index_dir:
                        try:
                            hashes = RagIndex.compute_source_hashes(
                                resolved_dirs, resolved_ckpt, lora_path=resolved_lora or None,
                                cache_dir=index_dir,
                            )
                            retriever.save_index(index_dir, source_hashes=hashes)
                        except Exception:
                            pass

    state = ConversationState()
    return (
        engine,
        state,
        f"Agent '{cfg.display_name}' loaded ({encoder}).  {cfg.description}",
        cfg.checkpoint,
        cfg.vocab,
    )


def load_engine(
    checkpoint_path: str,
    vocab_path: str,
    corpus_dir: str = "",
    encoder: str = "Model (decoder embeddings)",
    retrieval_threshold: Optional[float] = None,
    quantize: bool = False,
    lora_path: str = "",
    math_tool_enabled: bool = False,
    stat_block_constraint_enabled: bool = False,
    reranker: str = "None",
    rerank_candidates: int = 20,
) -> tuple[object, object, str]:
    """Load an ``InferenceEngine`` and a fresh ``ConversationState``.

    When ``corpus_dir`` points at a directory of ``.txt`` files, the engine is
    grounded in that corpus using the retrieval backend selected by ``encoder``:

    - ``"Model (decoder embeddings)"``: the trained transformer's own
      mean-pooled hidden states — the native hybrid path.
    - ``"MiniLM (all-MiniLM-L6-v2)"`` / ``"MPNet (all-mpnet-base-v2)"``:
      a dedicated sentence-transformers encoder.  Downloads ~90 MB on first
      use; requires ``pip install -e ".[encoder]"``.
    - ``"Lexical (Jaccard)"``: stemmed n-gram index with Jaccard scoring —
      no neural embedding, CPU-only, instant startup.

    When ``corpus_dir`` is blank the engine runs ungrounded.
    """
    from pathlib import Path

    from grimoire_ai.corpus.corpus import GrimoireCorpus
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.llm.inference.semantic import EXTERNAL_ENCODERS, SemanticRetriever, make_external_embed_fn
    from grimoire_ai.state.conversation import ConversationState

    checkpoint_path = (checkpoint_path or "").strip()
    if not checkpoint_path:
        return None, None, "No checkpoint path specified.", gr.update()

    corpus_dir = (corpus_dir or "").strip()
    use_lexical = encoder == "Lexical (Jaccard)"
    use_external = encoder in EXTERNAL_ENCODERS

    documents: list[tuple[str, str]] = []
    lexical_corpus = None
    status_suffix = ""

    if corpus_dir:
        path = Path(corpus_dir)
        if not path.is_dir():
            return None, None, f"Corpus directory not found: {corpus_dir}", gr.update()
        for txt_file in sorted(path.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            documents.append((text, txt_file.stem))
            if use_lexical:
                if lexical_corpus is None:
                    lexical_corpus = GrimoireCorpus()
                lexical_corpus.add_text(text, source=txt_file.stem)
        if not documents:
            return None, None, f"No .txt files found in {corpus_dir}", gr.update()

    math_tool = None
    if math_tool_enabled:
        from grimoire_ai.tools.math_tool import MathTool
        math_tool = MathTool()

    lora_path = (lora_path or "").strip()
    # LoRA adapters are incompatible with int8-quantized engines; silently
    # disable quantization when a LoRA path is provided so load_lora() succeeds.
    if lora_path:
        quantize = False

    engine = InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
        corpus=lexical_corpus,
        retrieval_threshold=retrieval_threshold if corpus_dir else None,
        quantize=quantize,
        math_tool=math_tool,
    )
    if stat_block_constraint_enabled:
        from grimoire_ai.llm.inference.constrained_decoding import StatBlockConstraint
        engine.stat_block_constraint = StatBlockConstraint(engine.tokenizer)
    if reranker != "None":
        from grimoire_ai.llm.inference.reranker import CROSS_ENCODER_MODELS, Reranker, make_cross_encoder_score_fn
        try:
            # Pin the reranker to the engine's own resolved device -- see
            # load_agent's Auto-route branch for why this can't be left to
            # independent auto-detection.
            score_fn = make_cross_encoder_score_fn(CROSS_ENCODER_MODELS[reranker], device=engine.device)
        except ImportError as exc:
            return None, None, str(exc), gr.update()
        engine.reranker = Reranker(score_fn)
        engine.rerank_candidates = rerank_candidates

    if lora_path:
        try:
            engine.load_lora(lora_path)
        except Exception as exc:
            return None, None, f"Failed to load LoRA adapter: {exc}", gr.update()

    if corpus_dir and not use_lexical:
        if use_external:
            try:
                embed_fn = make_external_embed_fn(EXTERNAL_ENCODERS[encoder])
            except ImportError as e:
                return None, None, str(e), gr.update()
            # Persistent index cache, same idiom as the model-embeddings
            # branch below but keyed by encoder (not checkpoint/LoRA -- an
            # external encoder's embeddings don't depend on either).
            index_dir = _semantic_index_dir_external([corpus_dir], EXTERNAL_ENCODERS[encoder])
            loaded_ok = False
            if index_dir and _index_is_fresh(index_dir, [corpus_dir], ""):
                try:
                    retriever = SemanticRetriever.from_index(index_dir, embed_fn=embed_fn)
                    loaded_ok = retriever.size > 0
                except Exception:
                    loaded_ok = False
            if not loaded_ok:
                retriever = SemanticRetriever(embed_fn=embed_fn)
                for text, source in documents:
                    retriever.add_text(text, source=source)
                retriever.index()
                if index_dir:
                    try:
                        from grimoire_ai.llm.inference.rag_index import RagIndex
                        hashes = RagIndex.compute_source_hashes([corpus_dir], "", cache_dir=index_dir)
                        retriever.save_index(index_dir, source_hashes=hashes)
                    except Exception:
                        pass
            engine.corpus = retriever
            engine.retrieval_threshold = retrieval_threshold
            status_suffix = (
                f" | {encoder}: {retriever.size} passage(s) "
                f"from {len(documents)} file(s)"
            )
        else:
            # Default: model's own decoder embeddings — use persistent index when fresh.
            index_dir = _semantic_index_dir([corpus_dir])
            loaded_ok = False
            if index_dir and _index_is_fresh(index_dir, [corpus_dir], checkpoint_path, lora_path):
                try:
                    retriever = SemanticRetriever.from_index(index_dir, embed_fn=engine.embed)
                    engine.corpus = retriever
                    engine.retrieval_threshold = retrieval_threshold
                    loaded_ok = retriever.size > 0
                except Exception:
                    pass
            if not loaded_ok:
                from grimoire_ai.llm.inference.rag_index import RagIndex
                retriever = engine.build_semantic_corpus(documents)
                if index_dir:
                    try:
                        hashes = RagIndex.compute_source_hashes(
                            [corpus_dir], checkpoint_path, lora_path=lora_path or None,
                            cache_dir=index_dir,
                        )
                        retriever.save_index(index_dir, source_hashes=hashes)
                    except Exception:
                        pass
            else:
                retriever = engine.corpus
            status_suffix = (
                f" | model embeddings: {retriever.size} passage(s) "
                f"from {len(documents)} file(s)"
            )
    elif corpus_dir:
        status_suffix = f" | lexical (Jaccard): {len(documents)} file(s)"

    if lora_path:
        status_suffix += f" | LoRA adapter: {lora_path}"

    state = ConversationState()
    return engine, state, f"Model loaded from {checkpoint_path}{status_suffix}", quantize


# ---------------------------------------------------------------------------
# Query heuristics
# ---------------------------------------------------------------------------

_FACTUAL_PREFIXES = ("what", "how many", "list", "when", "define")
_CREATIVE_PREFIXES = ("write", "create", "design", "imagine")


def _starts_with_word(text: str, phrase: str) -> bool:
    """Return True when *text* starts with *phrase* as a whole word/phrase.

    Prevents prefix false-positives: "list" must not match "listen", "listless";
    "define" must not match "definitely"; "what" must not match "whatever".
    """
    if not text.startswith(phrase):
        return False
    rest = text[len(phrase):]
    return not rest or not rest[0].isalpha()


def _query_gen_hints(query: str, temperature: float, max_new_tokens: int, adaptive_temperature: bool):
    """Adjust temperature and max_new_tokens based on simple query heuristics.

    Temperature hints (skipped when adaptive_temperature is True):
      - Factual opener  → cap temperature at 0.5   (min — don't raise it if already lower)
      - Creative opener → floor temperature at 0.9  (max — don't lower it if already higher)

    Token-budget hints (word-count approximation for speed):
      - Short query < 10 words  → floor max_new_tokens at 256 (ensure room to answer)
      - Long query  > 50 words  → cap max_new_tokens at 128   (concise focused answer)
    """
    query = (query or "")
    q = query.strip().lower()
    word_count = len(q.split())

    if not adaptive_temperature:
        if any(_starts_with_word(q, p) for p in _FACTUAL_PREFIXES):
            temperature = min(temperature, 0.5)
        elif any(_starts_with_word(q, p) for p in _CREATIVE_PREFIXES):
            temperature = max(temperature, 0.9)

    if word_count < 10:
        max_new_tokens = max(max_new_tokens, 256)
    elif word_count > 50:
        max_new_tokens = min(max_new_tokens, 128)

    return temperature, max_new_tokens


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def chat(
    query: str,
    engine_state,
    conv_state,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    adaptive_temperature: bool = False,
    loop_guard_enabled: bool = False,
) -> Generator[tuple[list[dict], object, str, str], None, None]:
    """Stream a response token-by-token, updating the Chatbot transcript.

    Yields ``(messages, conv_state, routing_label, query_box_value)`` --
    ``messages`` is the full transcript so far (oldest-first, in
    ``gr.Chatbot``'s message-dict format), including the in-progress
    assistant reply. The query box is left unchanged (``gr.update()``)
    while streaming and cleared only on the final yield, once the turn is
    actually recorded in ``conv_state``.
    """
    if engine_state is None:
        messages = history_to_messages(conv_state.history) if conv_state is not None else []
        messages.append({"role": "assistant", "content": "No model loaded. Use the Load button first."})
        yield messages, conv_state, "", gr.update()
        return
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.state.conversation import ConversationState
    temperature, max_new_tokens = _query_gen_hints(
        query, temperature, max_new_tokens, adaptive_temperature
    )
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        adaptive_temperature=adaptive_temperature,
        # RepetitionLoopGuard defaults (max_repeats=3, max_period=4) — see
        # constrained_decoding.py. 0 disables; the checkbox only offers an
        # on/off toggle since these two knobs are tightly coupled and the
        # class's own defaults are already sensible for general chat.
        loop_guard_max_repeats=3 if loop_guard_enabled else 0,
    )
    if conv_state is None:
        conv_state = ConversationState()
    # Computed once, before the loop -- add_turn() only fires after
    # chat_stream() is fully exhausted, so this correctly excludes the
    # in-flight turn from every intermediate yield.
    base_messages = history_to_messages(conv_state.history)
    last_partial = ""
    for partial in engine_state.chat_stream(query, conv_state, gen_config=gen_config):
        last_partial = partial
        yield (
            base_messages + [
                {"role": "user", "content": query},
                {"role": "assistant", "content": partial},
            ],
            conv_state,
            gr.update(),  # preserve previous routing label during streaming
            gr.update(),  # leave the query box alone until the turn completes
        )
    routing_info = ""
    if hasattr(engine_state, "last_route") and engine_state.last_route[0]:
        key, score = engine_state.last_route
        routing_info = f"↳ {key}  ·  {score:.3f}"
    final_messages = base_messages + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": last_partial},
    ]
    yield final_messages, conv_state, routing_info, ""


def clear_conversation(conv_state) -> tuple[object, list]:
    """Reset the conversation history."""
    from grimoire_ai.state.conversation import ConversationState
    if conv_state is not None:
        conv_state.clear()
    else:
        conv_state = ConversationState()
    return conv_state, []


def stage_exchange(conv_state) -> tuple[str, str]:
    """Copy the most recent chat exchange into the editable staging fields.

    Reads the last recorded ``Turn`` directly rather than parsing the
    rendered ``Chatbot`` messages back out -- simpler, and immune to
    accidentally grabbing an in-flight partial reply mid-stream.
    """
    if conv_state is None or not conv_state.history:
        return "", ""
    last_turn = conv_state.history[-1]
    return last_turn.user, last_turn.assistant


# ---------------------------------------------------------------------------
# Dataset builder helpers
# ---------------------------------------------------------------------------

def _dataset_preview(pairs: list[dict]) -> str:
    """Render the last three saved pairs as readable text."""
    if not pairs:
        return ""
    lines = []
    for p in pairs[-3:]:
        lines.append(f"prompt:   {p['prompt']}")
        lines.append(f"response: {p['response']}")
        lines.append("")
    return "\n".join(lines).rstrip()


def add_to_dataset(
    prompt: str,
    response: str,
    pairs: list[dict],
) -> tuple[list[dict], str, str, str, str]:
    """Append the staged pair to the dataset and clear the staging fields."""
    prompt = prompt.strip()
    response = response.strip()
    if not prompt or not response:
        status = "Both prompt and response must be non-empty."
        return pairs, _dataset_label(pairs), _dataset_preview(pairs), prompt, response
    pairs = pairs + [{"prompt": prompt, "response": response}]
    return pairs, _dataset_label(pairs), _dataset_preview(pairs), "", ""


def remove_last_pair(pairs: list[dict]) -> tuple[list[dict], str, str]:
    """Remove the most recently added pair."""
    pairs = pairs[:-1] if pairs else pairs
    return pairs, _dataset_label(pairs), _dataset_preview(pairs)


def load_dataset(path: str) -> tuple[list[dict], str, str, str]:
    """Load an existing JSONL file into session state."""
    path = path.strip() or "data/finetune/conversations.jsonl"
    p = Path(path)
    if not p.exists():
        return [], _dataset_label([]), "", f"File not found: {path}"
    pairs = []
    skipped = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "prompt" in obj and "response" in obj:
                    pairs.append({"prompt": obj["prompt"], "response": obj["response"]})
                else:
                    skipped += 1
            except json.JSONDecodeError:
                skipped += 1
    msg = f"Loaded {len(pairs)} pair(s) from {p}"
    if skipped:
        msg += f" ({skipped} invalid line(s) skipped)"
    return pairs, _dataset_label(pairs), _dataset_preview(pairs), msg


def export_dataset(pairs: list[dict], path: str, overwrite: bool) -> str:
    """Append (or overwrite) pairs to a JSONL file."""
    if not pairs:
        return "Nothing to export — add some pairs first."
    path = path.strip() or "data/finetune/conversations.jsonl"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with out.open(mode, encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    action = "Overwrote" if overwrite else "Appended"
    return f"{action} {len(pairs)} pair(s) to {out}"


def clear_dataset() -> tuple[list, str, str]:
    """Wipe all saved pairs."""
    return [], _dataset_label([]), ""


def _dataset_label(pairs: list[dict]) -> str:
    n = len(pairs)
    return f"{n} pair{'s' if n != 1 else ''} saved"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_chat_app() -> gr.Blocks:
    """Assemble and return the chat Gradio Blocks app."""
    _ckpts_all    = _scan_files("checkpoints/", "*.pt", recursive=True)
    _corpus_dirs  = _scan_subdirs("data/corpus/")
    _lora_choices = _scan_files("checkpoints/", "*.lora", recursive=True)
    _dp = _detect_device_profile()

    with gr.Blocks(title="Grimoire Chat", fill_height=True) as app:
        add_header()

        engine_state = gr.State(value=None)
        conv_state   = gr.State(value=None)

        with gr.Sidebar(label="Chat settings", open=True):
            gr.Markdown(
                "Select a named agent **or** load any checkpoint manually. "
                "Conversation history is maintained automatically."
            )

            # ---- Shared retrieval config (used by both agent and manual load)
            with gr.Accordion("Retrieval configuration", open=True):
                chat_encoder = gr.Dropdown(
                    choices=_ENCODER_CHOICES,
                    value="Model (decoder embeddings)",
                    label="Embedding backend",
                    info="How corpus passages are matched to your query. "
                         "Model: the trained transformer's own representations — no extra install. "
                         "MiniLM / MPNet: dedicated sentence encoders, better quality early in training — requires pip install -e \".[encoder]\". "
                         "Lexical: fast word-overlap matching, no neural embedding.",
                )
                chat_threshold = gr.Slider(
                    minimum=-1.0, maximum=1.0, value=0.0, step=0.05,
                    label="Retrieval threshold",
                    info="Minimum similarity score for a passage to be injected as context. "
                         "Queries below this score are answered without grounding (pure-chat). "
                         "Cosine scores live in [-1, 1]; 0.0 is a good starting point.",
                )
                chat_reranker = gr.Dropdown(
                    choices=_RERANKER_CHOICES,
                    value="None",
                    label="Reranker",
                    info="Rescore retrieved passages with a cross-encoder before injecting "
                         "them — reads the query and each passage together, more accurate "
                         "than the first-stage similarity score alone. Requires "
                         "pip install -e \".[encoder]\". 'None' keeps first-stage ranking as-is.",
                )
                chat_rerank_candidates = gr.Slider(
                    minimum=5, maximum=50, value=20, step=5,
                    label="Rerank candidate pool",
                    info="How many passages the first-stage retriever fetches before the "
                         "reranker rescores and keeps the top results. Only used when a "
                         "reranker is selected above.",
                )

            # ---- Agent selector -----------------------------------------
            _agent_names = _load_agent_names()
            with gr.Group(visible=bool(_agent_names)) as agent_group:
                gr.Markdown("### Select agent")
                agent_dropdown = gr.Dropdown(
                    choices=_agent_names,
                    value=_agent_names[0] if _agent_names else None,
                    label="Agent",
                )
                agent_load_btn = gr.Button("Load agent")
                chat_routing_threshold = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.05, step=0.01,
                    label="Routing threshold",
                    info="Minimum corpus similarity score for the router to commit to a "
                         "specialist agent. Queries scoring below this fall back to the "
                         "default agent. Only used in Auto-route mode.",
                    visible=_agent_names[:1] == ["Auto-route"],
                )
                agent_status = gr.Textbox(label="Agent status", interactive=False)

            # ---- Manual load --------------------------------------------
            with gr.Accordion("Load checkpoint manually", open=not bool(_agent_names)):
                chat_ckpt = gr.Dropdown(
                    choices=_ckpts_all,
                    value="",
                    label="Checkpoint path (.pt)",
                    allow_custom_value=True,
                )
                chat_vocab = gr.Textbox(
                    label="Vocabulary path (.json)",
                    value="data/tokenizer/bpe.json",
                    info="Must be the same vocabulary used during training — mismatching produces garbled output.",
                )
                load_btn = gr.Button("Load model")
                chat_corpus_dir = gr.Dropdown(
                    choices=_corpus_dirs,
                    value=None,
                    label="Corpus directory (optional)",
                    allow_custom_value=True,
                    info="Directory of .txt files used to ground replies in your corpus. Leave blank for ungrounded chat.",
                )
                chat_quantize = gr.Checkbox(
                    label="int8 quantization",
                    value=_dp["quantize"],
                    info="Quantize Linear layers to int8 after loading. Cuts memory ~4× and speeds up CPU inference. Ignored on CUDA/MPS.",
                )
                chat_lora = gr.Dropdown(
                    choices=_lora_choices,
                    value=None,
                    label="LoRA adapter (.lora) — optional",
                    allow_custom_value=True,
                    info="Path to a .lora file produced by LoRA fine-tuning. Leave blank to use the base checkpoint.",
                )
                chat_math_tool = gr.Checkbox(
                    label="Enable math tool",
                    value=False,
                    info=(
                        "Detect arithmetic in queries (e.g. '25% of 1200', '3^4') and "
                        "inject the computed result as context before generation.  "
                        "Also resolves <TOOL:python>…</TOOL> tags from fine-tuned models."
                    ),
                )
                chat_stat_block_constraint = gr.Checkbox(
                    label="Enable stat-block constraint",
                    value=False,
                    info=(
                        "Restrict Challenge Rating / XP / AC / HP values to well-formed "
                        "continuations at decode time — structurally blocks a hallucinated "
                        "value (e.g. an invalid CR) rather than hoping the model got it "
                        "right. Decode-time only; does not affect prose generation "
                        "elsewhere in the reply."
                    ),
                )
                load_status = gr.Textbox(label="Status", interactive=False)

            # ---- Generation controls ------------------------------------
            with gr.Accordion("Generation controls", open=True):
                chat_adaptive_temp = gr.Checkbox(
                    value=False,
                    label="Adaptive temperature",
                    info="Recompute temperature at every token from the model's own confidence "
                         "(entropy of the next-token distribution). Confident steps get more "
                         "diversity, uncertain steps get more focus. When on, the manual "
                         "Temperature slider is ignored and hidden.",
                )
                chat_temp = gr.Slider(
                    0.1, 2.0, value=0.8, step=0.05, label="Temperature",
                    info="Randomness of replies. Low (0.3) = focused & predictable. High (1.2) = creative but may ramble.",
                )
                chat_top_k = gr.Slider(
                    1, 200, value=50, step=1, label="Top-k",
                    info="Only the K most likely next words are considered at each step. Lower = safer; higher = more variety.",
                )
                chat_top_p = gr.Slider(
                    0.1, 1.0, value=0.9, step=0.05, label="Top-p",
                    info="Cuts off unlikely words until the remaining options together reach this probability. Works alongside Top-k.",
                )
                chat_tokens = gr.Slider(
                    16, 512, value=128, step=8, label="Max new tokens",
                    info="Maximum length of the generated reply in word-pieces.",
                )
                chat_loop_guard = gr.Checkbox(
                    value=False,
                    label="Prevent repetition loops",
                    info="Hard-bans a token that would extend an already-established "
                         "repeating loop ('does does does...' or short-phrase loops), "
                         "instead of just discounting it like repetition_penalty does. "
                         "Structural backstop, decode-time only — see "
                         "docs/architecture_optimization.md item #5.",
                )

            # When adaptive temperature is enabled the manual slider is
            # meaningless (temperature is recomputed per token), so hide it.
            chat_adaptive_temp.change(
                fn=lambda on: gr.update(visible=not on),
                inputs=[chat_adaptive_temp],
                outputs=[chat_temp],
            )

            # ---- Dataset builder ----------------------------------------
            with gr.Accordion("Dataset builder — collect fine-tune pairs", open=False):
                gr.Markdown(
                    "After a good exchange, click **✦ Save this exchange** below the "
                    "chat. Edit the prompt or response below if needed, then click "
                    "**Add pair**. Export to JSONL when you have enough pairs."
                )
                ds_prompt   = gr.Textbox(label="Prompt", lines=3, interactive=True)
                ds_response = gr.Textbox(label="Response", lines=3, interactive=True)
                with gr.Row():
                    ds_add_btn    = gr.Button("Add pair", variant="primary")
                    ds_undo_btn   = gr.Button("Undo last", scale=0, min_width=100)
                    ds_clear_btn  = gr.Button("Clear all", scale=0, min_width=100)
                ds_count   = gr.Textbox(
                    label="Dataset", value="0 pairs saved", interactive=False,
                )
                ds_preview = gr.Textbox(
                    label="Preview (last 3 pairs)", lines=8, interactive=False
                )
                gr.Markdown("---")
                ds_path = gr.Textbox(
                    label="Dataset file path",
                    value="data/finetune/conversations.jsonl",
                    info="Shared by Load and Export. File is created automatically if it doesn't exist.",
                )
                with gr.Row():
                    ds_load_btn   = gr.Button("Load existing", scale=0, min_width=130)
                    ds_export_btn = gr.Button("Export JSONL", variant="primary", scale=0, min_width=130)
                ds_overwrite = gr.Checkbox(
                    label="Overwrite file on export",
                    value=False,
                    info="Unchecked (default): new pairs are appended to the existing file. Check this only when you want to replace the file completely.",
                )
                ds_status = gr.Textbox(label="Status", interactive=False)

            dataset_state = gr.State(value=[])

        # ---- Main column: transcript (upper) + pinned input (lower) -----
        # No fixed height on the Chatbot -- with fill_height=True on Blocks,
        # the main column is a flex container, so a bare `scale` lets the
        # Chatbot claim the leftover vertical space after the input rows
        # take what they need, instead of guessing a pixel/vh budget that
        # only happens to fit one specific viewport size. Verified in a
        # real browser at 1280x720: transcript fills the remaining space
        # and Send/Clear/Routed-to stay within the viewport, no scroll
        # needed to reach them.
        chatbot = gr.Chatbot(
            value=[],
            scale=1,
            autoscroll=True,
            render_markdown=True,
        )
        with gr.Row(scale=0):
            chat_query = gr.Textbox(
                placeholder="Ask Grimoire…", lines=2, max_lines=6,
                scale=8, show_label=False, autofocus=True,
            )
            chat_btn = gr.Button("Send", variant="primary", scale=1)
        with gr.Row(scale=0):
            stage_btn    = gr.Button("✦ Save this exchange", scale=0, min_width=160)
            clear_btn    = gr.Button("Clear conversation", scale=0)
            chat_routing = gr.Textbox(
                label="Routed to",
                interactive=False,
                visible=True,
                scale=0,
                min_width=220,
                info="Agent and score for the last turn (Auto-route mode only).",
            )

        # ---- Event wiring -------------------------------------------
        _preview_outputs = [
            chat_routing_threshold,
            chat_temp, chat_top_k, chat_top_p, chat_tokens,
            chat_corpus_dir, chat_lora, chat_ckpt,
            chat_adaptive_temp,
        ]
        agent_dropdown.change(
            fn=_preview_agent_config,
            inputs=[agent_dropdown],
            outputs=_preview_outputs,
        )
        app.load(
            fn=_preview_agent_config,
            inputs=[agent_dropdown],
            outputs=_preview_outputs,
        )
        # Chat used to refresh these dropdowns on `chat_tab.select()`; there is
        # no tab to select into anymore now that Chat is the whole app, so an
        # app.load() at startup does the equivalent job.
        app.load(fn=_refresh_lora_choices, outputs=[chat_lora])

        agent_load_btn.click(
            fn=load_agent,
            inputs=[agent_dropdown, chat_encoder, chat_threshold, chat_quantize, chat_math_tool, chat_routing_threshold, chat_stat_block_constraint, chat_reranker, chat_rerank_candidates],
            outputs=[engine_state, conv_state, agent_status, chat_ckpt, chat_vocab],
        ).then(fn=lambda: [], outputs=[chatbot])
        load_btn.click(
            fn=load_engine,
            inputs=[chat_ckpt, chat_vocab, chat_corpus_dir, chat_encoder, chat_threshold, chat_quantize, chat_lora, chat_math_tool, chat_stat_block_constraint, chat_reranker, chat_rerank_candidates],
            outputs=[engine_state, conv_state, load_status, chat_quantize],
        ).then(fn=lambda: [], outputs=[chatbot])

        _chat_inputs = [
            chat_query, engine_state, conv_state,
            chat_temp, chat_top_k, chat_top_p, chat_tokens,
            chat_adaptive_temp, chat_loop_guard,
        ]
        _chat_outputs = [chatbot, conv_state, chat_routing, chat_query]
        chat_btn.click(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs)
        chat_query.submit(fn=chat, inputs=_chat_inputs, outputs=_chat_outputs)

        clear_btn.click(
            fn=clear_conversation,
            inputs=[conv_state],
            outputs=[conv_state, chatbot],
        )
        stage_btn.click(
            fn=stage_exchange,
            inputs=[conv_state],
            outputs=[ds_prompt, ds_response],
        )
        ds_add_btn.click(
            fn=add_to_dataset,
            inputs=[ds_prompt, ds_response, dataset_state],
            outputs=[dataset_state, ds_count, ds_preview, ds_prompt, ds_response],
        )
        ds_undo_btn.click(
            fn=remove_last_pair,
            inputs=[dataset_state],
            outputs=[dataset_state, ds_count, ds_preview],
        )
        ds_clear_btn.click(
            fn=clear_dataset,
            inputs=[],
            outputs=[dataset_state, ds_count, ds_preview],
        )
        ds_load_btn.click(
            fn=load_dataset,
            inputs=[ds_path],
            outputs=[dataset_state, ds_count, ds_preview, ds_status],
        )
        ds_export_btn.click(
            fn=export_dataset,
            inputs=[dataset_state, ds_path, ds_overwrite],
            outputs=[ds_status],
        )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch_chat(
    share: bool = False,
    port: int = 7861,
    inbrowser: bool = True,
    server_name: str = "127.0.0.1",
) -> None:
    """Build and launch the chat Gradio app.

    Args:
        share: If ``True``, create a public Gradio tunnel (requires internet).
        port: Local port to serve the app on. Defaults to 7861 (one above
            the training/eval app's 7860) so both can run simultaneously.
        inbrowser: If ``True`` (default), open the default browser automatically
            when the server is ready.
        server_name: Host to bind to. Defaults to localhost-only; pass
            ``"0.0.0.0"`` to accept connections from outside the local
            machine (e.g. when running inside a Docker container).
    """
    build_chat_app().queue().launch(
        server_name=server_name, server_port=port, share=share,
        inbrowser=inbrowser, theme=_THEME, css=_CSS,
    )


def main() -> None:
    """Console-script entry point (``grimoire-chat-ui``).

    Host/port/inbrowser are read from environment variables, mirroring
    ``train_app.main()`` -- unset locally, useful for containerized
    deployments running both apps side by side.
    """
    inbrowser_env = os.environ.get("GRIMOIRE_CHAT_UI_INBROWSER", "1").strip().lower()
    launch_chat(
        server_name=os.environ.get("GRIMOIRE_CHAT_UI_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRIMOIRE_CHAT_UI_PORT", "7861")),
        inbrowser=inbrowser_env not in ("0", "false", "no"),
    )


if __name__ == "__main__":
    main()
