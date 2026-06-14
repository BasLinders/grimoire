"""Grimoire training UI.

A five-tab Gradio web app for managing the full Grimoire workflow without
touching the CLI.

Tabs
----
Preprocess
    Train a BPE tokenizer on raw .txt files and write a binary corpus for
    pre-training.  Run this once before the Pre-train tab.

Pre-train
    Launch a pre-training run on a tokenised corpus binary.  Hyperparameters
    mirror the defaults in ``grimoire.llm.training.train``.  Loss is streamed
    live via the ``Trainer.on_log`` callback — no stdout polling.

Fine-tune
    Continue from a pre-trained checkpoint on a JSONL conversation dataset.
    Hyperparameters mirror ``grimoire.llm.training.finetune``.

Ingest
    Scrape and convert content into corpus ``.txt`` files.  Supports web
    URLs, PDF, DOCX, Markdown, plain text, and images (OCR).  Three cleaning
    presets control how aggressively boilerplate is stripped.

Chat
    Load any checkpoint and query the model interactively via
    ``InferenceEngine``.  Conversation history is maintained automatically.

Usage
-----
    python -m grimoire.ui.app
    # then open http://localhost:7860
"""

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Generator, Optional

import gradio as gr

# Per-tab stop events — replaced each time a new run starts.
_stop_events: dict[str, Optional[threading.Event]] = {
    "preprocess": None,
    "pretrain": None,
    "finetune": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_checkpoints(checkpoint_dir: str) -> list[str]:
    """Return sorted list of .pt files in ``checkpoint_dir``."""
    p = Path(checkpoint_dir)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.pt"))


def _fmt_elapsed(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _stream_training(train_fn) -> Generator[str, None, None]:
    """Run a training function in a background thread and stream loss lines.

    ``train_fn`` receives ``on_log``, ``on_save``, ``on_done``, and ``on_eval``
    callbacks.  Wraps ``_stream_task`` with a training-specific message
    formatter.  ``on_eval`` only fires when the run was given a validation set;
    runs without one simply never call it.
    """
    def _wrapped(on_progress):
        def on_log(step: int, loss: float, lr: float, elapsed: float) -> None:
            on_progress(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | {_fmt_elapsed(elapsed)}")

        def on_save(step: int, elapsed: float) -> None:
            on_progress(f"  ✔ checkpoint saved  step {step}  [{_fmt_elapsed(elapsed)}]")

        def on_done(step: int, elapsed: float) -> None:
            on_progress(f"\nTraining complete — {step} steps in {_fmt_elapsed(elapsed)}")

        def on_eval(step: int, val_loss: float, elapsed: float) -> None:
            on_progress(f"  ◆ eval  step {step:>6} | val loss {val_loss:.4f}  [{_fmt_elapsed(elapsed)}]")

        train_fn(on_log, on_save, on_done, on_eval)

    yield from _stream_task(_wrapped)


# ---------------------------------------------------------------------------
# Preprocess tab logic
# ---------------------------------------------------------------------------

def run_preprocess(
    input_dir: str,
    output_path: str,
    vocab_path: str,
    vocab_size: int,
) -> Generator[str, None, None]:
    """Train BPE tokenizer and write corpus binary; stream progress live."""
    from grimoire_ai.llm.data.preprocessing import preprocess

    stop_event = threading.Event()
    _stop_events["preprocess"] = stop_event

    def _task(on_progress):
        preprocess(
            input_path=input_dir.strip(),
            output_path=output_path.strip(),
            vocab_path=vocab_path.strip(),
            vocab_size=int(vocab_size),
            on_progress=on_progress,
        )

    yield from _wrap_with_buttons(_stream_task(_task))


def stop_preprocess() -> None:
    """Signal the running preprocess task to stop."""
    ev = _stop_events.get("preprocess")
    if ev is not None:
        ev.set()


# ---------------------------------------------------------------------------
# Pre-train tab logic
# ---------------------------------------------------------------------------

def apply_model_preset(preset_name: str):
    """Return field updates for the architectural param fields."""
    from grimoire_ai.llm.model.config import MODEL_PRESETS
    cfg = MODEL_PRESETS.get(preset_name)
    if cfg is None:
        return [gr.update()] * 5
    return [
        gr.update(value=cfg.d_model),
        gr.update(value=cfg.n_layers),
        gr.update(value=cfg.n_heads),
        gr.update(value=cfg.n_kv_heads),
        gr.update(value=cfg.d_ff),
    ]


def read_checkpoint_config(checkpoint_path: str) -> str:
    """Return a human-readable summary of the model config stored in a checkpoint.

    Used by the Fine-tune tab to show the architecture of the checkpoint being
    fine-tuned, so the user knows what model size they are working with.
    """
    path = checkpoint_path.strip() if checkpoint_path else ""
    if not path:
        return ""
    from pathlib import Path as _Path
    if not _Path(path).exists():
        return "Checkpoint not found."
    try:
        import torch as _torch
        ckpt = _torch.load(path, map_location="cpu", weights_only=True)
        cfg = ckpt.get("config", {})
        d_model   = cfg.get("d_model", "?")
        n_layers  = cfg.get("n_layers", "?")
        n_heads   = cfg.get("n_heads", "?")
        n_kv      = cfg.get("n_kv_heads", "?")
        d_ff      = cfg.get("d_ff", "?")
        vocab     = cfg.get("vocab_size", "?")
        max_seq   = cfg.get("max_seq_len", "?")
        step      = ckpt.get("step", "?")
        # Match to a known preset for a friendly label.
        from grimoire_ai.llm.model.config import MODEL_PRESETS
        label = next(
            (name for name, p in MODEL_PRESETS.items()
             if p.d_model == d_model and p.n_layers == n_layers),
            "custom",
        )
        return (
            f"Architecture: {label}  |  "
            f"d_model={d_model}  n_layers={n_layers}  n_heads={n_heads}  "
            f"n_kv_heads={n_kv}  d_ff={d_ff}  "
            f"vocab={vocab}  max_seq={max_seq}  |  "
            f"saved at step {step}"
        )
    except Exception as exc:
        return f"Could not read checkpoint: {exc}"


def run_pretrain(
    corpus_path: str,
    checkpoint_dir: str,
    resume_from: str,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    batch_size: int,
    accumulate_steps: int,
    log_every: int,
    save_every: int,
    val_split: float,
    eval_every: int,
    eval_batches: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
) -> Generator[str, None, None]:
    """Launch a pre-training run and stream log output.

    When ``val_split`` is greater than 0, the tail of the corpus is held out
    as a validation set and a validation loss is logged every ``eval_every``
    steps (averaged over at most ``eval_batches`` batches).  ``val_split = 0``
    disables evaluation and the run behaves exactly as before.
    """
    import torch
    from grimoire_ai.llm.model.config import TransformerConfig
    from grimoire_ai.llm.model.transformer import GrimoireTransformer
    from grimoire_ai.llm.training.train import _build_datasets
    from grimoire_ai.llm.training.trainer import Trainer

    stop_event = threading.Event()
    _stop_events["pretrain"] = stop_event
    resume = resume_from.strip() or None

    def _train(on_log, on_save, on_done, on_eval):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = TransformerConfig(
            d_model=int(d_model),
            n_layers=int(n_layers),
            n_heads=int(n_heads),
            n_kv_heads=int(n_kv_heads),
            d_ff=int(d_ff),
        )
        model = GrimoireTransformer(model_config)
        train_dataset, val_dataset = _build_datasets(
            corpus_path=corpus_path,
            val_corpus_path=None,
            val_split=float(val_split),
            seq_len=model_config.max_seq_len,
        )
        Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            batch_size=batch_size,
            accumulate_steps=accumulate_steps,
            log_every=log_every,
            save_every=save_every,
            eval_every=int(eval_every),
            eval_batches=int(eval_batches),
            checkpoint_dir=checkpoint_dir,
            device=device,
            on_log=on_log,
            on_save=on_save,
            on_done=on_done,
            on_eval=on_eval,
            stop_event=stop_event,
        ).train(resume_from=resume)

    yield from _wrap_with_buttons(_stream_training(_train))


def stop_pretrain() -> None:
    """Signal the running pre-training loop to stop after the current step."""
    ev = _stop_events.get("pretrain")
    if ev is not None:
        ev.set()


# ---------------------------------------------------------------------------
# Fine-tune tab logic
# ---------------------------------------------------------------------------

def run_finetune(
    pretrain_ckpt: str,
    resume_from: str,
    data_path: str,
    vocab_path: str,
    checkpoint_dir: str,
    total_steps: int,
    warmup_steps: int,
    peak_lr: float,
    batch_size: int,
    accumulate_steps: int,
    log_every: int,
    save_every: int,
    val_split: float,
    eval_every: int,
    eval_batches: int,
    max_seq_len: int,
) -> Generator[str, None, None]:
    """Launch a fine-tuning run and stream log output.

    When ``val_split`` is greater than 0, that fraction of the examples is
    randomly held out (seeded, no leakage) and a validation loss is logged
    every ``eval_every`` steps.  ``val_split = 0`` disables evaluation.
    """
    import torch
    from grimoire_ai.llm.data.conversation import ConversationDataset
    from grimoire_ai.llm.model.config import TransformerConfig
    from grimoire_ai.llm.model.transformer import GrimoireTransformer
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
    from grimoire_ai.llm.training.checkpoint import load_checkpoint
    from grimoire_ai.llm.training.finetune import split_dataset
    from grimoire_ai.llm.training.trainer import Trainer

    stop_event = threading.Event()
    _stop_events["finetune"] = stop_event
    resume = resume_from.strip() or None

    def _train(on_log, on_save, on_done, on_eval):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = load_checkpoint(pretrain_ckpt)
        config = TransformerConfig.from_dict(ckpt["config"])
        model = GrimoireTransformer(config)
        model.load_state_dict(ckpt["model"])
        tokenizer = BytePairEncoder.load(vocab_path)
        dataset = ConversationDataset(
            path=data_path,
            tokenizer=tokenizer,
            max_seq_len=min(max_seq_len, config.max_seq_len),
        )
        train_dataset, val_dataset = split_dataset(dataset, float(val_split))
        Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            batch_size=batch_size,
            accumulate_steps=accumulate_steps,
            log_every=log_every,
            save_every=save_every,
            eval_every=int(eval_every),
            eval_batches=int(eval_batches),
            checkpoint_dir=checkpoint_dir,
            device=device,
            on_log=on_log,
            on_save=on_save,
            on_done=on_done,
            on_eval=on_eval,
            stop_event=stop_event,
        ).train(resume_from=resume)

    yield from _wrap_with_buttons(_stream_training(_train))


def stop_finetune() -> None:
    """Signal the running fine-tuning loop to stop after the current step."""
    ev = _stop_events.get("finetune")
    if ev is not None:
        ev.set()


# ---------------------------------------------------------------------------
# Ingest tab logic
# ---------------------------------------------------------------------------

def _wrap_with_buttons(gen: Generator) -> Generator[tuple, None, None]:
    """Wrap a log-text generator to also yield start/stop button state.

    Yields ``(log_text, start_update, stop_update)`` triples so callers can
    wire the generator outputs to ``[log_box, start_btn, stop_btn]``.
    Disables the start button and enables the stop button while running,
    then reverses both when the generator is exhausted.
    """
    log_text = ""
    for log_text in gen:
        yield log_text, gr.update(interactive=False), gr.update(interactive=True)
    yield log_text, gr.update(interactive=True), gr.update(interactive=False)


def _stream_task(task_fn) -> Generator[str, None, None]:
    """Run ``task_fn(on_progress)`` in a background thread and stream messages.

    ``task_fn`` receives a single ``on_progress(msg: str)`` callback.
    Yields the full accumulated log so the Gradio Textbox always shows
    the complete history.
    """
    log_q: queue.Queue = queue.Queue()

    def on_progress(msg: str) -> None:
        log_q.put(msg)

    def _run() -> None:
        try:
            task_fn(on_progress)
        except Exception as exc:  # noqa: BLE001
            log_q.put(f"\nError: {exc}\n")
        finally:
            log_q.put(None)

    threading.Thread(target=_run, daemon=True).start()

    log_text = ""
    while True:
        try:
            msg = log_q.get(timeout=0.5)
        except queue.Empty:
            yield log_text
            continue

        if msg is None:
            log_text += "\nDone.\n"
            yield log_text
            break

        log_text += msg + "\n"
        yield log_text


def run_ingest(
    mode: str,
    url_or_dir: str,
    file_objs,
    output_dir: str,
    cleaning: str,
    recursive: bool,
    timeout: int,
) -> Generator[str, None, None]:
    """Run corpus ingestion and stream progress messages."""
    from grimoire_ai.corpus.ingest import CleaningLevel, ingest

    cleaning_level = CleaningLevel(cleaning)

    def _task(on_progress):
        if mode == "File":
            if not file_objs:
                raise ValueError("No file uploaded.")
            files = file_objs if isinstance(file_objs, list) else [file_objs]
            for f in files:
                on_progress(f"Ingesting {Path(f.name).name} ...")
                ingest(
                    source=f.name,
                    output_dir=output_dir.strip() or None,
                    recursive=False,
                    timeout=int(timeout),
                    cleaning=cleaning_level,
                    on_progress=on_progress,
                )
            return
        else:
            source = url_or_dir.strip()
            if not source:
                raise ValueError("Source path or URL is required.")

        ingest(
            source=source,
            output_dir=output_dir.strip() or None,
            recursive=recursive,
            timeout=int(timeout),
            cleaning=cleaning_level,
            on_progress=on_progress,
        )

    yield from _wrap_with_buttons(_stream_task(_task))


def _toggle_theme(current: str) -> tuple[str, str]:
    """Flip dark/light theme state and return the new button label."""
    new = "light" if current == "dark" else "dark"
    # Label always shows the OTHER mode (what the next click will do).
    label = "☀ Light mode" if new == "dark" else "🌙 Dark mode"
    return label, new


def _toggle_ingest_inputs(mode: str):
    """Show/hide source inputs depending on the selected mode."""
    return (
        gr.update(visible=mode in ("URL", "Directory")),  # url_or_dir textbox
        gr.update(visible=mode == "File"),                 # file upload
        gr.update(visible=mode == "Directory"),            # recursive checkbox
        gr.update(visible=mode == "URL"),                  # timeout input
    )


# ---------------------------------------------------------------------------
# Chat tab logic
# ---------------------------------------------------------------------------

_AGENTS_JSON = "agents.json"


def _semantic_cache_path(corpus_dirs: list[str], checkpoint: str) -> Optional[Path]:
    """Return the .pt cache path for this corpus/checkpoint pair, or None."""
    if not corpus_dirs:
        return None
    first_dir = Path(corpus_dirs[0])
    if not first_dir.is_dir():
        return None
    return first_dir / ".cache" / f"semantic_{Path(checkpoint).stem}.pt"


def _cache_is_fresh(cache_path: Path, corpus_dirs: list[str], checkpoint: str) -> bool:
    """Return True when *cache_path* is newer than all corpus .txt files and the checkpoint."""
    if not cache_path.is_file():
        return False
    cache_mtime = cache_path.stat().st_mtime
    ckpt = Path(checkpoint)
    if ckpt.is_file() and ckpt.stat().st_mtime > cache_mtime:
        return False
    for corpus_dir in corpus_dirs:
        d = Path(corpus_dir)
        if d.is_dir():
            for f in d.glob("*.txt"):
                if f.stat().st_mtime > cache_mtime:
                    return False
    return True


def _load_agent_names() -> list[str]:
    """Return display names from agents.json, or [] if the file is missing."""
    try:
        from grimoire_ai.agents.registry import AgentRegistry
        return AgentRegistry(_AGENTS_JSON).display_names()
    except (FileNotFoundError, ValueError):
        return []


def load_agent(
    display_name: str,
    encoder: str = "Model (decoder embeddings)",
    retrieval_threshold: Optional[float] = None,
    quantize: bool = False,
) -> tuple[object, object, str, str, str]:
    """Load an agent by display name, applying the chosen retrieval backend.

    Returns (engine, conv_state, status, checkpoint_path, vocab_path).
    The last two values are passed back so the manual path fields reflect what
    was actually loaded.
    """
    from grimoire_ai.agents.registry import AgentRegistry
    from grimoire_ai.llm.inference.semantic import EXTERNAL_ENCODERS, SemanticRetriever, make_external_embed_fn
    from grimoire_ai.state.conversation import ConversationState

    registry = AgentRegistry(_AGENTS_JSON)
    cfg = registry.get_by_display_name(display_name)

    use_lexical  = encoder == "Lexical (Jaccard)"
    use_external = encoder in EXTERNAL_ENCODERS

    # build_engine always loads the lexical corpus from corpus_dirs.
    # For semantic / external we replace it afterwards.
    engine = registry.build_engine(cfg.key, quantize=quantize)
    engine.retrieval_threshold = retrieval_threshold

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
                retriever = SemanticRetriever(embed_fn=embed_fn)
                for text, source in documents:
                    retriever.add_text(text, source=source)
                retriever.index()
                engine.corpus = retriever
            else:
                resolved_ckpt = str(registry._resolve(cfg.checkpoint))
                sem_cache = _semantic_cache_path(resolved_dirs, resolved_ckpt)
                loaded_ok = False
                if sem_cache and _cache_is_fresh(sem_cache, resolved_dirs, resolved_ckpt):
                    try:
                        engine.corpus = SemanticRetriever.from_cache(sem_cache, embed_fn=engine.embed)
                        loaded_ok = engine.corpus.size > 0
                    except Exception:
                        pass  # sem_cache still set — rebuild will overwrite the corrupt file
                if not loaded_ok:
                    retriever = engine.build_semantic_corpus(documents)
                    if sem_cache:
                        try:
                            retriever.save_cache(sem_cache)
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

    corpus_dir = (corpus_dir or "").strip()
    use_lexical = encoder == "Lexical (Jaccard)"
    use_external = encoder in EXTERNAL_ENCODERS

    documents: list[tuple[str, str]] = []
    lexical_corpus = None
    status_suffix = ""

    if corpus_dir:
        path = Path(corpus_dir)
        if not path.is_dir():
            return None, None, f"Corpus directory not found: {corpus_dir}"
        for txt_file in sorted(path.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            documents.append((text, txt_file.stem))
            if use_lexical:
                if lexical_corpus is None:
                    lexical_corpus = GrimoireCorpus()
                lexical_corpus.add_text(text, source=txt_file.stem)
        if not documents:
            return None, None, f"No .txt files found in {corpus_dir}"

    engine = InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
        corpus=lexical_corpus,
        retrieval_threshold=retrieval_threshold if corpus_dir else None,
        quantize=quantize,
    )

    if corpus_dir and not use_lexical:
        if use_external:
            try:
                embed_fn = make_external_embed_fn(EXTERNAL_ENCODERS[encoder])
            except ImportError as e:
                return None, None, str(e)
            retriever = SemanticRetriever(embed_fn=embed_fn)
            for text, source in documents:
                retriever.add_text(text, source=source)
            retriever.index()
            engine.corpus = retriever
            engine.retrieval_threshold = retrieval_threshold
            status_suffix = (
                f" | {encoder}: {retriever.size} passage(s) "
                f"from {len(documents)} file(s)"
            )
        else:
            # Default: model's own decoder embeddings — use cache when available.
            sem_cache = _semantic_cache_path([corpus_dir], checkpoint_path)
            loaded_ok = False
            if sem_cache and _cache_is_fresh(sem_cache, [corpus_dir], checkpoint_path):
                try:
                    retriever = SemanticRetriever.from_cache(sem_cache, embed_fn=engine.embed)
                    engine.corpus = retriever
                    engine.retrieval_threshold = retrieval_threshold
                    loaded_ok = retriever.size > 0
                except Exception:
                    pass  # sem_cache still set — rebuild will overwrite the corrupt file
            if not loaded_ok:
                retriever = engine.build_semantic_corpus(documents)
                if sem_cache:
                    try:
                        retriever.save_cache(sem_cache)
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

    state = ConversationState()
    return engine, state, f"Model loaded from {checkpoint_path}{status_suffix}"


def chat(
    query: str,
    engine_state,
    conv_state,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
    adaptive_temperature: bool = False,
) -> Generator[tuple[str, object], None, None]:
    """Stream a response token-by-token and update the conversation state."""
    if engine_state is None:
        yield "No model loaded. Use the Load button first.", conv_state
        return
    from grimoire_ai.llm.inference.sampler import GenerationConfig
    from grimoire_ai.state.conversation import ConversationState
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        adaptive_temperature=adaptive_temperature,
    )
    if conv_state is None:
        conv_state = ConversationState()
    for partial in engine_state.chat_stream(query, conv_state, gen_config=gen_config):
        yield partial, conv_state


# ---------------------------------------------------------------------------
# Scale calculator logic
# ---------------------------------------------------------------------------

def run_scale_calc(
    corpus_path: str,
    checkpoint_path: str,
    batch_size: int,
    accumulate_steps: int,
    seq_len: int,
    total_steps: int,
) -> str:
    """Compute Chinchilla scaling estimates and corpus statistics."""
    import math
    lines = []

    # --- Corpus token count ------------------------------------------------
    corpus_path = corpus_path.strip()
    corpus_tokens = None
    if corpus_path:
        try:
            import numpy as np
            # Corpus binaries are written as int32 by the preprocessing step
            # (see preprocessing.py / TokenizedDataset). Reading as any other
            # dtype miscounts tokens — uint16 would double the count.
            data = np.memmap(corpus_path, dtype=np.int32, mode="r")
            corpus_tokens = len(data)
            lines.append(f"Corpus tokens:       {corpus_tokens:>15,}")
        except Exception as exc:
            lines.append(f"Corpus read error:   {exc}")
    else:
        lines.append("Corpus tokens:       (no path provided)")

    # --- Model parameter count ---------------------------------------------
    n_params = None
    checkpoint_path = checkpoint_path.strip()
    if checkpoint_path:
        try:
            from grimoire_ai.llm.training.checkpoint import load_checkpoint
            from grimoire_ai.llm.model.config import TransformerConfig
            from grimoire_ai.llm.model.transformer import GrimoireTransformer
            ckpt = load_checkpoint(checkpoint_path)
            config = TransformerConfig.from_dict(ckpt["config"])
            model = GrimoireTransformer(config)
            n_params = model.num_parameters()
            lines.append(f"Model parameters:    {n_params:>15,}  (from checkpoint)")
        except Exception as exc:
            lines.append(f"Checkpoint error:    {exc}")
    if n_params is None:
        n_params = 25_000_000
        lines.append(f"Model parameters:    {n_params:>15,}  (default — load a checkpoint for exact count)")

    lines.append("")

    # --- Tokens per step ---------------------------------------------------
    tokens_per_step = int(batch_size) * int(accumulate_steps) * int(seq_len)
    lines.append(f"Tokens per step:     {tokens_per_step:>15,}  (batch {batch_size} × accum {accumulate_steps} × seq {seq_len})")

    # --- Chinchilla optimal ------------------------------------------------
    chinchilla_tokens = 20 * n_params
    chinchilla_steps  = chinchilla_tokens // tokens_per_step
    lines.append(f"Chinchilla-optimal tokens: {chinchilla_tokens:>10,}  (20 × parameters)")
    lines.append(f"Chinchilla-optimal steps:  {chinchilla_steps:>10,}  (at current batch config)")

    lines.append("")

    # --- Current run analysis ----------------------------------------------
    total_steps = int(total_steps)
    tokens_this_run = total_steps * tokens_per_step
    pct_chinchilla  = tokens_this_run / chinchilla_tokens * 100
    lines.append(f"Your total steps:    {total_steps:>15,}")
    lines.append(f"Tokens this run:     {tokens_this_run:>15,}  ({pct_chinchilla:.1f}% of Chinchilla optimal)")

    if corpus_tokens and corpus_tokens > 0:
        passes = tokens_this_run / corpus_tokens
        lines.append(f"Corpus passes:       {passes:>15.1f}x")
        lines.append("")
        if passes < 1.0:
            lines.append("⚠  Less than one full pass through the corpus.")
            lines.append("   The model won't have seen all your data. Increase total steps.")
        elif passes < 3.0:
            lines.append("✔  Good — 1–3 passes is healthy for a well-sized corpus.")
        elif passes < 10.0:
            lines.append("⚠  3–10 passes — risk of memorisation if corpus is small.")
            lines.append("   Consider expanding the corpus or reducing total steps.")
        else:
            lines.append("✘  More than 10 passes — the model is likely overfitting.")
            lines.append("   Expand the corpus significantly before training this long.")

    lines.append("")

    # --- Recommendation ----------------------------------------------------
    lines.append("─" * 52)
    if pct_chinchilla < 50:
        rec_steps = chinchilla_steps
        lines.append(f"Recommendation: significantly undertrained.")
        lines.append(f"  Target {rec_steps:,} steps for Chinchilla-optimal training.")
    elif pct_chinchilla < 80:
        lines.append(f"Recommendation: undertrained (~{pct_chinchilla:.0f}% of optimal).")
        lines.append(f"  Extending to {chinchilla_steps:,} steps would improve quality.")
    elif pct_chinchilla < 120:
        lines.append(f"Recommendation: well-trained ({pct_chinchilla:.0f}% of Chinchilla optimal). ✔")
    else:
        lines.append(f"Recommendation: over-budget ({pct_chinchilla:.0f}% of Chinchilla optimal).")
        lines.append(f"  Fine for quality, but diminishing returns beyond ~120%.")

    if corpus_tokens:
        # Suggest steps that give ~3 passes without exceeding Chinchilla budget
        steps_3_passes = math.ceil(corpus_tokens * 3 / tokens_per_step)
        lines.append("")
        lines.append(f"For 3 corpus passes:  {steps_3_passes:,} steps")
        lines.append(f"For Chinchilla opt.:  {chinchilla_steps:,} steps")
        lines.append(f"Use the larger of the two as your total_steps target.")

    return "\n".join(lines)


def clear_conversation(conv_state) -> tuple[object, str]:
    """Reset the conversation history."""
    from grimoire_ai.state.conversation import ConversationState
    if conv_state is not None:
        conv_state.clear()
    else:
        conv_state = ConversationState()
    return conv_state, ""


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


def stage_exchange(query: str, response: str) -> tuple[str, str]:
    """Copy the current chat exchange into the editable staging fields."""
    return query.strip(), response.strip()


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

_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Cinzel"), gr.themes.GoogleFont("Raleway"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Fira Code"), "monospace"],
).set(
    # Backgrounds
    body_background_fill="#0d0d14",
    body_background_fill_dark="#0d0d14",
    block_background_fill="#16161f",
    block_background_fill_dark="#16161f",
    input_background_fill="#1e1e2e",
    input_background_fill_dark="#1e1e2e",
    # Borders
    block_border_color="#2e2e45",
    block_border_color_dark="#2e2e45",
    input_border_color="#2e2e45",
    input_border_color_dark="#2e2e45",
    block_border_width="1px",
    # Text
    body_text_color="#c8c8d8",
    body_text_color_dark="#c8c8d8",
    block_title_text_color="#e8c97a",
    block_title_text_color_dark="#e8c97a",
    block_label_text_color="#9999bb",
    block_label_text_color_dark="#9999bb",
    input_placeholder_color="#8888aa",      # 4.8:1 on input bg #1e1e2e (WCAG AA ✓)
    input_placeholder_color_dark="#8888aa", # prevents Gradio dark block from overriding
    code_background_fill="#424268",         # purple chip, distinct from block bg; text 5.7:1 ✓
    code_background_fill_dark="#424268",
    # Buttons
    button_primary_background_fill="#b8860b",
    button_primary_background_fill_dark="#b8860b",
    button_primary_background_fill_hover="#d4a017",
    button_primary_background_fill_hover_dark="#d4a017",
    button_primary_text_color="#0d0d14",
    button_primary_text_color_dark="#0d0d14",
    button_secondary_background_fill="#1e1e2e",
    button_secondary_background_fill_dark="#1e1e2e",
    button_secondary_background_fill_hover="#2e2e45",
    button_secondary_background_fill_hover_dark="#2e2e45",
    button_secondary_text_color="#c8c8d8",
    button_secondary_text_color_dark="#c8c8d8",
    button_secondary_border_color="#2e2e45",
    button_secondary_border_color_dark="#2e2e45",
    # Shadows and radius
    block_shadow="0 0 18px 2px rgba(184,134,11,0.08)",
    block_radius="8px",
    input_radius="6px",
    button_large_radius="6px",
    button_small_radius="4px",
)

_CSS = """
/* ── Dark mode defaults ───────────────────────────────────────────────── */

/* Header row */
.grimoire-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 0 4px 0;
    border-bottom: 1px solid #2e2e45;
    margin-bottom: 12px;
}
.grimoire-header h1 {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #e8c97a 0%, #fffbe6 60%, #b8860b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: 0.08em;
}
/* Theme toggle button */
#theme-btn {
    background: transparent !important;
    border: 1px solid #2e2e45 !important;
    color: #9999bb !important;
    font-size: 0.78rem !important;
}
#theme-btn:hover {
    border-color: #b8860b !important;
    color: #e8c97a !important;
}
/* Tabs */
.tab-nav button {
    font-family: 'Cinzel', serif !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    color: #aaaacc !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.2s, border-color 0.2s;
}
.tab-nav button:hover {
    color: #c8a84b !important;
}
.tab-nav button.selected {
    color: #e8c97a !important;
    border-bottom: 2px solid #b8860b !important;
}
.tab-nav button:hover {
    color: #c8a84b !important;
}
/* Scrollable log boxes — green only in dark mode; light mode overridden by _LIGHT_CSS */
:root:not([data-theme="light"]) textarea {
    color: #a8e8a8 !important;
}
textarea {
    font-family: 'Fira Code', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.5 !important;
}
/* Placeholder text — dark mode */
input::placeholder,
textarea::placeholder {
    color: #8888aa !important;  /* 4.8:1 on #1e1e2e (WCAG AA ✓) */
    opacity: 1 !important;      /* Firefox reduces opacity by default */
}
/* Placeholder text — light mode. Uses attribute selector (higher specificity
   than the rule above) so it wins without depending on JS injection order. */
[data-theme="light"] input::placeholder,
[data-theme="light"] textarea::placeholder {
    color: #5c5c70 !important;  /* 5.4:1 on #eeeae0 (WCAG AA ✓) */
    opacity: 1 !important;
}
/* Primary CTA button text — explicit override so Gradio dark block can't revert to white */
button.primary {
    color: #0d0d14 !important;  /* 5.9:1 on amber #b8860b (WCAG AA ✓) */
}
/* Code highlight background — var() is updated by the JS toggle */
code, .code {
    background-color: var(--code-background-fill, #424268) !important;
    padding: 0.1em 0.35em !important;
    border-radius: 3px !important;
}
/* Sliders */
input[type=range]::-webkit-slider-thumb {
    background: #b8860b;
}
/* Stop buttons — #b08080 on #0d0d14 = 5.8:1 contrast (WCAG AA ✓) */
.stop-btn {
    background: transparent !important;
    border: 1px solid #4a2e2e !important;
    color: #b08080 !important;
}
.stop-btn:hover:not(:disabled) {
    border-color: #cc4444 !important;
    color: #ee6666 !important;
}
.stop-btn:disabled {
    opacity: 0.35 !important;
    cursor: not-allowed !important;
}
/* Shutdown button — #b08080 on #0d0d14 = 5.8:1 (WCAG AA ✓) */
#shutdown-btn {
    background: transparent !important;
    border: 1px solid #3a2a2a !important;
    color: #b08080 !important;
    font-size: 0.78rem !important;
}
#shutdown-btn:hover {
    border-color: #cc4444 !important;
    color: #ee6666 !important;
}
/* Button press feedback — scale + brightness dip gives a clear 'clicked' feel.
   Disabled buttons are excluded so they don't react at all. */
button:not(:disabled) {
    transition: transform 0.08s ease, filter 0.08s ease !important;
}
button:not(:disabled):active {
    transform: scale(0.95) !important;
    filter: brightness(0.80) !important;
}
/* Dropdown options list — dark mode fix.
   Gradio renders .options as a floating layer outside the block; it does not
   inherit the theme's input_background_fill, so it defaults to near-white
   and makes text invisible against the dark body.  Override explicitly. */
:root:not([data-theme="light"]) .options {
    background: #1e1e2e !important;
    border: 1px solid #2e2e45 !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.6) !important;
}
:root:not([data-theme="light"]) .options .item {
    color: #c8c8d8 !important;
    background: transparent !important;
}
:root:not([data-theme="light"]) .options .item:hover,
:root:not([data-theme="light"]) .options .item.selected {
    background: #2e2e45 !important;
    color: #e8c97a !important;
}
:root:not([data-theme="light"]) .options .item.active {
    background: #3a3a55 !important;
    color: #fffbe6 !important;
}
"""


# Light-mode CSS injected at runtime via JS — avoids Gradio specificity battles.
# All rules use !important so they beat Gradio's compiled theme stylesheet.
_LIGHT_CSS = (
    ".grimoire-header{border-bottom-color:#c8bfa8!important}"
    ".grimoire-header h1{background:linear-gradient(90deg,#6a4e00 0%,#b8860b 60%,#4a3000 100%)!important;"
    "-webkit-background-clip:text!important;-webkit-text-fill-color:transparent!important;"
    "background-clip:text!important}"
    "#theme-btn{border-color:#c8bfa8!important;color:#4a4a6a!important}"
    "#theme-btn:hover{border-color:#b8860b!important;color:#6a4e00!important}"
    ".tab-nav button{color:#4a4a6a!important}"
    ".tab-nav button.selected{color:#6a4e00!important;border-bottom:2px solid #b8860b!important}"
    ".tab-nav button:hover{color:#7a5800!important}"  # 5.9:1 on #f5f3ee (WCAG AA ✓)
    "textarea{color:#1a1a2e!important}"      # dark navy, no green in light mode; 15.4:1 ✓
    # Primary CTA button text — #0d0d14 on #b8860b = 5.9:1 ✓
    "button.primary{color:#0d0d14!important}"
    # Code highlight — #1a1a2e on #dcd8cc = 12.0:1 ✓; var() also updated via JS vars
    "code,.code{background-color:#dcd8cc!important}"
    # #884444 on #fffef9 = 7.6:1 (WCAG AA ✓)
    ".stop-btn{border:1px solid #d4a0a0!important;color:#884444!important}"
    ".stop-btn:hover:not(:disabled){border-color:#cc4444!important;color:#cc2222!important}"
    "#shutdown-btn{border-color:#d4a0a0!important;color:#884444!important}"
    "#shutdown-btn:hover{border-color:#cc4444!important;color:#cc2222!important}"
)

# CSS variables swapped by the JS toggle via style.setProperty(..., 'important').
# Using inline !important beats any !important in Gradio's theme <style> block.
_DARK_VARS = {
    "--body-background-fill": "#0d0d14",
    "--block-background-fill": "#16161f",
    "--input-background-fill": "#1e1e2e",
    "--block-border-color": "#2e2e45",
    "--input-border-color": "#2e2e45",
    "--body-text-color": "#c8c8d8",
    "--block-title-text-color": "#e8c97a",
    "--block-label-text-color": "#9999bb",
    "--input-placeholder-color": "#8888aa",
    "--code-background-fill": "#424268",
    "--button-primary-background-fill": "#b8860b",
    "--button-primary-background-fill-hover": "#d4a017",
    "--button-primary-text-color": "#0d0d14",
    "--button-secondary-background-fill": "#1e1e2e",
    "--button-secondary-background-fill-hover": "#2e2e45",
    "--button-secondary-text-color": "#c8c8d8",
    "--button-secondary-border-color": "#2e2e45",
}
_LIGHT_VARS = {
    "--body-background-fill": "#f5f3ee",
    "--block-background-fill": "#fffef9",
    "--input-background-fill": "#eeeae0",
    "--block-border-color": "#c8bfa8",
    "--input-border-color": "#c8bfa8",
    "--body-text-color": "#1a1a2e",
    "--block-title-text-color": "#6a4e00",
    "--block-label-text-color": "#4a4a6a",
    "--input-placeholder-color": "#5c5c70",  # 5.4:1 on #eeeae0 (WCAG AA ✓)
    "--code-background-fill": "#dcd8cc",     # 12.0:1 with #1a1a2e text (WCAG AA ✓)
    "--button-primary-background-fill": "#b8860b",
    "--button-primary-background-fill-hover": "#d4a017",
    "--button-primary-text-color": "#0d0d14",  # dark text on amber = 5.9:1 (WCAG AA ✓)
    "--button-secondary-background-fill": "#e8e4d8",
    "--button-secondary-background-fill-hover": "#d8d4c8",
    "--button-secondary-text-color": "#1a1a2e",
    "--button-secondary-border-color": "#c8bfa8",
}

def _vars_to_js(d: dict) -> str:
    """Serialise a {prop: value} dict as a JS object literal string."""
    return "{" + ",".join(f'"{k}":"{v}"' for k, v in d.items()) + "}"


def build_app() -> gr.Blocks:
    """Assemble and return the Gradio Blocks app."""
    with gr.Blocks(title="Grimoire") as app:
        theme_state = gr.State("dark")
        with gr.Row(elem_classes="grimoire-header"):
            gr.Markdown("# ✦ Grimoire")
            theme_btn = gr.Button(
                "☀ Light mode", scale=0, min_width=120, elem_id="theme-btn"
            )
            shutdown_btn = gr.Button(
                "⏻ Shut down", scale=0, min_width=110, elem_id="shutdown-btn"
            )
        # Shutdown: Python fn schedules os._exit on a background thread so
        # Gradio can return a clean response before the process dies.
        # JS first tries window.close() directly — this works in some browsers
        # when called from a click handler. If the browser blocks it, a 300 ms
        # fallback navigates to a goodbye page that contains its own "Close this
        # tab" button (user-initiated click makes window.close() reliable there).
        _SHUTDOWN_PAGE = (
            "data:text/html,"
            "%3Chtml%20style%3D%22background%3A%230d0d14%3Bcolor%3A%23e8c97a%3B"
            "font-family%3A%27Cinzel%27%2Cserif%3Bdisplay%3Aflex%3Balign-items%3A"
            "center%3Bjustify-content%3Acenter%3Bheight%3A100vh%3Bmargin%3A0%22%3E"
            "%3Cdiv%20style%3D%22text-align%3Acenter%22%3E"
            "%3Ch1%3E%E2%9C%A6%20Grimoire%3C%2Fh1%3E"
            "%3Cp%20style%3D%22color%3A%23c8c8d8%3Bmargin-bottom%3A1.5rem%22%3E"
            "The%20server%20has%20shut%20down.%3C%2Fp%3E"
            "%3Cbutton%20onclick%3D%22window.close()%22%20style%3D%22"
            "background%3Atransparent%3Bborder%3A1px%20solid%20%23b8860b%3B"
            "color%3A%23e8c97a%3Bfont-family%3Ainherit%3Bfont-size%3A0.9rem%3B"
            "padding%3A0.5rem%201.2rem%3Bborder-radius%3A4px%3Bcursor%3Apointer%22%3E"
            "Close%20this%20tab%3C%2Fbutton%3E"
            "%3C%2Fdiv%3E%3C%2Fhtml%3E"
        )

        def _request_shutdown():
            def _kill():
                time.sleep(1.0)
                os._exit(0)
            threading.Thread(target=_kill, daemon=True).start()

        shutdown_btn.click(
            fn=_request_shutdown,
            inputs=[],
            outputs=[],
            js=(
                f"() => {{"
                f"  window.close();"
                f"  setTimeout(() => window.location.replace('{_SHUTDOWN_PAGE}'), 300);"
                f"}}"
            ),
        )
        _JS_TOGGLE = f"""() => {{
            const root = document.documentElement;
            const isDark = root.getAttribute('data-theme') !== 'light';
            root.setAttribute('data-theme', isDark ? 'light' : 'dark');

            // Override Gradio theme CSS variables with inline !important,
            // which beats any !important in Gradio's compiled <style> block.
            const vars = isDark ? {_vars_to_js(_LIGHT_VARS)} : {_vars_to_js(_DARK_VARS)};
            for (const [p, v] of Object.entries(vars))
                root.style.setProperty(p, v, 'important');

            // Inject (or clear) the light-mode element-specific overrides.
            let el = document.getElementById('grimoire-theme-overrides');
            if (!el) {{
                el = document.createElement('style');
                el.id = 'grimoire-theme-overrides';
                document.head.appendChild(el);
            }}
            el.textContent = isDark ? `{_LIGHT_CSS}` : '';

            // Return pre-toggle state so Python updates the button label.
            return isDark ? 'dark' : 'light';
        }}"""
        theme_btn.click(
            fn=_toggle_theme,
            inputs=[theme_state],
            outputs=[theme_btn, theme_state],
            js=_JS_TOGGLE,
        )

        # ----------------------------------------------------------------
        with gr.Tab("Preprocess"):
            gr.Markdown(
                "Train the BPE tokenizer on raw `.txt` files and write a binary "
                "corpus ready for pre-training.  Run this once before Pre-train."
            )
            with gr.Row():
                pp_input = gr.Textbox(
                    label="Input directory (.txt files)",
                    value="data/corpus/saga/",
                )
                pp_output = gr.Textbox(
                    label="Output corpus binary (.bin)",
                    value="data/processed/corpus.bin",
                )
            with gr.Row():
                pp_vocab = gr.Textbox(
                    label="Vocabulary path (.json)",
                    value="data/tokenizer/bpe.json",
                    info="Trained and saved here on first run; reloaded on subsequent runs.",
                )
                pp_vocab_size = gr.Number(
                    label="Vocabulary size",
                    value=16384,
                    precision=0,
                    info="How many unique word-pieces the tokenizer learns. 16 384 is a good default; only used when training a new tokenizer.",
                )
            with gr.Row():
                pp_run_btn  = gr.Button("Start preprocessing", variant="primary")
                pp_stop_btn = gr.Button(
                    "Stop", interactive=False, elem_classes="stop-btn", scale=0, min_width=80
                )
            pp_log_box = gr.Textbox(
                label="Progress",
                lines=16,
                interactive=False,
                autoscroll=True,
            )
            pp_event = pp_run_btn.click(
                fn=run_preprocess,
                inputs=[pp_input, pp_output, pp_vocab, pp_vocab_size],
                outputs=[pp_log_box, pp_run_btn, pp_stop_btn],
            )
            pp_stop_btn.click(fn=stop_preprocess, inputs=[], outputs=[], cancels=[pp_event])

        # ----------------------------------------------------------------
        with gr.Tab("Pre-train"):
            gr.Markdown("Launch a pre-training run on a tokenised corpus binary.")
            with gr.Row():
                pt_corpus = gr.Textbox(
                    label="Corpus path (.bin)",
                    value="data/processed/corpus.bin",
                )
                pt_ckpt_dir = gr.Textbox(
                    label="Checkpoint directory",
                    value="checkpoints/pretrain/",
                )
            pt_resume = gr.Textbox(
                label="Resume from checkpoint (.pt)",
                placeholder="Leave blank to start from scratch",
                info="Continue a stopped run. Total steps is the final target — resuming from step 5 000 with 10 000 total runs only 5 000 more.",
            )
            with gr.Row():
                pt_steps  = gr.Number(
                    label="Total steps", value=10_000, precision=0,
                    info="Total training iterations. More = better quality, but slower.",
                )
                pt_warmup = gr.Number(
                    label="Warmup steps", value=500, precision=0,
                    info="Ramps the learning rate up gradually at the start to avoid unstable early updates. ~5% of Total steps is a safe default.",
                )
                pt_lr = gr.Number(
                    label="Peak LR", value=3e-4,
                    info="Peak learning rate (how large each weight update is). 3e-4 is well-tested for this model; lower = more stable but slower.",
                )
            with gr.Row():
                pt_batch = gr.Number(
                    label="Batch size", value=4, precision=0,
                    info="Sequences processed per step. Reduce to 1–2 if you run out of memory.",
                )
                pt_accum = gr.Number(
                    label="Gradient accum.", value=8, precision=0,
                    info="Simulates a larger batch without extra memory. Effective batch = Batch size × this value.",
                )
                pt_log = gr.Number(
                    label="Log every N steps", value=50, precision=0,
                )
                pt_save = gr.Number(
                    label="Save every N steps", value=1000, precision=0,
                    info="How often a snapshot is written to disk. More checkpoints = more recovery points but more disk space.",
                )

            # ---- Validation -------------------------------------------------
            with gr.Row():
                pt_val_split = gr.Number(
                    label="Validation split", value=0.0,
                    info="Fraction of the corpus tail held out for validation (e.g. 0.01 = 1%). 0 disables eval. The split is by token, so train and val share no text.",
                )
                pt_eval_every = gr.Number(
                    label="Eval every N steps", value=1000, precision=0,
                    info="How often to compute validation loss. Watch train vs val: both falling = healthy; val flattening/rising while train falls = overfitting.",
                )
                pt_eval_batches = gr.Number(
                    label="Eval batches", value=50, precision=0,
                    info="Max validation batches averaged per eval pass. 0 uses the whole held-out set; a small cap keeps eval fast.",
                )

            # ---- Model architecture -------------------------------------
            with gr.Accordion("Model architecture", open=False):
                gr.Markdown(
                    "Choose a preset or adjust individual dimensions. "
                    "**Changing these requires training from scratch** — "
                    "you cannot resume a run with a different architecture."
                )
                pt_preset = gr.Dropdown(
                    choices=["small-25M", "medium-85M", "large-250M"],
                    value="small-25M",
                    label="Size preset",
                    info="small-25M: fast, good for small corpora. medium-85M: recommended once corpus exceeds 100M tokens. large-250M: requires 8 GB+ VRAM.",
                )
                with gr.Row():
                    pt_d_model   = gr.Number(label="d_model",   value=512,  precision=0, info="Embedding dimension. Larger = more expressive but slower.")
                    pt_n_layers  = gr.Number(label="n_layers",  value=6,    precision=0, info="Number of transformer blocks stacked.")
                    pt_n_heads   = gr.Number(label="n_heads",   value=8,    precision=0, info="Number of attention heads. Must divide d_model evenly.")
                    pt_n_kv_heads= gr.Number(label="n_kv_heads",value=2,    precision=0, info="Key/value heads for grouped query attention. Must divide n_heads evenly.")
                    pt_d_ff      = gr.Number(label="d_ff",      value=1408, precision=0, info="Feed-forward hidden dimension. Default ≈ 2/3 × 4 × d_model.")

                pt_preset.change(
                    fn=apply_model_preset,
                    inputs=[pt_preset],
                    outputs=[pt_d_model, pt_n_layers, pt_n_heads, pt_n_kv_heads, pt_d_ff],
                )

            with gr.Row():
                pt_run_btn  = gr.Button("Start pre-training", variant="primary")
                pt_stop_btn = gr.Button(
                    "Stop", interactive=False, elem_classes="stop-btn", scale=0, min_width=80
                )
            pt_log_box  = gr.Textbox(
                label="Training log",
                lines=20,
                interactive=False,
                autoscroll=True,
            )
            pt_event = pt_run_btn.click(
                fn=run_pretrain,
                inputs=[
                    pt_corpus, pt_ckpt_dir, pt_resume,
                    pt_steps, pt_warmup, pt_lr,
                    pt_batch, pt_accum, pt_log, pt_save,
                    pt_val_split, pt_eval_every, pt_eval_batches,
                    pt_d_model, pt_n_layers, pt_n_heads, pt_n_kv_heads, pt_d_ff,
                ],
                outputs=[pt_log_box, pt_run_btn, pt_stop_btn],
            )
            pt_stop_btn.click(fn=stop_pretrain, inputs=[], outputs=[], cancels=[pt_event])

        # ----------------------------------------------------------------
        with gr.Tab("Fine-tune"):
            gr.Markdown(
                "Specialise a pre-trained model on a conversation dataset. "
                "Fine-tuning teaches the model to respond in a specific style or domain "
                "without retraining from scratch."
            )
            with gr.Row():
                ft_pretrain_ckpt = gr.Textbox(
                    label="Pre-trained checkpoint (.pt)",
                    info="The base model from Pre-train that fine-tuning builds on.",
                )
                ft_ckpt_info = gr.Textbox(
                    label="Checkpoint architecture",
                    interactive=False,
                    info="Architecture and step count read from the checkpoint above.",
                )
            ft_pretrain_ckpt.change(
                fn=read_checkpoint_config,
                inputs=[ft_pretrain_ckpt],
                outputs=[ft_ckpt_info],
            )
            with gr.Row():
                ft_data = gr.Textbox(
                    label="JSONL dataset path",
                    info='Each line must be a JSON object with "prompt" and "response" keys.',
                )
                ft_vocab = gr.Textbox(
                    label="Vocabulary path (.json)",
                    value="data/tokenizer/bpe.json",
                    info="Must be the same vocabulary used during pre-training — mismatching produces garbled output.",
                )
            with gr.Row():
                ft_ckpt_dir = gr.Textbox(
                    label="Output checkpoint directory",
                    value="checkpoints/finetune/",
                )
                ft_max_seq = gr.Number(
                    label="Max sequence length", value=512, precision=0,
                    info="Max tokens per prompt+response pair. Pairs longer than this are truncated. Longer = more memory.",
                )
            ft_resume = gr.Textbox(
                label="Resume fine-tune from checkpoint (.pt)",
                placeholder="Leave blank to start fine-tuning from scratch",
                info="Continue a stopped fine-tuning run. Restores optimizer state and step counter.",
            )
            with gr.Row():
                ft_steps  = gr.Number(
                    label="Total steps", value=500, precision=0,
                    info="Fine-tuning needs far fewer steps than pre-training. Too many can make the model forget its general knowledge.",
                )
                ft_warmup = gr.Number(
                    label="Warmup steps", value=10, precision=0,
                    info="Ramps the learning rate up gradually to avoid disruptive early updates.",
                )
                ft_lr = gr.Number(
                    label="Peak LR", value=5e-5,
                    info="Keep this much lower than pre-training (5e-5 vs 3e-4) to make careful adjustments without overwriting prior learning.",
                )
            with gr.Row():
                ft_batch = gr.Number(
                    label="Batch size", value=4, precision=0,
                    info="Reduce if you run out of memory.",
                )
                ft_accum = gr.Number(
                    label="Gradient accum.", value=4, precision=0,
                    info="Simulates a larger batch without extra memory. Effective batch = Batch size × this value.",
                )
                ft_log = gr.Number(
                    label="Log every N steps", value=25, precision=0,
                )
                ft_save = gr.Number(
                    label="Save every N steps", value=100, precision=0,
                    info="How often a snapshot is written to disk.",
                )
            with gr.Row():
                ft_val_split = gr.Number(
                    label="Validation split", value=0.0,
                    info="Fraction of examples randomly held out for validation (e.g. 0.1 = 10%). 0 disables eval. Fine-tune sets are small, so watch for val loss rising — the classic sign of over-fitting.",
                )
                ft_eval_every = gr.Number(
                    label="Eval every N steps", value=100, precision=0,
                    info="How often to compute validation loss on the held-out examples.",
                )
                ft_eval_batches = gr.Number(
                    label="Eval batches", value=0, precision=0,
                    info="Max validation batches averaged per eval pass. 0 uses the whole held-out set (recommended for small fine-tune sets).",
                )
            with gr.Row():
                ft_run_btn  = gr.Button("Start fine-tuning", variant="primary")
                ft_stop_btn = gr.Button(
                    "Stop", interactive=False, elem_classes="stop-btn", scale=0, min_width=80
                )
            ft_log_box  = gr.Textbox(
                label="Training log",
                lines=20,
                interactive=False,
                autoscroll=True,
            )
            ft_event = ft_run_btn.click(
                fn=run_finetune,
                inputs=[
                    ft_pretrain_ckpt, ft_resume, ft_data, ft_vocab, ft_ckpt_dir,
                    ft_steps, ft_warmup, ft_lr,
                    ft_batch, ft_accum, ft_log, ft_save,
                    ft_val_split, ft_eval_every, ft_eval_batches,
                    ft_max_seq,
                ],
                outputs=[ft_log_box, ft_run_btn, ft_stop_btn],
            )
            ft_stop_btn.click(fn=stop_finetune, inputs=[], outputs=[], cancels=[ft_event])

        # ----------------------------------------------------------------
        with gr.Tab("Scale"):
            gr.Markdown(
                "Estimate how many training steps your corpus and model size warrant, "
                "based on the Chinchilla scaling laws."
            )
            with gr.Row():
                sc_corpus = gr.Textbox(
                    label="Corpus binary (.bin)",
                    value="data/processed/corpus.bin",
                    info="The tokenised corpus written by the Preprocess tab. Used to count tokens.",
                )
                sc_checkpoint = gr.Textbox(
                    label="Checkpoint (.pt) — optional",
                    placeholder="checkpoints/pretrain/step_0010000.pt",
                    info="Load a checkpoint to read the exact parameter count. Leave blank to use the 25M default.",
                )
            with gr.Row():
                sc_batch  = gr.Number(label="Batch size",        value=4,   precision=0,
                    info="Must match what you use in Pre-train.")
                sc_accum  = gr.Number(label="Gradient accum.",   value=8,   precision=0,
                    info="Must match what you use in Pre-train.")
                sc_seq    = gr.Number(label="Sequence length",   value=1024, precision=0,
                    info="Context window. Match your training run — pre-training uses max_seq_len=1024. A smaller value here under-counts tokens-per-step and inflates the pass estimate.")
                sc_steps  = gr.Number(label="Planned total steps", value=20_000, precision=0,
                    info="The total_steps value you intend to use. Adjust to see how it changes the estimates.")
            sc_run_btn = gr.Button("Calculate", variant="primary")
            sc_output  = gr.Textbox(
                label="Results",
                lines=22,
                interactive=False,
                elem_classes=["scale-output"],
            )
            sc_run_btn.click(
                fn=run_scale_calc,
                inputs=[sc_corpus, sc_checkpoint, sc_batch, sc_accum, sc_seq, sc_steps],
                outputs=[sc_output],
            )

        # ----------------------------------------------------------------
        with gr.Tab("Ingest"):
            gr.Markdown(
                "Convert web pages, documents, and images into corpus `.txt` files."
            )

            ing_mode = gr.Radio(
                choices=["URL", "File", "Directory"],
                value="URL",
                label="Source type",
                info="URL: scrape a web page | File: upload a document | Directory: process a local folder.",
            )

            # URL / Directory: text box
            ing_url_or_dir = gr.Textbox(
                label="URL or directory path",
                placeholder="https://example.com/rules  or  data/documents/",
                visible=True,
            )
            # File: upload widget
            ing_file = gr.File(
                label="Upload files",
                visible=False,
                file_count="multiple",
                file_types=[".pdf", ".docx", ".xlsx", ".md", ".txt", ".png", ".jpg", ".jpeg", ".tiff"],
            )

            with gr.Row():
                ing_output = gr.Textbox(
                    label="Output directory",
                    value="data/raw/",
                    placeholder="data/raw/",
                    info="Extracted text files are saved here, ready for the Preprocess tab.",
                )
                ing_cleaning = gr.Radio(
                    choices=["minimal", "standard", "thorough"],
                    value="standard",
                    label="Cleaning level",
                    info=(
                        "minimal — whitespace only | "
                        "standard — collapse blank lines & extra spaces | "
                        "thorough — also drop very short lines & deduplicate paragraphs"
                    ),
                )

            with gr.Row():
                ing_recursive = gr.Checkbox(
                    label="Recursive (subdirectories)",
                    value=False,
                    visible=False,
                    info="Also process files inside subfolders.",
                )
                ing_timeout = gr.Number(
                    label="HTTP timeout (seconds)",
                    value=15,
                    precision=0,
                    visible=True,
                    info="Seconds to wait for a web page before giving up. Increase for slow sites.",
                )

            with gr.Row():
                ing_btn      = gr.Button("Start ingestion", variant="primary")
                ing_stop_btn = gr.Button(
                    "Stop", interactive=False, elem_classes="stop-btn", scale=0, min_width=80
                )
            ing_log = gr.Textbox(
                label="Progress",
                lines=12,
                interactive=False,
                autoscroll=True,
            )

            # Toggle input visibility when mode changes.
            ing_mode.change(
                fn=_toggle_ingest_inputs,
                inputs=[ing_mode],
                outputs=[ing_url_or_dir, ing_file, ing_recursive, ing_timeout],
            )

            ing_event = ing_btn.click(
                fn=run_ingest,
                inputs=[
                    ing_mode, ing_url_or_dir, ing_file,
                    ing_output, ing_cleaning, ing_recursive, ing_timeout,
                ],  # ing_file now returns a list when file_count="multiple"
                outputs=[ing_log, ing_btn, ing_stop_btn],
            )
            ing_stop_btn.click(fn=None, cancels=[ing_event])

        # ----------------------------------------------------------------
        with gr.Tab("Chat"):
            gr.Markdown(
                "Select a named agent **or** load any checkpoint manually. "
                "Conversation history is maintained automatically."
            )
            engine_state = gr.State(value=None)
            conv_state   = gr.State(value=None)

            # ---- Shared retrieval config (used by both agent and manual load)
            _ENCODER_CHOICES = [
                "Model (decoder embeddings)",
                "MiniLM (all-MiniLM-L6-v2)",
                "MPNet (all-mpnet-base-v2)",
                "Lexical (Jaccard)",
            ]
            with gr.Accordion("Retrieval configuration", open=True):
                with gr.Row():
                    chat_encoder = gr.Dropdown(
                        choices=_ENCODER_CHOICES,
                        value="Model (decoder embeddings)",
                        label="Embedding backend",
                        info="How corpus passages are matched to your query. "
                             "Model: the trained transformer's own representations — no extra install. "
                             "MiniLM / MPNet: dedicated sentence encoders, better quality early in training — requires pip install -e \".[encoder]\". "
                             "Lexical: fast word-overlap matching, no neural embedding.",
                        scale=2,
                    )
                    chat_threshold = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.0, step=0.05,
                        label="Retrieval threshold",
                        info="Minimum similarity score for a passage to be injected as context. "
                             "Queries below this score are answered without grounding (pure-chat). "
                             "Cosine scores live in [-1, 1]; 0.0 is a good starting point.",
                        scale=3,
                    )

            # ---- Agent selector -----------------------------------------
            _agent_names = _load_agent_names()
            with gr.Group(visible=bool(_agent_names)) as agent_group:
                gr.Markdown("### Select agent")
                with gr.Row():
                    agent_dropdown = gr.Dropdown(
                        choices=_agent_names,
                        value=_agent_names[0] if _agent_names else None,
                        label="Agent",
                        scale=3,
                    )
                    agent_load_btn = gr.Button("Load agent", scale=1)
                agent_status = gr.Textbox(label="Agent status", interactive=False)

            # ---- Manual load --------------------------------------------
            with gr.Accordion("Load checkpoint manually", open=not bool(_agent_names)):
                with gr.Row():
                    chat_ckpt = gr.Textbox(
                        label="Checkpoint path (.pt)",
                    )
                    chat_vocab = gr.Textbox(
                        label="Vocabulary path (.json)",
                        value="data/tokenizer/bpe.json",
                        info="Must be the same vocabulary used during training — mismatching produces garbled output.",
                    )
                    load_btn = gr.Button("Load model")
                with gr.Row():
                    chat_corpus_dir = gr.Textbox(
                        label="Corpus directory (optional)",
                        value="",
                        info="Directory of .txt files used to ground replies in your corpus. Leave blank for ungrounded chat.",
                    )
                    chat_quantize = gr.Checkbox(
                        label="int8 quantization",
                        value=False,
                        info="Quantize Linear layers to int8 after loading. Cuts memory ~4× and speeds up CPU inference. Ignored on CUDA.",
                        scale=0,
                        min_width=200,
                    )
                load_status = gr.Textbox(label="Status", interactive=False)

            # ---- Generation controls ------------------------------------
            chat_adaptive_temp = gr.Checkbox(
                value=False,
                label="Adaptive temperature",
                info="Recompute temperature at every token from the model's own confidence "
                     "(entropy of the next-token distribution). Confident steps get more "
                     "diversity, uncertain steps get more focus. When on, the manual "
                     "Temperature slider is ignored and hidden.",
            )
            with gr.Row():
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

            # When adaptive temperature is enabled the manual slider is
            # meaningless (temperature is recomputed per token), so hide it.
            chat_adaptive_temp.change(
                fn=lambda on: gr.update(visible=not on),
                inputs=[chat_adaptive_temp],
                outputs=[chat_temp],
            )

            chat_query    = gr.Textbox(label="Your query", lines=3)
            chat_response = gr.Textbox(label="Response", lines=8, interactive=False)
            with gr.Row():
                chat_btn      = gr.Button("Send", variant="primary")
                stage_btn     = gr.Button("✦ Save this exchange", scale=0, min_width=160)
                clear_btn     = gr.Button("Clear conversation", scale=0)

            # ---- Dataset builder ----------------------------------------
            with gr.Accordion("Dataset builder — collect fine-tune pairs", open=False):
                gr.Markdown(
                    "After a good exchange, click **✦ Save this exchange** above. "
                    "Edit the prompt or response below if needed, then click **Add pair**. "
                    "Export to JSONL when you have enough pairs."
                )
                with gr.Row():
                    ds_prompt   = gr.Textbox(label="Prompt", lines=3, interactive=True)
                    ds_response = gr.Textbox(label="Response", lines=3, interactive=True)
                with gr.Row():
                    ds_add_btn    = gr.Button("Add pair", variant="primary")
                    ds_undo_btn   = gr.Button("Undo last", scale=0, min_width=100)
                    ds_clear_btn  = gr.Button("Clear all", scale=0, min_width=100)
                ds_count   = gr.Textbox(
                    label="Dataset", value="0 pairs saved", interactive=False, scale=0
                )
                ds_preview = gr.Textbox(
                    label="Preview (last 3 pairs)", lines=8, interactive=False
                )
                gr.Markdown("---")
                with gr.Row():
                    ds_path = gr.Textbox(
                        label="Dataset file path",
                        value="data/finetune/conversations.jsonl",
                        info="Shared by Load and Export. File is created automatically if it doesn't exist.",
                        scale=3,
                    )
                    ds_load_btn   = gr.Button("Load existing", scale=0, min_width=130)
                    ds_export_btn = gr.Button("Export JSONL", variant="primary", scale=0, min_width=130)
                with gr.Row():
                    ds_overwrite = gr.Checkbox(
                        label="Overwrite file on export",
                        value=False,
                        info="Unchecked (default): new pairs are appended to the existing file. Check this only when you want to replace the file completely.",
                    )
                ds_status = gr.Textbox(label="Status", interactive=False)

            # ---- Dataset state ------------------------------------------
            dataset_state = gr.State(value=[])

            # ---- Event wiring -------------------------------------------
            agent_load_btn.click(
                fn=load_agent,
                inputs=[agent_dropdown, chat_encoder, chat_threshold, chat_quantize],
                outputs=[engine_state, conv_state, agent_status, chat_ckpt, chat_vocab],
            )
            load_btn.click(
                fn=load_engine,
                inputs=[chat_ckpt, chat_vocab, chat_corpus_dir, chat_encoder, chat_threshold, chat_quantize],
                outputs=[engine_state, conv_state, load_status],
            )
            chat_btn.click(
                fn=chat,
                inputs=[chat_query, engine_state, conv_state,
                        chat_temp, chat_top_k, chat_top_p, chat_tokens,
                        chat_adaptive_temp],
                outputs=[chat_response, conv_state],
            )
            clear_btn.click(
                fn=clear_conversation,
                inputs=[conv_state],
                outputs=[conv_state, chat_response],
            )
            stage_btn.click(
                fn=stage_exchange,
                inputs=[chat_query, chat_response],
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

def launch(share: bool = False, port: int = 7860, inbrowser: bool = True) -> None:
    """Build and launch the Gradio app.

    Args:
        share: If ``True``, create a public Gradio tunnel (requires internet).
        port: Local port to serve the app on.
        inbrowser: If ``True`` (default), open the default browser automatically
            when the server is ready.
    """
    build_app().queue().launch(
        server_port=port, share=share, inbrowser=inbrowser, theme=_THEME, css=_CSS
    )


if __name__ == "__main__":
    launch()
