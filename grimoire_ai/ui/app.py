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

    ``train_fn`` receives ``on_log``, ``on_save``, and ``on_done`` callbacks.
    Wraps ``_stream_task`` with a training-specific message formatter.
    """
    def _wrapped(on_progress):
        def on_log(step: int, loss: float, lr: float, elapsed: float) -> None:
            on_progress(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e} | {_fmt_elapsed(elapsed)}")

        def on_save(step: int, elapsed: float) -> None:
            on_progress(f"  ✔ checkpoint saved  step {step}  [{_fmt_elapsed(elapsed)}]")

        def on_done(step: int, elapsed: float) -> None:
            on_progress(f"\nTraining complete — {step} steps in {_fmt_elapsed(elapsed)}")

        train_fn(on_log, on_save, on_done)

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
) -> Generator[str, None, None]:
    """Launch a pre-training run and stream log output."""
    import torch
    from grimoire_ai.llm.data.dataset import TokenizedDataset
    from grimoire_ai.llm.model.config import TransformerConfig
    from grimoire_ai.llm.model.transformer import GrimoireTransformer
    from grimoire_ai.llm.training.trainer import Trainer

    stop_event = threading.Event()
    _stop_events["pretrain"] = stop_event
    resume = resume_from.strip() or None

    def _train(on_log, on_save, on_done):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_config = TransformerConfig()
        model = GrimoireTransformer(model_config)
        dataset = TokenizedDataset(corpus_path, seq_len=model_config.max_seq_len)
        Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            batch_size=batch_size,
            accumulate_steps=accumulate_steps,
            log_every=log_every,
            save_every=save_every,
            checkpoint_dir=checkpoint_dir,
            device=device,
            on_log=on_log,
            on_save=on_save,
            on_done=on_done,
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
    max_seq_len: int,
) -> Generator[str, None, None]:
    """Launch a fine-tuning run and stream log output."""
    import torch
    from grimoire_ai.llm.data.conversation import ConversationDataset
    from grimoire_ai.llm.model.config import TransformerConfig
    from grimoire_ai.llm.model.transformer import GrimoireTransformer
    from grimoire_ai.llm.tokenizer.bpe import BytePairEncoder
    from grimoire_ai.llm.training.checkpoint import load_checkpoint
    from grimoire_ai.llm.training.trainer import Trainer

    stop_event = threading.Event()
    _stop_events["finetune"] = stop_event
    resume = resume_from.strip() or None

    def _train(on_log, on_save, on_done):
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
        Trainer(
            model=model,
            train_dataset=dataset,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak_lr=peak_lr,
            batch_size=batch_size,
            accumulate_steps=accumulate_steps,
            log_every=log_every,
            save_every=save_every,
            checkpoint_dir=checkpoint_dir,
            device=device,
            on_log=on_log,
            on_save=on_save,
            on_done=on_done,
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
    file_obj,
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
            if file_obj is None:
                raise ValueError("No file uploaded.")
            source = file_obj.name
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


def _load_agent_names() -> list[str]:
    """Return display names from agents.json, or [] if the file is missing."""
    try:
        from grimoire_ai.agents.registry import AgentRegistry
        return AgentRegistry(_AGENTS_JSON).display_names()
    except (FileNotFoundError, ValueError):
        return []


def load_agent(display_name: str) -> tuple[object, object, str, str, str]:
    """Load an agent by display name.

    Returns (engine, conv_state, status, checkpoint_path, vocab_path).
    The last two values are passed back so the manual path fields reflect what
    was actually loaded.
    """
    from grimoire_ai.agents.registry import AgentRegistry
    from grimoire_ai.state.conversation import ConversationState

    registry = AgentRegistry(_AGENTS_JSON)
    cfg = registry.get_by_display_name(display_name)
    engine = registry.build_engine(cfg.key)
    state = ConversationState()
    return (
        engine,
        state,
        f"Agent '{cfg.display_name}' loaded.  {cfg.description}",
        cfg.checkpoint,
        cfg.vocab,
    )


def load_engine(
    checkpoint_path: str,
    vocab_path: str,
) -> tuple[object, object, str]:
    """Load an ``InferenceEngine`` and a fresh ``ConversationState``."""
    from grimoire_ai.llm.inference.engine import InferenceEngine
    from grimoire_ai.state.conversation import ConversationState

    engine = InferenceEngine(
        checkpoint_path=checkpoint_path,
        tokenizer_path=vocab_path,
    )
    state = ConversationState()
    return engine, state, f"Model loaded from {checkpoint_path}"


def chat(
    query: str,
    engine_state,
    conv_state,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
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
    )
    if conv_state is None:
        conv_state = ConversationState()
    for partial in engine_state.chat_stream(query, conv_state, gen_config=gen_config):
        yield partial, conv_state


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
                    ft_max_seq,
                ],
                outputs=[ft_log_box, ft_run_btn, ft_stop_btn],
            )
            ft_stop_btn.click(fn=stop_finetune, inputs=[], outputs=[], cancels=[ft_event])

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
                label="Upload file",
                visible=False,
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
                ],
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
                load_status = gr.Textbox(label="Status", interactive=False)

            # ---- Generation controls ------------------------------------
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
                inputs=[agent_dropdown],
                outputs=[engine_state, conv_state, agent_status, chat_ckpt, chat_vocab],
            )
            load_btn.click(
                fn=load_engine,
                inputs=[chat_ckpt, chat_vocab],
                outputs=[engine_state, conv_state, load_status],
            )
            chat_btn.click(
                fn=chat,
                inputs=[chat_query, engine_state, conv_state,
                        chat_temp, chat_top_k, chat_top_p, chat_tokens],
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
