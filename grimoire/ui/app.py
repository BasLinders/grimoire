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

import queue
import threading
from pathlib import Path
from typing import Generator, Optional

import gradio as gr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_checkpoints(checkpoint_dir: str) -> list[str]:
    """Return sorted list of .pt files in ``checkpoint_dir``."""
    p = Path(checkpoint_dir)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.pt"))


def _stream_training(train_fn) -> Generator[str, None, None]:
    """Run a training function in a background thread and stream loss lines.

    ``train_fn`` receives an ``on_log(step, loss, lr)`` callback.
    Wraps ``_stream_task`` with a training-specific message formatter.
    """
    def _wrapped(on_progress):
        def on_log(step: int, loss: float, lr: float) -> None:
            on_progress(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e}")
        train_fn(on_log)

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
    """Train BPE tokenizer and write corpus binary; stream progress."""
    from grimoire.llm.data.preprocessing import preprocess

    def _task(on_progress):
        import io
        import sys

        # Redirect stdout so the preprocess() print statements flow into the UI.
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf

        try:
            preprocess(
                input_path=input_dir.strip(),
                output_path=output_path.strip(),
                vocab_path=vocab_path.strip(),
                vocab_size=int(vocab_size),
            )
        finally:
            sys.stdout = old_stdout
            captured = buf.getvalue()

        for line in captured.splitlines():
            on_progress(line)

    yield from _stream_task(_task)


# ---------------------------------------------------------------------------
# Pre-train tab logic
# ---------------------------------------------------------------------------

def run_pretrain(
    corpus_path: str,
    checkpoint_dir: str,
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
    from grimoire.llm.data.dataset import TokenizedDataset
    from grimoire.llm.model.config import TransformerConfig
    from grimoire.llm.model.transformer import GrimoireTransformer
    from grimoire.llm.training.trainer import Trainer

    def _train(on_log):
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
        ).train()

    yield from _stream_training(_train)


# ---------------------------------------------------------------------------
# Fine-tune tab logic
# ---------------------------------------------------------------------------

def run_finetune(
    resume: str,
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
    from grimoire.llm.data.conversation import ConversationDataset
    from grimoire.llm.model.config import TransformerConfig
    from grimoire.llm.model.transformer import GrimoireTransformer
    from grimoire.llm.tokenizer.bpe import BytePairEncoder
    from grimoire.llm.training.checkpoint import load_checkpoint
    from grimoire.llm.training.trainer import Trainer

    def _train(on_log):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = load_checkpoint(resume)
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
        ).train(resume_from=None)

    yield from _stream_training(_train)


# ---------------------------------------------------------------------------
# Ingest tab logic
# ---------------------------------------------------------------------------

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
    from grimoire.corpus.ingest import CleaningLevel, ingest

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

    yield from _stream_task(_task)


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
        from grimoire.agents.registry import AgentRegistry
        return AgentRegistry(_AGENTS_JSON).display_names()
    except (FileNotFoundError, ValueError):
        return []


def load_agent(display_name: str) -> tuple[object, object, str, str, str]:
    """Load an agent by display name.

    Returns (engine, conv_state, status, checkpoint_path, vocab_path).
    The last two values are passed back so the manual path fields reflect what
    was actually loaded.
    """
    from grimoire.agents.registry import AgentRegistry
    from grimoire.state.conversation import ConversationState

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
    from grimoire.llm.inference.engine import InferenceEngine
    from grimoire.state.conversation import ConversationState

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
) -> tuple[str, object]:
    """Generate a response and update the conversation state."""
    if engine_state is None:
        return "No model loaded. Use the Load button first.", conv_state
    from grimoire.llm.inference.sampler import GenerationConfig
    from grimoire.state.conversation import ConversationState
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    if conv_state is None:
        conv_state = ConversationState()
    response = engine_state.chat(query, conv_state, gen_config=gen_config)
    return response, conv_state


def clear_conversation(conv_state) -> tuple[object, str]:
    """Reset the conversation history."""
    from grimoire.state.conversation import ConversationState
    if conv_state is not None:
        conv_state.clear()
    else:
        conv_state = ConversationState()
    return conv_state, ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Assemble and return the Gradio Blocks app."""
    with gr.Blocks(title="Grimoire") as app:
        gr.Markdown("# Grimoire")

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
                    info="Trained and saved here if it does not exist; loaded otherwise.",
                )
                pp_vocab_size = gr.Number(
                    label="Vocabulary size",
                    value=16384,
                    precision=0,
                    info="Used only when training a new tokenizer.",
                )
            pp_run_btn = gr.Button("Start preprocessing", variant="primary")
            pp_log_box = gr.Textbox(
                label="Progress",
                lines=16,
                interactive=False,
                autoscroll=True,
            )
            pp_run_btn.click(
                fn=run_preprocess,
                inputs=[pp_input, pp_output, pp_vocab, pp_vocab_size],
                outputs=pp_log_box,
            )

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
            with gr.Row():
                pt_steps      = gr.Number(label="Total steps",      value=10_000, precision=0)
                pt_warmup     = gr.Number(label="Warmup steps",     value=500,    precision=0)
                pt_lr         = gr.Number(label="Peak LR",          value=3e-4)
            with gr.Row():
                pt_batch      = gr.Number(label="Batch size",       value=4,  precision=0)
                pt_accum      = gr.Number(label="Gradient accum.",  value=8,  precision=0)
                pt_log        = gr.Number(label="Log every N steps",value=50, precision=0)
                pt_save       = gr.Number(label="Save every N steps",value=1000,precision=0)
            pt_run_btn  = gr.Button("Start pre-training", variant="primary")
            pt_log_box  = gr.Textbox(
                label="Training log",
                lines=20,
                interactive=False,
                autoscroll=True,
            )
            pt_run_btn.click(
                fn=run_pretrain,
                inputs=[
                    pt_corpus, pt_ckpt_dir,
                    pt_steps, pt_warmup, pt_lr,
                    pt_batch, pt_accum, pt_log, pt_save,
                ],
                outputs=pt_log_box,
            )

        # ----------------------------------------------------------------
        with gr.Tab("Fine-tune"):
            gr.Markdown(
                "Continue from a pre-trained checkpoint on a JSONL conversation dataset."
            )
            with gr.Row():
                ft_resume   = gr.Textbox(label="Pre-trained checkpoint (.pt)")
                ft_data     = gr.Textbox(label="JSONL dataset path")
                ft_vocab    = gr.Textbox(
                    label="Vocabulary path (.json)",
                    value="data/tokenizer/bpe.json",
                )
            with gr.Row():
                ft_ckpt_dir = gr.Textbox(
                    label="Output checkpoint directory",
                    value="checkpoints/finetune/",
                )
                ft_max_seq  = gr.Number(label="Max sequence length", value=512, precision=0)
            with gr.Row():
                ft_steps    = gr.Number(label="Total steps",      value=500,  precision=0)
                ft_warmup   = gr.Number(label="Warmup steps",     value=10,   precision=0)
                ft_lr       = gr.Number(label="Peak LR",          value=5e-5)
            with gr.Row():
                ft_batch    = gr.Number(label="Batch size",       value=4, precision=0)
                ft_accum    = gr.Number(label="Gradient accum.",  value=4, precision=0)
                ft_log      = gr.Number(label="Log every N steps",value=25,precision=0)
                ft_save     = gr.Number(label="Save every N steps",value=100,precision=0)
            ft_run_btn  = gr.Button("Start fine-tuning", variant="primary")
            ft_log_box  = gr.Textbox(
                label="Training log",
                lines=20,
                interactive=False,
                autoscroll=True,
            )
            ft_run_btn.click(
                fn=run_finetune,
                inputs=[
                    ft_resume, ft_data, ft_vocab, ft_ckpt_dir,
                    ft_steps, ft_warmup, ft_lr,
                    ft_batch, ft_accum, ft_log, ft_save,
                    ft_max_seq,
                ],
                outputs=ft_log_box,
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
            )

            # URL / Directory: text box
            ing_url_or_dir = gr.Textbox(
                label="URL or directory path",
                placeholder="https://example.com/rules  or  data/documents/",
                visible=True,
            )
            # File: upload widget
            ing_file = gr.File(label="Upload file", visible=False)

            with gr.Row():
                ing_output = gr.Textbox(
                    label="Output directory",
                    value="data/raw/",
                    placeholder="data/raw/",
                )
                ing_cleaning = gr.Radio(
                    choices=["minimal", "standard", "thorough"],
                    value="standard",
                    label="Cleaning level",
                    info=(
                        "minimal — whitespace only | "
                        "standard — collapse blank lines & spaces | "
                        "thorough — also drop short lines & deduplicate paragraphs"
                    ),
                )

            with gr.Row():
                ing_recursive = gr.Checkbox(
                    label="Recursive (subdirectories)",
                    value=False,
                    visible=False,
                )
                ing_timeout = gr.Number(
                    label="HTTP timeout (seconds)",
                    value=15,
                    precision=0,
                    visible=True,
                )

            ing_btn = gr.Button("Start ingestion", variant="primary")
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

            ing_btn.click(
                fn=run_ingest,
                inputs=[
                    ing_mode, ing_url_or_dir, ing_file,
                    ing_output, ing_cleaning, ing_recursive, ing_timeout,
                ],
                outputs=ing_log,
            )

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
                    chat_ckpt  = gr.Textbox(label="Checkpoint path (.pt)")
                    chat_vocab = gr.Textbox(
                        label="Vocabulary path (.json)",
                        value="data/tokenizer/bpe.json",
                    )
                    load_btn = gr.Button("Load model")
                load_status = gr.Textbox(label="Status", interactive=False)

            # ---- Generation controls ------------------------------------
            with gr.Row():
                chat_temp   = gr.Slider(0.1, 2.0, value=0.8,  step=0.05, label="Temperature")
                chat_top_k  = gr.Slider(1,   200, value=50,   step=1,    label="Top-k")
                chat_top_p  = gr.Slider(0.1, 1.0, value=0.9,  step=0.05, label="Top-p")
                chat_tokens = gr.Slider(16,  512, value=128,  step=8,    label="Max new tokens")

            chat_query    = gr.Textbox(label="Your query", lines=3)
            chat_response = gr.Textbox(label="Response", lines=8, interactive=False)
            with gr.Row():
                chat_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear conversation")

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

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def launch(share: bool = False, port: int = 7860) -> None:
    """Build and launch the Gradio app.

    Args:
        share: If ``True``, create a public Gradio tunnel (requires internet).
        port: Local port to serve the app on.
    """
    build_app().queue().launch(server_port=port, share=share)


if __name__ == "__main__":
    launch()
