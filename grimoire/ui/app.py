"""Grimoire training UI.

A three-tab Gradio web app for managing Grimoire training runs and testing
the model interactively — without touching the CLI.

Tabs
----
Pre-train
    Launch a pre-training run on a tokenised corpus binary.  Hyperparameters
    mirror the defaults in ``grimoire.llm.training.train``.  Loss is streamed
    live via the ``Trainer.on_log`` callback — no stdout polling.

Fine-tune
    Continue from a pre-trained checkpoint on a JSONL conversation dataset.
    Hyperparameters mirror ``grimoire.llm.training.finetune``.

Chat
    Load any checkpoint and query the model interactively via
    ``InferenceEngine``.  Optionally point at a corpus directory to enable
    retrieval-augmented responses.

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
    """Run ``train_fn`` in a background thread and stream log lines.

    ``train_fn`` receives a single argument: an ``on_log`` callback with
    signature ``(step: int, loss: float, lr: float) -> None``.  It must put
    a ``None`` sentinel into the shared queue when it finishes (or errors).

    Yields the full accumulated log text so the Gradio ``Textbox`` always
    shows the complete history, not just the latest line.
    """
    log_q: queue.Queue = queue.Queue()

    def on_log(step: int, loss: float, lr: float) -> None:
        log_q.put(f"step {step:>6} | loss {loss:.4f} | lr {lr:.2e}\n")

    def _run() -> None:
        try:
            train_fn(on_log)
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
            yield log_text  # keep the UI responsive between log intervals
            continue

        if msg is None:
            log_text += "\nDone.\n"
            yield log_text
            break

        log_text += msg
        yield log_text


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
# Chat tab logic
# ---------------------------------------------------------------------------

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
        gr.Markdown("# Grimoire Training UI")

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
        with gr.Tab("Chat"):
            gr.Markdown(
                "Load a checkpoint and have a multi-turn conversation. "
                "Conversation history is maintained automatically."
            )
            engine_state = gr.State(value=None)
            conv_state   = gr.State(value=None)

            with gr.Row():
                chat_ckpt   = gr.Textbox(label="Checkpoint path (.pt)")
                chat_vocab  = gr.Textbox(
                    label="Vocabulary path (.json)",
                    value="data/tokenizer/bpe.json",
                )
                load_btn    = gr.Button("Load model")
            load_status = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                chat_temp   = gr.Slider(0.1, 2.0,  value=0.8,  step=0.05, label="Temperature")
                chat_top_k  = gr.Slider(1,   200,  value=50,   step=1,    label="Top-k")
                chat_top_p  = gr.Slider(0.1, 1.0,  value=0.9,  step=0.05, label="Top-p")
                chat_tokens = gr.Slider(16,  512,  value=128,  step=8,    label="Max new tokens")

            chat_query    = gr.Textbox(label="Your query", lines=3)
            chat_response = gr.Textbox(label="Response", lines=8, interactive=False)
            with gr.Row():
                chat_btn  = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear conversation")

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
