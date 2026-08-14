"""Helpers, theme, and chrome shared by both Grimoire UIs.

``train_app.py`` (Preprocess/Pre-train/Fine-tune/Scale/Evaluate/Ingest/Corpus)
and ``chat_app.py`` (the rebuilt chat interface) both import from this
module rather than duplicating filesystem-scan helpers, device-profile
detection, semantic-index freshness checks, or the theme/CSS/header chrome.
Everything here is genuinely used by both apps -- confirmed by grepping call
sites in the original single-file ``app.py`` before the split, not guessed.

Training-only streaming machinery (``_stream_task``, ``_wrap_with_buttons``,
``_stream_training``, the stop-event registry, ``_fmt_elapsed``) is NOT
here -- Chat streams directly via ``InferenceEngine.chat_stream`` and never
touches that machinery, so it lives in ``train_app.py`` instead.
"""

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import gradio as gr

# ---------------------------------------------------------------------------
# Filesystem scan helpers
# ---------------------------------------------------------------------------

def _scan_files(base_dir: str, pattern: str, recursive: bool = False) -> list[str]:
    """Return files matching pattern under base_dir, sorted newest-first."""
    p = Path(base_dir)
    if not p.exists():
        return []
    matches = list(p.rglob(pattern) if recursive else p.glob(pattern))
    matches.sort(key=lambda f: f.stat().st_mtime if f.exists() else 0, reverse=True)
    return [str(f) for f in matches]


def _scan_subdirs(base_dir: str) -> list[str]:
    """Return immediate subdirectories of base_dir (hidden dirs excluded), newest-first."""
    p = Path(base_dir)
    if not p.exists():
        return []
    dirs = [d for d in p.iterdir() if d.is_dir() and not d.name.startswith(".")]
    dirs.sort(key=lambda d: d.stat().st_mtime if d.exists() else 0, reverse=True)
    return [str(d) + "/" for d in dirs]


# ---------------------------------------------------------------------------
# Dropdown refresh helpers used by both apps.
#
# Training-only refreshers (_refresh_ckpts_pretrain, _refresh_jsonl_choices)
# and chat-only ones (_refresh_lora_choices) live in their respective app
# modules instead, since only one side ever wires them to a `.select()`.
# ---------------------------------------------------------------------------

def _refresh_ckpts_all():
    return gr.update(choices=_scan_files("checkpoints/", "*.pt", recursive=True))


def _refresh_corpus_dirs():
    return gr.update(choices=_scan_subdirs("data/corpus/"))


# ---------------------------------------------------------------------------
# Device profile
# ---------------------------------------------------------------------------

def _detect_device_profile() -> dict:
    """Probe the available compute device and derive safe training/inference defaults.

    Returns a dict with keys:
      device    – "cuda", "mps", or "cpu"
      vram_gb   – total GPU/unified memory in GiB (0.0 on CPU)
      pt_batch  – recommended pre-train batch size
      pt_accum  – recommended pre-train gradient accumulation steps
      ft_batch  – recommended fine-tune batch size
      ft_accum  – recommended fine-tune gradient accumulation steps
      grad_ckpt – whether to suggest gradient checkpointing
      quantize  – whether to suggest int8 quantization (CPU inference only)
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        elif torch.backends.mps.is_available():
            device = "mps"
            # Apple Silicon uses unified memory — there's no separate VRAM
            # pool to query. ``recommended_max_memory`` (torch >= 2.3) is the
            # closest available signal; fall back to total system RAM (the
            # unified pool the GPU shares) via the POSIX sysconf API when the
            # torch API isn't present, then to a conservative neutral default.
            try:
                vram_gb = torch.mps.recommended_max_memory() / (1024 ** 3)
            except Exception:
                try:
                    import os
                    vram_gb = (
                        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    ) / (1024 ** 3)
                except Exception:
                    vram_gb = 8.0
        else:
            return dict(device="cpu", vram_gb=0.0,
                        pt_batch=1, pt_accum=32,
                        ft_batch=1, ft_accum=16,
                        grad_ckpt=False, quantize=True)
    except Exception:
        # torch unavailable or device probe failed — return neutral defaults.
        return dict(device="cpu", vram_gb=0.0,
                    pt_batch=4, pt_accum=8,
                    ft_batch=4, ft_accum=4,
                    grad_ckpt=False, quantize=False)

    # Batch sizing: target effective batch = 32 for pre-train, 16 for fine-tune.
    if vram_gb < 4:
        pt_batch, ft_batch = 1, 1
    elif vram_gb < 8:
        pt_batch, ft_batch = 2, 2
    elif vram_gb < 16:
        pt_batch, ft_batch = 4, 4
    else:
        pt_batch, ft_batch = 8, 8

    return dict(
        device=device,
        vram_gb=vram_gb,
        pt_batch=pt_batch,
        pt_accum=max(1, 32 // pt_batch),
        ft_batch=ft_batch,
        ft_accum=max(1, 16 // ft_batch),
        grad_ckpt=vram_gb < 16,
        quantize=False,
    )


# ---------------------------------------------------------------------------
# Semantic index freshness -- used by Chat's load_agent/load_engine and
# Evaluate's run_eval_ui.
# ---------------------------------------------------------------------------

def _semantic_index_dir(corpus_dirs: list[str]) -> Optional[Path]:
    """Return the .semantic_index directory path for this corpus, or None."""
    if not corpus_dirs:
        return None
    first_dir = Path(corpus_dirs[0])
    if not first_dir.is_dir():
        return None
    return first_dir / ".semantic_index"


def _semantic_index_dir_external(corpus_dirs: list[str], encoder_model_id: str) -> Optional[Path]:
    """Return the persistent index directory for an *external* sentence encoder.

    Distinct from ``_semantic_index_dir`` (the model's own decoder-embedding
    index) and keyed by ``encoder_model_id`` — MiniLM and MPNet embeddings
    aren't even the same dimensionality, so they can never share a cache
    directory the way switching between them would otherwise silently imply.
    """
    if not corpus_dirs:
        return None
    first_dir = Path(corpus_dirs[0])
    if not first_dir.is_dir():
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", encoder_model_id.lower()).strip("-")
    return first_dir / f".semantic_index_external_{slug}"


def _index_is_fresh(
    index_dir: Path, corpus_dirs: list[str], checkpoint: str, lora_path: str = "",
) -> bool:
    """Return True when the on-disk RagIndex is up to date (content-hash based).

    ``lora_path`` is folded into the hash so an index built from base-model
    embeddings is never silently reused once a LoRA adapter is active (or
    vice versa) — the embeddings differ depending on which weights produced
    them. Pass ``checkpoint=""`` (as external-encoder callers do) when the
    index doesn't depend on a Grimoire checkpoint at all — an external
    sentence-transformers encoder's embeddings are a function of the corpus
    text alone, not of any checkpoint/LoRA weights.
    """
    from grimoire_ai.llm.inference.rag_index import RagIndex
    hashes = RagIndex.compute_source_hashes(
        corpus_dirs, checkpoint, lora_path=lora_path or None, cache_dir=index_dir,
    )
    return not RagIndex.is_stale(index_dir, hashes)


#: Retrieval embedding backend choices, shared by the Chat and Evaluate tabs.
_ENCODER_CHOICES = [
    "Model (decoder embeddings)",
    "MiniLM (all-MiniLM-L6-v2)",
    "MPNet (all-mpnet-base-v2)",
    "Lexical (Jaccard)",
]

#: Cross-encoder reranker choices for the Chat tab's retrieval configuration.
#: "None" disables reranking (default) -- retrieval results are used exactly
#: as the first-stage retriever (above) ranked them.
_RERANKER_CHOICES = [
    "None",
    "TinyBERT (ms-marco-TinyBERT-L-2-v2)",
    "MiniLM-12 (ms-marco-MiniLM-L-12-v2)",
]


# ---------------------------------------------------------------------------
# Theme + CSS
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
    # background_fill_primary/secondary are the generic Base-theme tokens
    # gr.Sidebar (and its collapse toggle button) use directly -- unlike
    # block_background_fill/input_background_fill above, nothing else in
    # this file happened to set these, so Sidebar silently fell back to
    # Gradio's own default value (a Tailwind slate navy unrelated to this
    # palette) and never moved when the light/dark toggle ran, since
    # _DARK_VARS/_LIGHT_VARS below didn't cover it either.
    background_fill_primary="#1e1e2e",
    background_fill_primary_dark="#1e1e2e",
    background_fill_secondary="#16161f",
    background_fill_secondary_dark="#16161f",
    # Borders
    block_border_color="#2e2e45",
    block_border_color_dark="#2e2e45",
    input_border_color="#2e2e45",
    input_border_color_dark="#2e2e45",
    # border_color_primary: the generic token Sidebar's collapse toggle
    # button borders with directly, same gap as background_fill_* above.
    border_color_primary="#2e2e45",
    border_color_primary_dark="#2e2e45",
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
    # gr.Sidebar and its collapse toggle read these directly rather than
    # block-background-fill/input-background-fill -- without them here the
    # toggle changes every other panel's color but leaves the Sidebar stuck
    # on whatever it resolved to at load time.
    "--background-fill-primary": "#1e1e2e",
    "--background-fill-secondary": "#16161f",
    "--block-border-color": "#2e2e45",
    "--input-border-color": "#2e2e45",
    "--border-color-primary": "#2e2e45",
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
    "--background-fill-primary": "#eeeae0",
    "--background-fill-secondary": "#fffef9",
    "--block-border-color": "#c8bfa8",
    "--input-border-color": "#c8bfa8",
    "--border-color-primary": "#c8bfa8",
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


def _toggle_theme(current: str) -> tuple[str, str]:
    """Flip dark/light theme state and return the new button label."""
    new = "light" if current == "dark" else "dark"
    # Label always shows the OTHER mode (what the next click will do).
    label = "☀ Light mode" if new == "dark" else "🌙 Dark mode"
    return label, new


# Shutdown: Python fn schedules os._exit on a background thread so Gradio can
# return a clean response before the process dies. JS first tries
# window.close() directly — this works in some browsers when called from a
# click handler. If the browser blocks it, a 300 ms fallback navigates to a
# goodbye page that contains its own "Close this tab" button (user-initiated
# click makes window.close() reliable there).
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


def add_header() -> None:
    """Add the Grimoire title/theme-toggle/shutdown row to the app.

    Call this as the first thing inside ``with gr.Blocks(...) as app:`` in
    both ``train_app.py`` and ``chat_app.py`` — Gradio components attach to
    whichever ``Blocks`` context is currently active, so this needs no
    ``app`` argument, just to be called from inside that ``with`` block.
    """
    theme_state = gr.State("dark")
    with gr.Row(elem_classes="grimoire-header"):
        gr.Markdown("# ✦ Grimoire")
        theme_btn = gr.Button(
            "☀ Light mode", scale=0, min_width=120, elem_id="theme-btn"
        )
        shutdown_btn = gr.Button(
            "⏻ Shut down", scale=0, min_width=110, elem_id="shutdown-btn"
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
        // which beats any !important in Gradio's compiled <style> block --
        // EXCEPT for a property a descendant redefines itself. Gradio 6
        // tags <body> with its own "dark"/"" class from prefers-color-scheme
        // (independent of our data-theme attribute) and some of its own
        // CSS keys background-fill-primary/secondary and border-color-
        // primary off that class -- newer components like gr.Sidebar read
        // those directly. A redefinition on <body> shadows our override on
        // <html> for anything inheriting through <body>, !important or not,
        // since custom-property cascade resolves per-element -- so this has
        // to also set every var on <body> itself, not just <html>.
        const vars = isDark ? {_vars_to_js(_LIGHT_VARS)} : {_vars_to_js(_DARK_VARS)};
        for (const [p, v] of Object.entries(vars)) {{
            root.style.setProperty(p, v, 'important');
            document.body.style.setProperty(p, v, 'important');
        }}

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
