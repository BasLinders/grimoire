"""Command-line entry point for the Grimoire training/eval UI.

``python -m grimoire_ai.ui`` runs the training/eval app -- see
``grimoire_ai.ui.train_app`` for the app itself (Preprocess/Pre-train/
Fine-tune/Scale/Evaluate/Ingest/Corpus) and its own ``main()`` for the
env-var-driven launch logic (host/port/inbrowser, shared with containerized
deployments -- see ``Dockerfile``). Chat is a separate app: run
``python -m grimoire_ai.ui.chat_app`` instead.
"""

import sys


def main() -> None:
    try:
        from grimoire_ai.ui.train_app import main as _train_main
    except ImportError:
        sys.exit(
            "The training UI requires the 'ui' extra. Install it with:\n"
            '    pip install -e ".[ui]"'
        )
    _train_main()


if __name__ == "__main__":
    main()
