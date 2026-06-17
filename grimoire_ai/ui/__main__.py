"""Command-line entry point for the Grimoire training UI.

Host/port/inbrowser are read from environment variables so the same launch
logic works unchanged for local development and for containerized
deployments (see ``Dockerfile``) — locally these env vars are simply unset
and the original defaults apply.
"""

import os
import sys


def main() -> None:
    try:
        from grimoire_ai.ui.app import launch
    except ImportError:
        sys.exit(
            "The training UI requires the 'ui' extra. Install it with:\n"
            '    pip install -e ".[ui]"'
        )

    inbrowser_env = os.environ.get("GRIMOIRE_UI_INBROWSER", "1").strip().lower()
    launch(
        server_name=os.environ.get("GRIMOIRE_UI_HOST", "127.0.0.1"),
        port=int(os.environ.get("GRIMOIRE_UI_PORT", "7860")),
        inbrowser=inbrowser_env not in ("0", "false", "no"),
    )


if __name__ == "__main__":
    main()
