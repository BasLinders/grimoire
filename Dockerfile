# Grimoire training UI — CPU image.
#
# Build:
#   docker build -t grimoire-ai .
#
# Run (mount your data/checkpoints/agents.json from the host so they
# persist across container restarts and survive image rebuilds):
#   docker run --rm -p 7860:7860 \
#       -v "$(pwd)/data:/app/data" \
#       -v "$(pwd)/checkpoints:/app/checkpoints" \
#       -v "$(pwd)/agents.json:/app/agents.json" \
#       grimoire-ai
#
# For CUDA, install a matching torch build on top of this image (see
# docs/setup-training.md) or swap the base image for an nvidia/cuda one
# and reinstall torch with a cu12x index URL before `pip install -e .`.

FROM python:3.11-slim

WORKDIR /app

# Install the CPU build of torch first so it's cached independently of
# application code changes — avoids re-downloading torch on every rebuild.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY grimoire_ai ./grimoire_ai
RUN pip install --no-cache-dir ".[ui]"

COPY agents.json ./agents.json

# Bind to all interfaces and skip the local-browser auto-open inside the
# container — see grimoire_ai/ui/__main__.py.
ENV GRIMOIRE_UI_HOST=0.0.0.0
ENV GRIMOIRE_UI_PORT=7860
ENV GRIMOIRE_UI_INBROWSER=0

EXPOSE 7860

CMD ["grimoire-ui"]
