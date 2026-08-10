# microbiome-agent — container image
#
# Bundles the analysis library, MCP server, agent loop, eval harness, and the
# FastAPI service wrapper into one image. The agent spawns its own MCP server
# as a subprocess over stdio (see agent/__main__.py and api/app.py), so no
# second container / network hop is needed — this is a single-process image.
#
# Build:
#   docker build -t microbiome-agent .
#
# Run the API service (see api/app.py):
#   docker run --rm -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... microbiome-agent
#
# Run the CLI instead:
#   docker run --rm -e ANTHROPIC_API_KEY=sk-ant-... microbiome-agent \
#       python -m microbiome_agent.agent "Analyse the example dataset."
#
# Run the free (no-API-key) eval dry-run to sanity-check the image:
#   docker run --rm microbiome-agent python -m microbiome_agent.eval --dry-run

FROM python:3.12-slim AS base

# scikit-bio and a couple of transitive deps need a compiler toolchain to
# build from sdist on some platforms/arches; keep the image otherwise slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so this layer is cached across source-only changes.
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi "uvicorn[standard]"

COPY src/ src/
COPY README.md ROADMAP.md ./

RUN pip install --no-cache-dir -e .

# Non-root: no reason for the analysis/agent process to run as root.
RUN useradd --create-home --shell /bin/bash agent
USER agent

EXPOSE 8000

# Fails fast and loudly if ANTHROPIC_API_KEY is unset when the container
# actually needs to call the model — deliberately not baked in as an ENV
# default so a missing key surfaces immediately rather than a silent 401.
CMD ["uvicorn", "microbiome_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
