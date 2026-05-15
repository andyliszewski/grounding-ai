# Dockerfile for the grounding-ai corpus-search MCP server.
#
# This image runs only the MCP server entry point
# (`python -m mcp_servers.corpus_search.server`), not the full ingestion
# pipeline. Heavy ingestion-only deps (pix2tex, music21, pypdfium2, scipy,
# unstructured, pdfminer) are intentionally skipped to keep the image lean.
#
# Runtime expects a populated corpus + embeddings on a mounted volume:
#   docker run -i --rm \
#     -v /path/to/corpus:/data/corpus \
#     -v /path/to/embeddings:/data/embeddings \
#     -v /path/to/agents:/app/agents \
#     ghcr.io/andyliszewski/grounding-ai
#
# Without volumes, the server still starts and responds to MCP introspection
# (tools/list returns search_corpus + list_corpus_agents) — searches will
# error per-call until a corpus is mounted.

FROM python:3.13-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1 is the OpenMP runtime FAISS links against on Linux.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first to avoid sentence-transformers pulling
# the full CUDA wheel (saves ~7 GB on the final image).
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0"

RUN pip install \
    "mcp>=1.0.0" \
    "faiss-cpu>=1.7.0" \
    "sentence-transformers>=2.2.0" \
    "pyyaml>=6.0" \
    "rank_bm25>=0.2.2"

COPY grounding /app/grounding
COPY mcp_servers /app/mcp_servers

ENV CORPUS_DIR=/data/corpus \
    EMBEDDINGS_DIR=/data/embeddings \
    AGENTS_DIR=/app/agents

CMD ["python", "-m", "mcp_servers.corpus_search.server"]
