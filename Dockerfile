# Personal Assistant AI - container image
#
# Two-stage build. The builder stage compiles wheels; the runtime stage copies
# only the installed packages, so build tooling never ships in the final image.
#
# Build:
#   docker build -t personal-assistant-ai .
#
# The default image installs everything EXCEPT sentence-transformers, which
# pulls PyTorch and adds roughly 2.5 GB. Without it the app runs on the
# deterministic lexical embedder. To build the full-quality image:
#   docker build --build-arg WITH_LOCAL_EMBEDDINGS=true -t personal-assistant-ai:full .

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim AS builder

ARG WITH_LOCAL_EMBEDDINGS=false

# Build tooling is needed only to compile any package lacking a wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements-core.txt .

# requirements-core.txt is the single source of truth. sentence-transformers
# is filtered out unless explicitly requested, rather than maintained in a
# second requirements file that could drift.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && if [ "$WITH_LOCAL_EMBEDDINGS" = "true" ]; then \
         /opt/venv/bin/pip install --no-cache-dir -r requirements-core.txt; \
       else \
         grep -v '^sentence-transformers' requirements-core.txt > /build/slim.txt \
         && /opt/venv/bin/pip install --no-cache-dir -r /build/slim.txt; \
       fi

# ---------------------------------------------------------------- runtime ---
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Personal Assistant AI" \
      org.opencontainers.image.description="RAG assistant over personal notes, with deadline reminders" \
      org.opencontainers.image.source="https://github.com/ZeroXJune/Generative_AI"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Running as root inside a container is a needless risk: a container escape
# would land with root on the host namespace.
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser data/raw/ ./data/raw/
COPY --chown=appuser:appuser docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh && mkdir -p data/processed && chown -R appuser:appuser data

USER appuser
EXPOSE 8501

# Streamlit's own health endpoint; the container is unhealthy until it serves.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status==200 else 1)"

ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["serve"]
