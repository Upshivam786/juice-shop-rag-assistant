# Point 15 - Production deployment improvements over the original Dockerfile:
#   - PYTHONUNBUFFERED=1 so log lines (including our structured JSON logs)
#     flush immediately instead of being stdout-buffered, which matters for
#     `docker logs -f` and any log shipper reading the stream in real time.
#   - Runs as a non-root user - if the container is ever compromised via a
#     dependency vulnerability, the attacker doesn't get root inside it.
#   - HEALTHCHECK wired to the new /health/ready endpoint (checks Chroma +
#     LLM key are actually usable, not just "process is up"), so
#     `docker ps` and orchestrators can tell a genuinely broken container
#     apart from a merely-slow-to-start one.
#   - Same embedding-model bake-in step you already have - kept unchanged,
#     since that's the fix for the original ONNX re-download bug.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image at build time so the container
# never needs network access to Hugging Face at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app /app/app

# Non-root user for defense in depth.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/ready', timeout=4).status == 200 else 1)"

# --timeout-graceful-shutdown gives in-flight requests (including SSE
# streams) a window to finish cleanly on SIGTERM instead of being killed
# mid-response when the container is stopped/recreated.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "10"]
