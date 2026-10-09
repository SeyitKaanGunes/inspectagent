# syntax=docker/dockerfile:1
# InspectAgent - generic multi-arch container (uvicorn on :8080) for local use, ECS/Fargate or App Runner.
# Build context = repository root, filtered by deploy/.dockerignore:
#   cp deploy/.dockerignore .dockerignore
#   docker build -f deploy/Dockerfile -t inspectagent:local .
#   docker buildx build --platform linux/arm64,linux/amd64 -f deploy/Dockerfile -t <repo>:<tag> --push .
#   docker run --rm -p 8080:8080 inspectagent:local      ->  http://localhost:8080/
ARG PYTHON_IMAGE=python:3.13-slim
FROM ${PYTHON_IMAGE}

ARG GIT_SHA=unknown
ARG IMAGE_TAG=dev
LABEL org.opencontainers.image.title="InspectAgent" \
      org.opencontainers.image.description="Agentic visual inspection with OpenCV 5 (generic uvicorn image)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.version="${IMAGE_TAG}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8080 \
    INSPECTAGENT_STORAGE=local \
    INSPECTAGENT_MODELS_DIR=/app/models \
    INSPECTAGENT_SAMPLES_DIR=/app/samples \
    INSPECTAGENT_RESULTS_DIR=/app/results \
    INSPECTAGENT_LOCAL_STORAGE_DIR=/data/runs \
    INSPECTAGENT_IMAGE_TAG=${IMAGE_TAG} \
    INSPECTAGENT_GIT_SHA=${GIT_SHA}

WORKDIR /app

COPY requirements.lock.txt requirements.txt ./
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.lock.txt \
    && python -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'numpy', numpy.__version__)"

# Unprivileged runtime user; /data/runs holds local traces/artifacts/reviews (mount a volume to persist).
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/runs \
    && chown -R app:app /data

COPY --chown=app:app inspectagent/ ./inspectagent/
COPY --chown=app:app scripts/ ./scripts/
COPY --chown=app:app samples/ ./samples/
COPY --chown=app:app models/ ./models/
COPY --chown=app:app results/ ./results/

USER app
EXPOSE 8080
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8080'), timeout=4).read()" || exit 1

# PORT defaults to 8080 (override with -e PORT=... for platforms that inject it).
CMD ["sh", "-c", "exec python -m uvicorn inspectagent.api.app:app --host 0.0.0.0 --port \"${PORT:-8080}\" --proxy-headers --forwarded-allow-ips '*'"]
