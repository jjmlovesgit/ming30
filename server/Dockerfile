# Adapter image only: it expects an audio.cpp server (and the GPU) on the host.
#
#   docker build -t ming30-server server/
#   docker run --rm -p 8070:8070 \
#     -e MING30_BACKEND_URL=http://host.docker.internal:8090 \
#     -v ${PWD}/server/voices:/app/voices \
#     ming30-server
#
# The audio.cpp server itself is a native CUDA binary and is not containerised
# here; run it on the host (see ../RUNNING-ming30.md).

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MING30_HOST=0.0.0.0 \
    MING30_PORT=8070 \
    MING30_VOICE_DIR=/app/voices \
    MING30_TMP_DIR=/app/.ming30-tmp

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ming30 ./ming30
COPY scripts ./scripts
COPY examples ./examples
RUN mkdir -p /app/voices /app/.ming30-tmp

EXPOSE 8070
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8070/health')"

CMD ["python", "-m", "ming30"]
