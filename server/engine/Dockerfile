# Native audio.cpp engine image (Linux, CUDA).
#
# The release tarball is a single statically linked binary plus model_specs/,
# so this image is small on top of the CUDA runtime and needs no toolchain.
#
#   docker build -t ming30-engine:0.9.3 server/engine/
#
# Requires the NVIDIA Container Toolkit on the host (GPU passthrough).

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ARG AUDIO_CPP_VERSION=v0.9.3
ARG AUDIO_CPP_FLAVOR=cuda12.8
ARG AUDIO_CPP_URL=https://github.com/0xShug0/audio.cpp/releases/download

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /engine

RUN curl -fsSL -o /tmp/engine.tar.gz \
        "${AUDIO_CPP_URL}/${AUDIO_CPP_VERSION}/audio-${AUDIO_CPP_VERSION}-bin-ubuntu-x64-${AUDIO_CPP_FLAVOR}-colab.tar.gz" \
    && tar -xzf /tmp/engine.tar.gz -C /engine \
    && rm /tmp/engine.tar.gz \
    && chmod +x /engine/audiocpp_server \
    && /engine/audiocpp_server --version

# The model_specs/ directory next to the binary supplies option validation for
# each family; the GGUF's embedded spec is used as a fallback.
ENV PATH="/engine:${PATH}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8090

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD curl -fsS http://127.0.0.1:8090/health || exit 1

ENTRYPOINT ["/engine/audiocpp_server"]
CMD ["--config", "/config/server.json", "--no-ui"]
