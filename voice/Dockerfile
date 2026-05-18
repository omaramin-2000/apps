ARG BUILD_FROM
FROM ${BUILD_FROM}
ARG BUILD_ARCH

# Set shell
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install
WORKDIR /usr/src
ENV PIP_BREAK_SYSTEM_PACKAGES=1

COPY requirements.txt ./

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3 \
        python3-pip \
    && pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && pip3 install --no-cache-dir -r ./requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm -rf /var/lib/apt/lists/*

COPY src/voice_messages/*.py ./voice_messages/
COPY src/templates/*.html ./templates/
COPY src/citrinet/*.py ./citrinet/
COPY src/default_config.yaml ./
COPY src/*.py ./

WORKDIR /
COPY rootfs /

HEALTHCHECK --start-period=10m \
    CMD curl -f http://localhost:5000/health || exit 1
