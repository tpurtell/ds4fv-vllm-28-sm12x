# syntax=docker/dockerfile:1.7

ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce
FROM --platform=linux/arm64 ${VLLM_BASE_IMAGE}

ARG B12X_REPOSITORY=https://github.com/tpurtell/sparkinfer-glmrt
ARG B12X_COMMIT=a13677130cd144772bc7528238fe2244bbe3d0d4
ARG B12X_VLLM_ADAPTER_COMMIT=30038602b71395f481ef4a6edfe4fcf8551d9c15
ARG B12X_VLLM_ADAPTER_BASE=https://raw.githubusercontent.com/local-inference-lab/vllm/30038602b71395f481ef4a6edfe4fcf8551d9c15
ARG RAY_VERSION=2.48.0
ARG VLLM_SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages

LABEL org.opencontainers.image.title="DeepSeek V4 Flash/Vision for DGX Spark" \
      org.opencontainers.image.description="arm64-only vLLM 0.28 runtime for GB10 / SM121" \
      org.opencontainers.image.source="https://github.com/tpurtell/ds4fv-vllm-28-sm12x" \
      org.opencontainers.image.base.digest="sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce" \
      io.tpurtell.target.arch="linux/arm64" \
      io.tpurtell.target.cuda.arch="sm_121a" \
      io.tpurtell.b12x.commit="a13677130cd144772bc7528238fe2244bbe3d0d4" \
      io.tpurtell.b12x.vllm-adapter.commit="${B12X_VLLM_ADAPTER_COMMIT}" \
      io.tpurtell.ray.version="${RAY_VERSION}"

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Fail closed if BuildKit ignored the requested target platform. B12x is kept as
# an editable, immutable source snapshot because its CuTe kernels compile for the
# serving Spark at runtime.
RUN test "$(uname -m)" = "aarch64"
RUN python3 -m pip install --no-cache-dir "ray[default]==${RAY_VERSION}"
RUN B12X_REPOSITORY="${B12X_REPOSITORY}" B12X_COMMIT="${B12X_COMMIT}" \
    python3 - <<'PY'
import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

repository = os.environ["B12X_REPOSITORY"].removesuffix(".git")
commit = os.environ["B12X_COMMIT"]
archive = Path("/tmp/b12x.tar.gz")
urllib.request.urlretrieve(f"{repository}/archive/{commit}.tar.gz", archive)
with tarfile.open(archive) as tar:
    tar.extractall("/tmp", filter="data")
sources = list(Path("/tmp").glob("sparkinfer-glmrt-*"))
if len(sources) != 1:
    raise RuntimeError(f"expected one B12x source tree, found: {sources}")
shutil.rmtree("/opt/b12x", ignore_errors=True)
shutil.move(sources[0], "/opt/b12x")
archive.unlink()
PY
RUN python3 -m pip install --no-cache-dir --no-deps -e /opt/b12x

COPY patches/apply-vllm-b12x.py /tmp/apply-vllm-b12x.py
COPY patches/apply-vllm-vision.py /tmp/apply-vllm-vision.py
COPY overlay/vllm/model_executor/models/deepseek_v4_vision.py \
     ${VLLM_SITE_PACKAGES}/vllm/model_executor/models/deepseek_v4_vision.py
RUN python3 /tmp/apply-vllm-b12x.py "${VLLM_SITE_PACKAGES}/vllm" \
      --b12x-root /opt/b12x/b12x \
      --source-base "${B12X_VLLM_ADAPTER_BASE}" \
 && python3 /tmp/apply-vllm-vision.py "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 -m compileall -q "${VLLM_SITE_PACKAGES}/vllm" \
 && rm /tmp/apply-vllm-b12x.py /tmp/apply-vllm-vision.py

COPY scripts/start-native.sh /opt/ds4fv/bin/start-native
RUN chmod 0755 /opt/ds4fv/bin/start-native

ENV PYTHONPATH=/opt/b12x:/usr/local/lib/python3.12/dist-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_MODULE_LOADING=LAZY \
    CUTE_DSL_ARCH=sm_121a \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace

ENTRYPOINT ["/opt/ds4fv/bin/start-native"]
