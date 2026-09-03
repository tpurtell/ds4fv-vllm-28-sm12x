# syntax=docker/dockerfile:1.7

ARG EXL3_SOURCE_IMAGE=ghcr.io/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark@sha256:bf383b32a03bdcfef19e42b52778df413c0c47d07c3f4d4e66c78002d17beb74
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce
FROM ${EXL3_SOURCE_IMAGE} AS exl3_source
FROM --platform=linux/arm64 ${VLLM_BASE_IMAGE}

ARG B12X_REPOSITORY=https://github.com/tpurtell/sparkinfer-glmrt
ARG B12X_COMMIT=3fc8d1491d1313c0ca64b2b95772972b7f42ee9d
ARG B12X_VLLM_ADAPTER_COMMIT=30038602b71395f481ef4a6edfe4fcf8551d9c15
ARG B12X_VLLM_ADAPTER_BASE=https://raw.githubusercontent.com/local-inference-lab/vllm/30038602b71395f481ef4a6edfe4fcf8551d9c15
ARG RAY_VERSION=2.48.0
ARG INSTANTTENSOR_VERSION=0.1.9
ARG VLLM_SITE_PACKAGES=/usr/local/lib/python3.12/dist-packages

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Fail closed if BuildKit ignored the requested target platform. B12x is kept as
# an editable, immutable source snapshot because its CuTe kernels compile for the
# serving Spark at runtime.
RUN test "$(uname -m)" = "aarch64"
RUN python3 -m pip install --no-cache-dir "ray[default]==${RAY_VERSION}"
RUN python3 -m pip install --no-cache-dir --no-deps \
      "instanttensor==${INSTANTTENSOR_VERSION}"
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
COPY patches/apply-vllm-b12x-shared-stream.py /tmp/apply-vllm-b12x-shared-stream.py
COPY patches/apply-vllm-b12x-configured-stream.py /tmp/apply-vllm-b12x-configured-stream.py
COPY patches/apply-vllm-vision.py /tmp/apply-vllm-vision.py
COPY patches/apply-vllm-exl3.py /tmp/apply-vllm-exl3.py
COPY patches/apply-vllm-dspark-adaptive-sm121.py \
     /tmp/apply-vllm-dspark-adaptive-sm121.py
COPY patches/apply-vllm-long-prefill-jit.py \
     /tmp/apply-vllm-long-prefill-jit.py
COPY patches/apply-vllm-indexer-workspace.py \
     /tmp/apply-vllm-indexer-workspace.py
COPY patches/apply-vllm-dsv4-kv-groups.py \
     /tmp/apply-vllm-dsv4-kv-groups.py
COPY patches/apply-vllm-dcp-swa.py \
     /tmp/apply-vllm-dcp-swa.py
COPY patches/apply-vllm-dsv4-nvfp4.py \
     /tmp/apply-vllm-dsv4-nvfp4.py
COPY patches/apply-vllm-dcp-dsv4.py \
     /tmp/apply-vllm-dcp-dsv4.py
COPY patches/apply-vllm-dcp-rate-aware.py \
     /tmp/apply-vllm-dcp-rate-aware.py
COPY patches/apply-vllm-dsv4-tokenizer-threadsafe.py \
     /tmp/apply-vllm-dsv4-tokenizer-threadsafe.py
COPY patches/apply-flashinfer-dspark-sm121.py \
     /tmp/apply-flashinfer-dspark-sm121.py
COPY --from=exl3_source \
    /opt/vllm/vllm/model_executor/layers/quantization/exl3.py \
    ${VLLM_SITE_PACKAGES}/vllm/model_executor/layers/quantization/exl3.py
COPY overlay/vllm/model_executor/models/deepseek_v4_vision.py \
     ${VLLM_SITE_PACKAGES}/vllm/model_executor/models/deepseek_v4_vision.py
COPY overlay/vllm/models/deepseek_v4/common/ops/nvfp4_ds_mla.py \
     ${VLLM_SITE_PACKAGES}/vllm/models/deepseek_v4/common/ops/nvfp4_ds_mla.py
RUN echo "209769899a069615e7c8ace17d52515f89ffaf2c73a77532ee45f6de1919710c  ${VLLM_SITE_PACKAGES}/vllm/model_executor/layers/quantization/exl3.py" \
      | sha256sum --check --strict \
 && python3 /tmp/apply-flashinfer-dspark-sm121.py "${VLLM_SITE_PACKAGES}" \
 && python3 /tmp/apply-vllm-b12x.py "${VLLM_SITE_PACKAGES}/vllm" \
      --b12x-root /opt/b12x/b12x \
      --source-base "${B12X_VLLM_ADAPTER_BASE}" \
 && python3 /tmp/apply-vllm-b12x-shared-stream.py "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-b12x-configured-stream.py "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-vision.py "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-exl3.py "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dspark-adaptive-sm121.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-long-prefill-jit.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-indexer-workspace.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dsv4-kv-groups.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dcp-swa.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dsv4-nvfp4.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dcp-dsv4.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dcp-rate-aware.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 /tmp/apply-vllm-dsv4-tokenizer-threadsafe.py \
      "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 -m compileall -q "${VLLM_SITE_PACKAGES}/vllm" \
 && python3 -m py_compile \
      "${VLLM_SITE_PACKAGES}/flashinfer/mla/_sparse_mla_sm120.py" \
      "${VLLM_SITE_PACKAGES}/flashinfer/jit/mla.py" \
 && rm /tmp/apply-vllm-b12x.py /tmp/apply-vllm-b12x-shared-stream.py \
       /tmp/apply-vllm-b12x-configured-stream.py /tmp/apply-vllm-vision.py \
       /tmp/apply-vllm-exl3.py /tmp/apply-vllm-dspark-adaptive-sm121.py \
       /tmp/apply-vllm-long-prefill-jit.py \
       /tmp/apply-vllm-indexer-workspace.py \
       /tmp/apply-vllm-dsv4-kv-groups.py \
       /tmp/apply-vllm-dcp-swa.py \
       /tmp/apply-vllm-dsv4-nvfp4.py \
       /tmp/apply-vllm-dcp-dsv4.py \
       /tmp/apply-vllm-dcp-rate-aware.py \
       /tmp/apply-vllm-dsv4-tokenizer-threadsafe.py \
       /tmp/apply-flashinfer-dspark-sm121.py

COPY scripts/start-native.sh /opt/ds4fv/bin/start-native
COPY scripts/release-warmup.py /opt/ds4fv/bin/release-warmup
COPY scripts/container-healthcheck.py /opt/ds4fv/bin/container-healthcheck.py
COPY tests/spark_exl3_no_gpu_smoke.py /opt/ds4fv/tests/spark_exl3_no_gpu_smoke.py
COPY tests/spark_b12x_no_gpu_smoke.py /opt/ds4fv/tests/spark_b12x_no_gpu_smoke.py
COPY tests/spark_dcp_swa_no_gpu_smoke.py /opt/ds4fv/tests/spark_dcp_swa_no_gpu_smoke.py
COPY tests/spark_dcp_dsv4_no_gpu_smoke.py /opt/ds4fv/tests/spark_dcp_dsv4_no_gpu_smoke.py
COPY tests/spark_dcp_rate_aware_no_gpu_smoke.py /opt/ds4fv/tests/spark_dcp_rate_aware_no_gpu_smoke.py
COPY tests/spark_dsv4_tokenizer_threadsafe_no_gpu.py /opt/ds4fv/tests/spark_dsv4_tokenizer_threadsafe_no_gpu.py
COPY tests/spark_vision_layout_hash_no_gpu_smoke.py /opt/ds4fv/tests/spark_vision_layout_hash_no_gpu_smoke.py
RUN CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_dcp_swa_no_gpu_smoke.py \
 && CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_dcp_dsv4_no_gpu_smoke.py \
 && CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_dcp_rate_aware_no_gpu_smoke.py \
 && CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_dsv4_tokenizer_threadsafe_no_gpu.py \
 && CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_vision_layout_hash_no_gpu_smoke.py \
 && chmod 0755 /opt/ds4fv/bin/start-native /opt/ds4fv/bin/release-warmup

# Keep the recipe identity in a metadata-only tail layer so changing the
# commit does not invalidate the pinned dependency and source-patch layers.
ARG RECIPE_COMMIT=unknown
LABEL org.opencontainers.image.title="DeepSeek V4 Flash/Vision for DGX Spark" \
      org.opencontainers.image.description="arm64-only vLLM 0.28 runtime for GB10 / SM121" \
      org.opencontainers.image.source="https://github.com/tpurtell/ds4fv-vllm-28-sm12x" \
      org.opencontainers.image.revision="${RECIPE_COMMIT}" \
      org.opencontainers.image.base.digest="sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce" \
      io.tpurtell.target.arch="linux/arm64" \
      io.tpurtell.target.cuda.arch="sm_121a" \
      io.tpurtell.b12x.commit="${B12X_COMMIT}" \
      io.tpurtell.b12x.vllm-adapter.commit="${B12X_VLLM_ADAPTER_COMMIT}" \
      io.tpurtell.exl3.source.sha256="209769899a069615e7c8ace17d52515f89ffaf2c73a77532ee45f6de1919710c" \
      io.tpurtell.ray.version="${RAY_VERSION}"

ENV PYTHONPATH=/opt/b12x:/usr/local/lib/python3.12/dist-packages \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_MODULE_LOADING=LAZY \
    CUTE_DSL_ARCH=sm_121a \
    FLASHINFER_WORKSPACE_BASE=/cache/huggingface/vllm-cache/flashinfer-workspace \
    TILELANG_CACHE_DIR=/cache/huggingface/tilelang-cache \
    TRITON_CACHE_DIR=/cache/huggingface/triton-cache \
    VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace

HEALTHCHECK --interval=10s --timeout=5s --start-period=60m --retries=3 \
  CMD ["python3", "/opt/ds4fv/bin/container-healthcheck.py"]

ENTRYPOINT ["/opt/ds4fv/bin/start-native"]
