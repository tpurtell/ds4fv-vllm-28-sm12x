#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side launcher: every Docker and GPU action occurs on the selected Spark.
spark_host=${SPARK_HOST:-dodo}
image=${DS4FV_IMAGE:-ds4fv-vllm-28-sm12x:exl3-dev}
container_name=${CONTAINER_NAME:-ds4fv-exl3}
hf_cache=${HF_CACHE:-/home/tj/.cache/huggingface}
model_repo=${MODEL_REPO:-}
model_revision=${MODEL_REVISION:-}
model_kind=${MODEL_KIND:-text}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-}

if [[ -z "${gpu_memory_utilization}" ]]; then
  if [[ "${model_kind}" == vision ]]; then
    gpu_memory_utilization=0.86
  else
    gpu_memory_utilization=0.85
  fi
fi

remote() {
  local host=$1 remote_command
  shift
  printf -v remote_command '%q ' "$@"
  ssh -o BatchMode=yes "${host}" "${remote_command}"
}

if [[ "$(remote "${spark_host}" uname -m)" != aarch64 ]]; then
  echo "${spark_host} is not an arm64 DGX Spark" >&2
  exit 64
fi

if remote "${spark_host}" docker container inspect "${container_name}" \
  >/dev/null 2>&1; then
  remote "${spark_host}" docker rm -f "${container_name}" >/dev/null
fi

remote "${spark_host}" docker run -d \
  --name "${container_name}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1:-1 \
  --ulimit nofile=1048576:1048576 \
  --ulimit stack=67108864:67108864 \
  -v "${hf_cache}:/cache/huggingface" \
  -e DS4FV_ROLE=exl3 \
  -e MODEL_KIND="${model_kind}" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
  -e MODEL_REPO="${model_repo}" \
  -e MODEL_REVISION="${model_revision}" \
  -e MODEL_PATH="${MODEL_PATH:-}" \
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}" \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e B12X_COMPILE_CACHE_DIR=/cache/huggingface/b12x-compile-cache \
  -e TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/cache/huggingface/tilelang-cache}" \
  -e TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/huggingface/triton-cache}" \
  -e DS4FV_USE_B12X_COMPRESSED_MLA="${DS4FV_USE_B12X_COMPRESSED_MLA:-0}" \
  -e VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/huggingface/vllm-cache}" \
  -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  -e LOAD_FORMAT="${LOAD_FORMAT:-instanttensor}" \
  -e ENABLE_DSPARK="${ENABLE_DSPARK:-1}" \
  -e DSPARK_TOKENS="${DSPARK_TOKENS:-}" \
  -e DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}" \
  -e DSPARK_ADAPTIVE_VERIFICATION="${DSPARK_ADAPTIVE_VERIFICATION:-0}" \
  -e DS4FV_STARTUP_WARMUP="${DS4FV_STARTUP_WARMUP:-1}" \
  -e DS4FV_ENGINE_READY_TIMEOUT_S="${DS4FV_ENGINE_READY_TIMEOUT_S:-3600}" \
  -e DS4FV_STARTUP_WARMUP_TIMEOUT_S="${DS4FV_STARTUP_WARMUP_TIMEOUT_S:-1800}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}" \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}" \
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}" \
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}" \
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}" \
  -e GPU_MEMORY_UTILIZATION="${gpu_memory_utilization}" \
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}" \
  -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}" \
  "${image}" >/dev/null

echo "Started ${spark_host}/${container_name}."
echo "Health stays starting until release warmup finishes."
echo "Follow startup: ssh ${spark_host} docker logs -f ${container_name}"
