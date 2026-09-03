#!/usr/bin/env bash
set -Eeuo pipefail

# Host-side orchestrator. It only starts containers on the named Sparks; the
# image itself performs the SM121 runtime check before importing vLLM.
head_host=${HEAD_HOST:-ostrich}
worker_host=${WORKER_HOST:-dodo}
head_ip=${HEAD_IP:-10.55.0.1}
worker_ip=${WORKER_IP:-10.55.0.2}
image=${DS4FV_IMAGE:-ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev}
container_prefix=${CONTAINER_PREFIX:-ds4fv-native}
hf_cache=${HF_CACHE:-/home/tj/.cache/huggingface}
head_name=${container_prefix}-head
worker_name=${container_prefix}-worker

nccl_hca=${NCCL_IB_HCA:-=rocep1s0f0,roceP2p1s0f0}
nccl_gid_index=${NCCL_IB_GID_INDEX:-3}
nccl_socket_ifname=${NCCL_SOCKET_IFNAME:-enp1s0f0np0}
nccl_cross_nic=${NCCL_CROSS_NIC:-2}
operation=launch

case "${1:-}" in
  --check-only) operation=check ;;
  -h|--help)
    echo "usage: $0 [--check-only]"
    exit 0
    ;;
  "") ;;
  *)
    echo "unknown argument: $1" >&2
    exit 64
    ;;
esac

remote() {
  local host=$1 remote_command
  shift
  printf -v remote_command '%q ' "$@"
  ssh -o BatchMode=yes "${host}" "${remote_command}"
}

check_spark_fabric() {
  local host=$1 check_command
  check_command='test "$(uname -m)" = aarch64'
  check_command+=' && test "$(cat /sys/class/infiniband/rocep1s0f0/ports/1/gid_attrs/types/3)" = "RoCE v2"'
  check_command+=' && test "$(cat /sys/class/infiniband/roceP2p1s0f0/ports/1/gid_attrs/types/3)" = "RoCE v2"'
  remote "${host}" bash -c "${check_command}"
}

remove_container_if_present() {
  local host=$1 name=$2
  if remote "${host}" docker container inspect "${name}" >/dev/null 2>&1; then
    remote "${host}" docker rm -f "${name}" >/dev/null
  fi
}

check_spark_fabric "${head_host}"
check_spark_fabric "${worker_host}"
if [[ "${operation}" == check ]]; then
  echo "Validated arm64 and dual-RoCE GID index 3 on ${head_host}/${worker_host}."
  exit 0
fi

common_args=(
  --gpus all
  --network host
  --ipc host
  --shm-size 64g
  --device /dev/infiniband:/dev/infiniband
  --ulimit memlock=-1:-1
  --ulimit nofile=1048576:1048576
  --ulimit stack=67108864:67108864
  -v "${hf_cache}:/cache/huggingface"
  -e HF_HOME=/cache/huggingface
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  -e MODEL_KIND="${MODEL_KIND:-vision}"
  -e EXL3_PROFILE="${EXL3_PROFILE:-k2.2}"
  -e MOE_MODE="${MOE_MODE:-tp}"
  -e MODEL_REPO="${MODEL_REPO:-}"
  -e MODEL_REVISION="${MODEL_REVISION:-}"
  -e MODEL_PATH="${MODEL_PATH:-}"
  -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-}"
  -e TP_SIZE=2
  -e DCP_SIZE="${DCP_SIZE:-1}"
  -e DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
  -e DS4FV_WORLD_SIZE=2
  -e RAY_HEAD_IP="${head_ip}"
  -e RAY_PORT="${RAY_PORT:-6379}"
  -e CUTE_DSL_ARCH=sm_121a
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e B12X_COMPILE_CACHE_DIR=/cache/huggingface/b12x-compile-cache
  -e TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-/cache/huggingface/tilelang-cache}"
  -e TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/cache/huggingface/triton-cache}"
  -e VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/cache/huggingface/vllm-cache}"
  -e DS4FV_USE_B12X_COMPRESSED_MLA="${DS4FV_USE_B12X_COMPRESSED_MLA:-0}"
  -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-500000}"
  -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
  -e MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
  -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
  -e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
  -e INSTANTTENSOR_BACKEND="${INSTANTTENSOR_BACKEND:-BUFFERED}"
  -e INSTANTTENSOR_IO_DEPTH="${INSTANTTENSOR_IO_DEPTH:-128}"
  -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
  -e ENABLE_DSPARK="${ENABLE_DSPARK:-1}"
  -e DSPARK_TOKENS="${DSPARK_TOKENS:-}"
  -e DRAFT_SAMPLE_METHOD="${DRAFT_SAMPLE_METHOD:-greedy}"
  -e DSPARK_ADAPTIVE_VERIFICATION="${DSPARK_ADAPTIVE_VERIFICATION:-0}"
  -e DS4FV_STARTUP_WARMUP="${DS4FV_STARTUP_WARMUP:-1}"
  -e DS4FV_ENGINE_READY_TIMEOUT_S="${DS4FV_ENGINE_READY_TIMEOUT_S:-3600}"
  -e DS4FV_STARTUP_WARMUP_TIMEOUT_S="${DS4FV_STARTUP_WARMUP_TIMEOUT_S:-1800}"
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  -e DS4FV_PROFILER_CONFIG="${DS4FV_PROFILER_CONFIG:-}"
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"
  -e VLLM_USE_BREAKABLE_CUDAGRAPH="${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}"
  -e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD="${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-1024}"
  -e NCCL_IB_DISABLE=0
  -e NCCL_NET=IB
  -e NCCL_IB_HCA="${nccl_hca}"
  -e NCCL_IB_GID_INDEX="${nccl_gid_index}"
  -e NCCL_IB_ADDR_FAMILY=AF_INET
  -e NCCL_IB_ROCE_VERSION_NUM=2
  -e NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-1}"
  -e NCCL_CROSS_NIC="${nccl_cross_nic}"
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  -e NCCL_SOCKET_IFNAME="${nccl_socket_ifname}"
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${nccl_socket_ifname}}"
  -e RAY_memory_monitor_refresh_ms=0
  -e RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
  -e RAY_USAGE_STATS_ENABLED=0
)

remove_container_if_present "${head_host}" "${head_name}"
remove_container_if_present "${worker_host}" "${worker_name}"

remote "${head_host}" docker run -d \
  --name "${head_name}" \
  "${common_args[@]}" \
  -e DS4FV_ROLE=head \
  -e VLLM_HOST_IP="${head_ip}" \
  "${image}" >/dev/null

for _ in $(seq 1 60); do
  if remote "${head_host}" docker exec "${head_name}" \
    python3 -m ray.scripts.scripts status >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

remote "${worker_host}" docker run -d \
  --name "${worker_name}" \
  "${common_args[@]}" \
  -e DS4FV_ROLE=worker \
  -e VLLM_HOST_IP="${worker_ip}" \
  "${image}" >/dev/null

echo "Started ${head_host}/${head_name} and ${worker_host}/${worker_name}."
echo "Health stays starting until release warmup finishes."
echo "Follow startup: ssh ${head_host} docker logs -f ${head_name}"
