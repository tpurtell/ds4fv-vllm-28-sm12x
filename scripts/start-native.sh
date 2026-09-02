#!/usr/bin/env bash
set -Eeuo pipefail

# Container entrypoint for a two-Spark Ray cluster.  The architecture check is
# deliberately before any vLLM import so this Spark-only image fails closed on
# an amd64 workstation or a non-GB10 CUDA device.
if [[ "$(uname -m)" != aarch64 ]]; then
  echo "This image is arm64-only and must run on a DGX Spark." >&2
  exit 64
fi

python3 - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("A visible GB10 GPU is required")
capability = torch.cuda.get_device_capability(0)
if capability != (12, 1):
    raise SystemExit(f"Expected DGX Spark SM121, found compute capability {capability}")
PY

role=${DS4FV_ROLE:-head}
if [[ $# -gt 0 ]]; then
  case "$1" in
    head|ray-head|ray-worker|worker|serve|exl3)
      role=$1
      shift
      ;;
  esac
fi

ray_python=(python3 -m ray.scripts.scripts)
ray_head_ip=${RAY_HEAD_IP:-${VLLM_HOST_IP:-}}
ray_port=${RAY_PORT:-6379}
world_size=${DS4FV_WORLD_SIZE:-2}
ready_file=/tmp/ds4fv-release-ready

require_value() {
  local name=$1 value=${!1:-}
  if [[ -z "${value}" ]]; then
    echo "${name} must be set" >&2
    exit 64
  fi
}

check_fabric_env() {
  require_value VLLM_HOST_IP
  require_value RAY_HEAD_IP
  require_value NCCL_IB_HCA
  require_value NCCL_IB_GID_INDEX
  require_value NCCL_SOCKET_IFNAME
  require_value GLOO_SOCKET_IFNAME
}

configure_dspark_args() {
  local output_name=$1 default_tokens=${2:-5}
  local adaptive_verification draft_sample_method draft_tokens
  local -n output=${output_name}
  output=()
  case "${ENABLE_DSPARK:-1}" in
    0) ;;
    1)
      draft_sample_method=${DRAFT_SAMPLE_METHOD:-greedy}
      draft_tokens=${DSPARK_TOKENS:-${default_tokens}}
      adaptive_verification=${DSPARK_ADAPTIVE_VERIFICATION:-0}
      case "${draft_sample_method}" in
        probabilistic|greedy) ;;
        *)
          echo "DRAFT_SAMPLE_METHOD must be probabilistic or greedy" >&2
          exit 64
          ;;
      esac
      case "${adaptive_verification}" in
        0) adaptive_verification=false ;;
        1) adaptive_verification=true ;;
        *)
          echo "DSPARK_ADAPTIVE_VERIFICATION must be 0 or 1" >&2
          exit 64
          ;;
      esac
      output=(
        --speculative-config
        "{\"method\":\"dspark\",\"num_speculative_tokens\":${draft_tokens},\"draft_sample_method\":\"${draft_sample_method}\",\"enable_adaptive_verification\":${adaptive_verification}}"
      )
      ;;
    *)
      echo "ENABLE_DSPARK must be 0 or 1" >&2
      exit 64
      ;;
  esac
}

run_vllm_with_warmup() {
  local warmup_role=$1
  shift
  local server_pid warmup_status server_status

  rm -f "${ready_file}"
  "$@" &
  server_pid=$!

  forward_term() {
    kill -TERM "${server_pid}" 2>/dev/null || true
  }
  trap forward_term TERM INT

  case "${DS4FV_STARTUP_WARMUP:-1}" in
    1)
      set +e
      python3 /opt/ds4fv/bin/release-warmup \
        --server-pid "${server_pid}" \
        --base-url "http://127.0.0.1:${API_PORT:-8000}" \
        --role "${warmup_role}"
      warmup_status=$?
      set -e
      if [[ ${warmup_status} -ne 0 ]]; then
        printf 'DS4FV release startup warmup failed with status %s\n' \
          "${warmup_status}" >&2
        forward_term
        wait "${server_pid}" 2>/dev/null || true
        exit "${warmup_status}"
      fi
      ;;
    0) ;;
    *)
      echo "DS4FV_STARTUP_WARMUP must be 0 or 1" >&2
      forward_term
      wait "${server_pid}" 2>/dev/null || true
      exit 2
      ;;
  esac

  touch "${ready_file}"
  printf 'DS4FV release startup warmup complete; container is ready.\n'

  set +e
  wait "${server_pid}"
  server_status=$?
  set -e
  exit "${server_status}"
}

start_ray_head() {
  check_fabric_env
  "${ray_python[@]}" start \
    --head \
    --node-ip-address="${VLLM_HOST_IP}" \
    --port="${ray_port}" \
    --include-dashboard=false \
    --disable-usage-stats
}

wait_for_ray_gpus() {
  local attempts=${RAY_READY_ATTEMPTS:-180}
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if RAY_ADDRESS="${ray_head_ip}:${ray_port}" \
      EXPECTED_GPUS="${world_size}" python3 - <<'PY'
import os
import ray

ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
visible = int(ray.cluster_resources().get("GPU", 0))
expected = int(os.environ["EXPECTED_GPUS"])
raise SystemExit(0 if visible >= expected else 1)
PY
    then
      return 0
    fi
    sleep 2
  done
  echo "Ray did not report ${world_size} GPUs before the readiness deadline" >&2
  return 1
}

serve_model() {
  check_fabric_env

  local model_kind=${MODEL_KIND:-text}
  local dspark_default_tokens=5
  local moe_mode=${MOE_MODE:-tp}
  local model_repo model_revision served_name
  local -a model_args=() vision_args=() prefix_args=() moe_args=()
  local -a speculative_args=()
  case "${model_kind}" in
    text)
      model_repo=${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-0731}
      model_revision=${MODEL_REVISION:-9e165c30e2704aec5d9d593cce3eebd58bbef1cb}
      served_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-native}
      ;;
    vision)
      model_repo=${MODEL_REPO:-deepseek-ai/DeepSeek-V4-Flash-Vision-Exp}
      model_revision=${MODEL_REVISION:-86f746b36186f0e567729a5c06a8c918caba82a9}
      served_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp-native}
      vision_args=(
        --hf-overrides
        '{"architectures":["DeepseekV4VisionForConditionalGeneration"],"is_mm_prefix_lm":true,"vision_text_sliding_window":128,"sliding_window":512}'
        --disable-chunked-mm-input
        --mm-processor-cache-gb 0
        --limit-mm-per-prompt '{"image":16}'
      )
      dspark_default_tokens=3
      ;;
    *)
      echo "MODEL_KIND must be 'text' or 'vision', got '${model_kind}'" >&2
      exit 64
      ;;
  esac

  case "${moe_mode}" in
    tp) ;;
    ep) moe_args=(--enable-expert-parallel) ;;
    *)
      echo "MOE_MODE must be 'tp' or 'ep', got '${moe_mode}'" >&2
      exit 64
      ;;
  esac

  if [[ -n "${MODEL_PATH:-}" ]]; then
    model_args=("${MODEL_PATH}")
  else
    if [[ -z "${model_revision}" ]]; then
      echo "MODEL_REVISION is required for a repository-backed text launch" >&2
      exit 64
    fi
    model_args=("${model_repo}" --revision "${model_revision}")
  fi

  if [[ "${model_kind}" == vision || "${ENABLE_PREFIX_CACHING:-1}" != 1 ]]; then
    prefix_args=(--no-enable-prefix-caching)
  else
    prefix_args=(--enable-prefix-caching)
  fi
  configure_dspark_args speculative_args "${dspark_default_tokens}"

  run_vllm_with_warmup "native-${model_kind}" vllm serve "${model_args[@]}" \
    --served-model-name "${served_name}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --distributed-executor-backend ray \
    --tensor-parallel-size "${TP_SIZE:-2}" \
    "${moe_args[@]}" \
    --moe-backend b12x \
    --linear-backend b12x \
    --disable-custom-all-reduce \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --generation-config vllm \
    "${prefix_args[@]}" \
    "${vision_args[@]}" \
    "${speculative_args[@]}" \
    "$@"
}

serve_exl3() {
  local model_kind=${MODEL_KIND:-text}
  local dspark_default_tokens=5
  local model_repo model_revision served_name model_ref warmup_role
  local -a revision_args=() prefix_args=() speculative_args=() vision_args=()
  case "${model_kind}" in
    text)
      model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3}
      model_revision=${MODEL_REVISION:-7827301eed170e2a5e394f45a13cc66561c601ed}
      served_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-0731-exl3-k2.1-d2.2-v3}
      warmup_role=exl3
      ;;
    vision)
      model_repo=${MODEL_REPO:-wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1}
      model_revision=${MODEL_REVISION:-8aab722f04f7e8963af83de5acb16138474e0228}
      served_name=${SERVED_MODEL_NAME:-deepseek-v4-flash-vision-exp-exl3-k2.2-d2-v1}
      warmup_role=exl3-vision
      dspark_default_tokens=3
      vision_args=(
        --hf-overrides
        '{"architectures":["DeepseekV4VisionForConditionalGeneration"],"is_mm_prefix_lm":true,"vision_text_sliding_window":128,"sliding_window":512,"vision_n_layers":32,"vision_dim":1024,"vision_n_heads":16,"vision_inter_dim":2816,"vision_patch_size":14,"vision_rope_theta":10000.0,"vision_downsample_ratio":3,"vision_max_n_token":384,"vision_min_pixels":147456,"vision_max_wh_ratio":8}'
        --disable-chunked-mm-input
        --mm-processor-cache-gb 0
        --limit-mm-per-prompt '{"image":16}'
      )
      ;;
    *)
      echo "MODEL_KIND must be 'text' or 'vision', got '${model_kind}'" >&2
      exit 64
      ;;
  esac

  if [[ -n "${MODEL_PATH:-}" ]]; then
    model_ref=${MODEL_PATH}
  else
    model_ref=${model_repo}
    revision_args=(--revision "${model_revision}")
  fi
  if [[ "${model_kind}" == vision || "${ENABLE_PREFIX_CACHING:-1}" != 1 ]]; then
    prefix_args=(--no-enable-prefix-caching)
  else
    prefix_args=(--enable-prefix-caching)
  fi
  configure_dspark_args speculative_args "${dspark_default_tokens}"

  run_vllm_with_warmup "${warmup_role}" vllm serve "${model_ref}" \
    "${revision_args[@]}" \
    --served-model-name "${served_name}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --quantization exl3 \
    --load-format "${LOAD_FORMAT:-instanttensor}" \
    --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND:-uni}" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --moe-backend b12x \
    --linear-backend b12x \
    --max-model-len "${MAX_MODEL_LEN:-131072}" \
    --max-num-seqs "${MAX_NUM_SEQS:-4}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --generation-config vllm \
    --enable-chunked-prefill \
    "${prefix_args[@]}" \
    "${vision_args[@]}" \
    "${speculative_args[@]}" \
    "$@"
}

case "${role}" in
  head)
    start_ray_head
    wait_for_ray_gpus
    serve_model "$@"
    ;;
  ray-head)
    start_ray_head
    exec tail -f /dev/null
    ;;
  ray-worker|worker)
    check_fabric_env
    exec "${ray_python[@]}" start \
      --address="${ray_head_ip}:${ray_port}" \
      --node-ip-address="${VLLM_HOST_IP}" \
      --disable-usage-stats \
      --block
    ;;
  serve)
    wait_for_ray_gpus
    serve_model "$@"
    ;;
  exl3)
    serve_exl3 "$@"
    ;;
  *)
    echo "Unknown DS4FV role '${role}'; expected head, ray-head, worker, serve, or exl3" >&2
    exit 64
    ;;
esac
