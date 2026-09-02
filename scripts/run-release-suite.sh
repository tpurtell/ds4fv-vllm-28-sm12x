#!/usr/bin/env bash
set -Eeuo pipefail

# HTTP-only release orchestrator. It never starts vLLM, Docker, or GPU code;
# point it at a service already launched on the frozen Spark image.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
role=${ROLE:?ROLE must be native-vision, exl3, or exl3-vision}
base_url=${BASE_URL:-http://127.0.0.1:8000}
model=${MODEL:?MODEL must be the served model name}
image_id=${IMAGE_ID:?IMAGE_ID must be the frozen sha256 image ID}
recipe_commit=${RECIPE_COMMIT:-$(git -C "${script_dir}/.." rev-parse HEAD)}
stamp=${BENCHMARK_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
output_root=${OUTPUT_ROOT:-${script_dir}/../benchmarks/${stamp}-${role}}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
tool_eval_reference_date=${TOOL_EVAL_REFERENCE_DATE:-$(date -u +%F)}
tool_eval_required_version=${TOOL_EVAL_REQUIRED_VERSION:-2.3.2.dev3+g5df1e9e0c}

case "${role}" in
  native-vision)
    dspark_tokens=${DSPARK_TOKENS:-3}
    content_contract_floor=38
    ;;
  exl3-vision)
    dspark_tokens=${DSPARK_TOKENS:-3}
    content_contract_floor=38
    ;;
  exl3)
    dspark_tokens=${DSPARK_TOKENS:-5}
    content_contract_floor=34
    ;;
  *)
    echo "ROLE must be native-vision, exl3, or exl3-vision" >&2
    exit 64
    ;;
esac
if [[ ! "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "IMAGE_ID must be a full sha256:<64 hex> image ID" >&2
  exit 64
fi
if [[ ! "${recipe_commit}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "RECIPE_COMMIT must be a full 40-hex commit" >&2
  exit 64
fi

tool_eval_cmd=()
if [[ -n "${TOOL_EVAL_BENCH:-}" ]]; then
  tool_eval_cmd=("${TOOL_EVAL_BENCH}")
elif command -v tool-eval-bench >/dev/null 2>&1; then
  tool_eval_cmd=(tool-eval-bench)
elif command -v uv >/dev/null 2>&1 \
  && [[ -f "${script_dir}/../../tool-eval-bench/pyproject.toml" ]]; then
  tool_eval_cmd=(
    uv run --project "${script_dir}/../../tool-eval-bench" tool-eval-bench
  )
else
  echo "Full release qualification requires tool-eval-bench." >&2
  echo "Install it or set TOOL_EVAL_BENCH to its executable path." >&2
  exit 69
fi
if ! tool_eval_version_output=$("${tool_eval_cmd[@]}" --version 2>&1); then
  echo "Unable to identify the selected tool-eval-bench executable:" >&2
  echo "${tool_eval_version_output}" >&2
  exit 69
fi
tool_eval_version=${tool_eval_version_output#tool-eval-bench }
if [[ "${tool_eval_version}" != "${tool_eval_required_version}" ]]; then
  echo "Release qualification requires tool-eval-bench ${tool_eval_required_version}; selected ${tool_eval_version}." >&2
  echo "Run the suite from the qualification workstation or set TOOL_EVAL_BENCH to the exact executable." >&2
  exit 69
fi
if [[ -e "${output_root}" ]] && [[ -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output directory ${output_root}" >&2
  exit 64
fi
mkdir -p "${output_root}"

python3 - "${output_root}/manifest.json" "${role}" "${base_url}" "${model}" \
  "${image_id}" "${recipe_commit}" "${dspark_tokens}" \
  "${content_contract_floor}" "${kv_cache_dtype}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    path,
    role,
    base_url,
    model,
    image_id,
    commit,
    tokens,
    contract_floor,
    kv_cache_dtype,
) = sys.argv[1:]
Path(path).write_text(json.dumps({
    "schema": "ds4fv-release-suite-manifest.v1",
    "started_utc": datetime.now(timezone.utc).isoformat(),
    "role": role,
    "base_url": base_url,
    "model": model,
    "image_id": image_id,
    "recipe_commit": commit,
    "dspark_tokens": int(tokens),
    "dspark_policy": "fixed",
    "draft_sample_method": "greedy",
    "content_contract_floor": int(contract_floor),
    "kv_cache_dtype": kv_cache_dtype,
}, indent=2) + "\n")
PY

common=(
  --base-url "${base_url}"
  --model "${model}"
  --role "${role}"
  --image-id "${image_id}"
  --recipe-commit "${recipe_commit}"
)

python3 "${script_dir}/benchmark-decode.py" \
  "${common[@]}" \
  --dspark-tokens "${dspark_tokens}" \
  --dspark-policy fixed \
  --draft-sample-method greedy \
  --output "${output_root}/code-agent-decode.json"

python3 "${script_dir}/benchmark-prefill.py" \
  "${common[@]}" \
  --output "${output_root}/cold-prefill.json"

python3 "${script_dir}/benchmark-content-types.py" \
  "${common[@]}" \
  --dspark-tokens "${dspark_tokens}" \
  --dspark-policy fixed \
  --draft-sample-method greedy \
  --repeats 5 \
  --orchid-warmups 1 \
  --minimum-contract-passes "${content_contract_floor}" \
  --output "${output_root}/content-types.json"

python3 "${script_dir}/test-tool-call.py" \
  "${common[@]}" \
  --output "${output_root}/tool-call.json"

if [[ "${role}" == native-vision || "${role}" == exl3-vision ]]; then
  python3 "${script_dir}/test-native-vision-vllm.py" \
    --base-url "${base_url}" \
    --model "${model}" \
    --image-id "${image_id}" \
    --recipe-commit "${recipe_commit}" \
    --image-counts 1 4 16 \
    --reject-image-count 17 \
    --image-limit 16 \
    --output "${output_root}/native-vision.json"

  python3 "${script_dir}/test-vision-prefix-replay.py" \
    --base-url "${base_url}" \
    --model "${model}" \
    --role "${role}" \
    --image-id "${image_id}" \
    --recipe-commit "${recipe_commit}" \
    --output "${output_root}/vision-prefix-replay.json"
fi

python3 "${script_dir}/test-long-context.py" \
  "${common[@]}" \
  --tokens 128000 \
  --output "${output_root}/long-context-128k.json"

python3 "${script_dir}/test-prefix-replay.py" \
  --base-url "${base_url}" \
  --model "${model}" \
  --role "${role}" \
  --image-id "${image_id}" \
  --recipe-commit "${recipe_commit}" \
  --tokens 128000 \
  --output "${output_root}/prefix-replay-128k.json"

python3 "${script_dir}/soak-api.py" \
  "${common[@]}" \
  --concurrency 4 \
  --runs 20 \
  --output "${output_root}/post-long-context-c4-soak.json"

# The full default 69-scenario matrix is a release artifact. Do not pass
# --short or --hardmode: the stable score contract is exactly 138 points.
"${tool_eval_cmd[@]}" \
  --model "${model}" \
  --backend vllm \
  --base-url "${base_url%/}/v1/" \
  --parallel 1 \
  --temperature 0.0 \
  --seed 0 \
  --reference-date "${tool_eval_reference_date}" \
  --timeout 300 \
  --max-turns 8 \
  --json-file "${output_root}/tool-eval-bench.json" \
  --output-dir "${output_root}/tool-eval-reports" \
  --no-live

python3 - "${output_root}/manifest.json" \
  "${output_root}/tool-eval-bench.json" "${tool_eval_required_version}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
required_version = sys.argv[3]
manifest = json.loads(manifest_path.read_text())
result = json.loads(result_path.read_text())
scores = result.get("scores", {})
if result.get("tool_eval_bench_version") != required_version:
    raise SystemExit(
        "tool-eval-bench result version mismatch: "
        f"{result.get('tool_eval_bench_version')!r} != {required_version!r}"
    )
if result.get("status") != "completed":
    raise SystemExit("tool-eval-bench did not complete")
if result.get("total_scenarios") != 69:
    raise SystemExit("tool-eval-bench must execute all 69 default scenarios")
if scores.get("max_points") != 138:
    raise SystemExit("tool-eval-bench score contract changed from /138")
categories = scores.get("category_scores", [])
manifest["tool_eval_bench"] = {
    "artifact": result_path.name,
    "version": result.get("tool_eval_bench_version"),
    "run_id": result.get("run_id"),
    "scenario_count": result["total_scenarios"],
    "points": scores.get("total_points"),
    "max_points": scores["max_points"],
    "normalized_score": result.get("final_score"),
    "rating": result.get("rating"),
    "pass_count": sum(item.get("pass_count", 0) for item in categories),
    "partial_count": sum(item.get("partial_count", 0) for item in categories),
    "fail_count": sum(item.get("fail_count", 0) for item in categories),
    "status": result["status"],
}
manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

python3 - "${output_root}/manifest.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
manifest["passed"] = True
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

printf 'Release suite passed; artifacts: %s\n' "${output_root}"
