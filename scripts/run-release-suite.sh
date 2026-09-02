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

case "${role}" in
  native-vision)
    dspark_tokens=${DSPARK_TOKENS:-3}
    ;;
  exl3-vision)
    dspark_tokens=${DSPARK_TOKENS:-3}
    ;;
  exl3)
    dspark_tokens=${DSPARK_TOKENS:-5}
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
if [[ -e "${output_root}" ]] && [[ -n "$(find "${output_root}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to overwrite non-empty output directory ${output_root}" >&2
  exit 64
fi
mkdir -p "${output_root}"

python3 - "${output_root}/manifest.json" "${role}" "${base_url}" "${model}" \
  "${image_id}" "${recipe_commit}" "${dspark_tokens}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, role, base_url, model, image_id, commit, tokens = sys.argv[1:]
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
  --repeats 5 \
  --orchid-warmups 1 \
  --require-contracts \
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
fi

python3 "${script_dir}/test-long-context.py" \
  "${common[@]}" \
  --tokens 128000 \
  --output "${output_root}/long-context-128k.json"

if [[ "${role}" == exl3 ]]; then
  python3 "${script_dir}/test-prefix-replay.py" \
    --base-url "${base_url}" \
    --model "${model}" \
    --image-id "${image_id}" \
    --recipe-commit "${recipe_commit}" \
    --tokens 128000 \
    --output "${output_root}/prefix-replay-128k.json"
fi

python3 "${script_dir}/soak-api.py" \
  "${common[@]}" \
  --concurrency 4 \
  --runs 20 \
  --output "${output_root}/post-long-context-c4-soak.json"

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
