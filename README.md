# DeepSeek V4 Flash/Vision on two DGX Sparks

This repository builds an **arm64-only, NVIDIA GB10 / SM121** vLLM image for
`deepseek-ai/DeepSeek-V4-Flash-0731` and
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. The no-option serving profile is
the one-Spark Vision EXL3 K2.2/D2 checkpoint with FP8 KV cache. Native
checkpoints use two DGX Sparks; an amd64 image is intentionally out of scope
until the EXL3 quantization work is complete.

The current milestone pins vLLM 0.28.0 and the current `tpurtell/sparkinfer-glmrt`
B12x tree. Vision support is implemented in-recipe because the experimental
checkpoint advertises the text-only `DeepseekV4ForCausalLM` architecture even
though it contains a 32-layer vision tower, multimodal router biases, and
bidirectional image-block attention.

## Safety boundary

Do not run this image, vLLM, or any GPU validation on the image-build
workstation. Build and runtime validation belong on the DGX Sparks
(`ostrich`, `dodo`, `kiwi`, or `emu`). Local checks in this repository are
source-only checks that do not import vLLM or initialize CUDA.

## Pinned inputs

| Component | Immutable input |
| --- | --- |
| Runtime base | `vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce` |
| vLLM | `0.28.0` (`2cf0a6915ce544dc493a0990f2ea38d81601128a`) |
| Ray | `2.48.0` |
| B12x | `tpurtell/sparkinfer-glmrt@3fc8d1491d1313c0ca64b2b95772972b7f42ee9d` |
| Native checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Vision checkpoint | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9` |

See [PROVENANCE.md](PROVENANCE.md) for package and checkpoint details and
[TUNING.md](TUNING.md) for the adaptation log and qualification status.

## Build

Run this on a Spark, never on the workstation:

```bash
docker build --platform linux/arm64 --progress=plain \
  --build-arg RECIPE_COMMIT="$(git rev-parse HEAD)" \
  -t ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev .
```

The Dockerfile rejects non-`aarch64` builds and sets the CuTe target to
`sm_121a`. The final image contains no amd64 build stage.

## Two-Spark launch

The launcher defaults to `ostrich` (`10.55.0.1`) plus `dodo` (`10.55.0.2`),
uses both active RoCE HCAs at GID index 3, and starts one Ray GPU worker per
Spark. The same image tag and Hugging Face cache path must exist on both nodes.

Check the fabric without starting containers:

```bash
scripts/launch-two-spark.sh --check-only
```

Launch the pinned native Vision checkpoint with TP2 across both Sparks. The
release default uses DCP1, so each rank retains a complete KV view:

```bash
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.0 \
scripts/launch-two-spark.sh
```

Set `MOE_MODE=ep` to keep dense layers TP2 while distributing experts EP2.
It is a qualified option, but TP2 is the native default because EP2 was slower
on the retained concurrency-one prefill baseline. Vision uses the
checkpoint-specific architecture, 128-token text SWA, a
512-token physical image cache span, no multimodal processor cache, prefix
caching, and a hard 16-image request limit. vLLM includes each multimodal
feature identifier and its block-relative position in the prefix hash, and the
release suite verifies both exact-image reuse and changed-image isolation:

```bash
MOE_MODE=tp \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.0 \
scripts/launch-two-spark.sh
```

The two-Spark profile defaults to `MAX_MODEL_LEN=500000` and `DCP_SIZE=1`.
This keeps APC enabled, avoids DCP communication on the latency-sensitive
decode path, and retains enough FP8 KV capacity for two maximum-length
requests. Set `DCP_SIZE=2` explicitly to use the retained experimental
rate-aware path. The launcher replaces only its two
exact named containers
(`ds4fv-native-head` and `ds4fv-native-worker` by default). Fabric merge and
cross-NIC policies remain environment overrides, but the defaults are now
qualified: merged dual rail with `NCCL_CROSS_NIC=2` was 4.84% faster than one
HCA on the matched native text TP2 prefill run.

The retained opt-in DCP2 implementation contains the required rank-local
length, slot-mapping, LSE merge, and hybrid-cache ownership corrections.
Replicating bounded SWA and C128 cache families while sharding C4 reduced its
repeated C1 target-pass overhead from 10.2% to 4.7% versus the matched
500K/APC DCP1 control, but DCP1 remained faster and is the release default.
`ag_rs` is the DCP2 communication default; set
`DCP_COMM_BACKEND=a2a` only for diagnostics, since it recovered just 1.2% at
C1 and reduced the measured KV pool. DCP-group query replication was also
tested and rejected: its extra small-M Q projection cost more than the removed
query all-gather and reduced both C1 throughput and KV capacity.

Native DSpark is enabled by default. Vision uses the qualified K3 depth (one
pass through its three next-token predictor layers), while native text
defaults to K5. K6's second predictor pass is not enabled by default because
it both reduced matched decode throughput and intermittently wedged repeated
16-image service. Drafting is greedy by default; use
`DRAFT_SAMPLE_METHOD=probabilistic` for stochastic drafting,
`DSPARK_TOKENS=<n>` for an explicitly qualified depth, or `ENABLE_DSPARK=0`
for a target-only control. Regular CUDA graphs are retained and the slower
breakable-graph mode is disabled by default. vLLM's stock adaptive verifier is
available with `DSPARK_ADAPTIVE_VERIFICATION=1`, but fixed-depth verification
remains the qualified default; this does not include ds4rt's online
request-local residual controller.

The shared B12x compressed sparse MLA decode adapter is retained as an opt-in
experiment with `DS4FV_USE_B12X_COMPRESSED_MLA=1`. It wins the isolated exact
kernel workload, but the matched full-model content suite was 1.62% slower, so
the qualified default remains vLLM's existing split decode path.

Both serving roles expose DeepSeek V4's native tokenizer, reasoning parser,
and automatic tool parser. The release suite includes a deterministic tool-call
contract so these API paths cannot silently regress while kernel work changes.

Docker health is deliberately stricter than vLLM's internal `/health`: the
container remains unready until real requests have exercised DSpark's exact
shape buckets, rendered greedy C1/C2/C4, chunk-crossing prefill,
structured/tool parsing, and (for Vision) 1/4/16-image paths. Set
`DS4FV_STARTUP_WARMUP=0` only for development diagnostics; such a run is not a
release candidate. Triton, TileLang, B12x, and FlashInfer JIT caches persist on
the mounted Hugging Face cache volume.

## Default one-Spark Vision EXL3 launch

The new mixed Vision K2.2/D2 checkpoint fits on one Spark and uses the same
image. This is the recipe's no-option model profile; FP8 KV, 500K maximum model
length, prefix caching, and greedy K3 drafting are the defaults:

```bash
SPARK_HOST=kiwi \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.0 \
scripts/launch-one-spark-exl3.sh
```

It serves as `deepseek-v4-flash-vision-exp-exl3-k2.2-d2-v1` by default.
The older text K2.1 checkpoint remains available with `MODEL_KIND=text`, but it
is not part of the final performance/quality evidence matrix. Any compatible
profile may opt into NVFP4 DS-MLA with `KV_CACHE_DTYPE=nvfp4_ds_mla`; the
backend fails closed when the required DeepSeek-V4/B12x path is unavailable.
Only the primary Vision EXL3 profile is being qualified for release capacity,
quality, and performance claims. FP8 is its qualified production default;
NVFP4 is the qualified higher-capacity option.

## Release benchmarks

The measured results are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md), and
the frozen-image harness is documented in [benchmarks/README.md](benchmarks/README.md).
It measures native Vision
TP2+DCP1 with FP8 and the primary one-Spark Vision EXL3 model with matched FP8
and NVFP4 runs. The suites cover code-agent decode/concurrency and context
depth curves, unique 8K--128K prefill, the weighted semantic/structured blend,
tool use, 128K retrieval, role-specific Vision or prefix-cache checks, and a
post-long-context C4 soak. The harness normally freezes one exact candidate
image ID; the transparent `v0.1.0` final-digest delta exception and exact
provenance are documented with the results.

## Status

Release `v0.1.0` is qualified for arm64 GB10/SM121. The exact release image
`sha256:dcafc6bf649d70a014ff4350eba85cd7e721dec0ecb9a24ea38bd58401ffe8bd`
passed final startup, structured/tool, Vision/APC, decode-envelope, and
post-ready JIT checks. Native Vision TP2+DCP1 reached 51.38 tok/s at C1,
137.22 tok/s aggregate at C4, and 2,081 prompt tok/s at 8K; the one-Spark
Vision EXL3 default reached 39.66/58.28/130.24 tok/s at C1/C2/C4 with FP8 KV.
On the exact release image, NVFP4 increased the one-Spark physical KV pool
from 1.762M to 2.039M tokens (+15.7%) while retaining the same 500K request
limit.

See the [no-GPU contract evidence](validation/2026-09-01-spark-no-gpu.md) and
[native Vision TP2/EP2 runtime evidence](validation/2026-09-01-spark-tp2-vision.md).
The [native text TP2 runtime](validation/2026-09-01-spark-tp2-text.md) also
loads all shards, serves correctly, and reaches 2,234.87 tok/s on its retained
8,192+1 baseline. The retained matched fabric test makes merged dual rail the
default, and the [compressed MLA qualification](validation/2026-09-02-b12x-compressed-mla.md)
records why its faster isolated B12x kernel remains opt-in. The mixed K2/K3
microbenchmark and matched one-Spark full-model gates passed. Reference-logit
comparison and destructive worker/fabric recovery testing remain future work;
they are not claims of this release.
