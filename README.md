# DeepSeek V4 Flash/Vision on two DGX Sparks

This repository builds an **arm64-only, NVIDIA GB10 / SM121** vLLM image for
`deepseek-ai/DeepSeek-V4-Flash-0731` and
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Native checkpoints are served over
two DGX Sparks with tensor parallelism; an amd64 image is intentionally out of
scope until the EXL3 quantization work is complete.

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
| B12x | `tpurtell/sparkinfer-glmrt@1713e2acb8e810888e4be2545e4a31baf0667448` |
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

Launch the pinned native text checkpoint with TP2 experts:

```bash
MODEL_KIND=text \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev \
scripts/launch-two-spark.sh
```

Set `MOE_MODE=ep` to keep dense layers TP2 while distributing experts EP2.
It is a qualified option, but TP2 is the native default because EP2 was slower
on the retained concurrency-one prefill baseline. Vision uses the
checkpoint-specific architecture, 128-token text SWA, a
512-token physical image cache span, no multimodal processor cache, and no
prefix cache, with a hard 16-image request limit:

```bash
MODEL_KIND=vision MOE_MODE=tp \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev \
scripts/launch-two-spark.sh
```

The launcher replaces only its two exact named containers
(`ds4fv-native-head` and `ds4fv-native-worker` by default). Fabric merge and
cross-NIC policies remain environment overrides, but the defaults are now
qualified: merged dual rail with `NCCL_CROSS_NIC=2` was 4.84% faster than one
HCA on the matched native text TP2 prefill run.

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

## One-Spark EXL3 launch

The mixed K2/K3 EXL3 checkpoint fits on one Spark and uses the same image:

```bash
SPARK_HOST=kiwi \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev \
scripts/launch-one-spark-exl3.sh
```

Its default is native DSpark K5 with greedy drafting. The target-only and
explicit tuning controls above apply to this launcher as well. On the matched
one-Spark 256-input/128-output gate this profile reached 34.59 tok/s, 21.03%
above the older probabilistic-K5 result and 1.88% above vLLM's stock adaptive
K5; the [matched content qualification](validation/2026-09-02-exl3-greedy-adaptive.md)
also favored fixed K5 by 4.85%.

## Release benchmarks

The frozen-image harness is documented in
[benchmarks/README.md](benchmarks/README.md). It runs separate native Vision
TP2 and one-Spark EXL3 suites with code-agent decode/concurrency and context
depth curves, unique 8K--128K prefill, the weighted semantic/structured blend,
tool use, 128K retrieval, role-specific Vision or prefix-cache checks, and a
post-long-context C4 soak; full runs begin only after both roles use one exact
production-candidate image ID.

## Status

The combined image builds independently on both arm64 Sparks, and its B12x
TP/EP plus Vision contract smokes pass with CUDA hidden. Native Vision TP2 and
dense-TP2/MoE-EP2 both load all 48 shards, complete warmup/graph capture, and
serve image-sensitive requests across `ostrich` and `dodo`. On the retained
8,192+1, concurrency-one workload, TP2 reached 2,187.25 tok/s versus 1,947.72
tok/s for EP2, so TP2 is the default.

See the [no-GPU contract evidence](validation/2026-09-01-spark-no-gpu.md) and
[native Vision TP2/EP2 runtime evidence](validation/2026-09-01-spark-tp2-vision.md).
The [native text TP2 runtime](validation/2026-09-01-spark-tp2-text.md) also
loads all shards, serves correctly, and reaches 2,234.87 tok/s on its retained
8,192+1 baseline. The retained matched fabric test makes merged dual rail the
default, and the [compressed MLA qualification](validation/2026-09-02-b12x-compressed-mla.md)
records why its faster isolated B12x kernel remains opt-in. Reference-logit
comparison, longer reliability testing, and the frozen-image release suites
remain pending; the mixed K2/K3 microbenchmark and matched one-Spark full-model
gates have passed.
