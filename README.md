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
| B12x | `tpurtell/sparkinfer-glmrt@a13677130cd144772bc7528238fe2244bbe3d0d4` |
| Native checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Vision checkpoint | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9` |

See [PROVENANCE.md](PROVENANCE.md) for package and checkpoint details and
[TUNING.md](TUNING.md) for the adaptation log and qualification status.

## Build

Run this on a Spark, never on the workstation:

```bash
docker build --platform linux/arm64 --progress=plain \
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
prefix cache:

```bash
MODEL_KIND=vision MOE_MODE=tp \
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev \
scripts/launch-two-spark.sh
```

The launcher replaces only its two exact named containers
(`ds4fv-native-head` and `ds4fv-native-worker` by default). Fabric merge and
cross-NIC policies remain environment overrides until benchmark evidence
selects final defaults.

## Status

The combined image builds independently on both arm64 Sparks, and its B12x
TP/EP plus Vision contract smokes pass with CUDA hidden. Native Vision TP2 and
dense-TP2/MoE-EP2 both load all 48 shards, complete warmup/graph capture, and
serve image-sensitive requests across `ostrich` and `dodo`. On the retained
8,192+1, concurrency-one workload, TP2 reached 2,187.25 tok/s versus 1,947.72
tok/s for EP2, so TP2 is the default.

See the [no-GPU contract evidence](validation/2026-09-01-spark-no-gpu.md) and
[native Vision TP2 runtime evidence](validation/2026-09-01-spark-tp2-vision.md).
The native text checkpoint, matched single/dual-rail tests, reference-logit
comparison, and the eventual EXL3 one-Spark mixed-layer qualification remain
pending. The native TP2 result is not a substitute for the required equal-work
mixed-K2/K3 versus uniform-K2 comparison.
