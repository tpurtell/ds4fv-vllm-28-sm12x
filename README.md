# DeepSeek V4 Flash/Vision on one DGX Spark

This repository builds an **arm64-only, NVIDIA GB10 / SM121** vLLM image whose
primary deployment is the one-Spark Vision EXL3 K2.2/D2 checkpoint with FP8 KV
cache. The same build can run the official native DeepSeek V4 Flash/Vision
checkpoints across two Sparks for direct evaluation and quality comparison; the
two-Spark profile is not the primary deployment. An amd64 image is intentionally
out of scope until the EXL3 quantization work is complete.

The current milestone pins vLLM 0.28.0 and the current `tpurtell/sparkinfer-glmrt`
B12x tree. Vision support is implemented in-recipe because the experimental
checkpoint advertises the text-only `DeepseekV4ForCausalLM` architecture even
though it contains a 32-layer vision tower, multimodal router biases, and
bidirectional image-block attention.

## One-Spark quick start

The default profile serves
`wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1` with FP8 KV,
a 500K maximum model length, prefix caching, and fixed greedy K3 drafting.
Clone this repository directly on an SM121 DGX Spark and launch it locally:

```bash
git clone https://github.com/tpurtell/ds4fv-vllm-28-sm12x.git
cd ds4fv-vllm-28-sm12x
scripts/launch-one-spark-exl3.sh
docker logs -f ds4fv-exl3
```

No environment variables are required. The launcher runs Docker locally,
pulls `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1` if necessary, and uses
`${XDG_CACHE_HOME:-$HOME/.cache}/huggingface` for the persistent model and
compile cache. The first launch can download the large K2.2/D2 checkpoint;
subsequent launches reuse it. Set `HF_CACHE=/another/path` to use an existing
cache elsewhere.

Once startup warmup completes, the OpenAI-compatible API is available on port
8000. Check it locally with:

```bash
curl http://127.0.0.1:8000/v1/models
```

If the checkpoint is already complete and the Spark must remain offline, use
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The pure-K2 Vision checkpoint is
also available as the measured one-Spark quantization comparison:

```bash
EXL3_PROFILE=k2 \
scripts/launch-one-spark-exl3.sh
```

To launch on a Spark over SSH instead, set its hostname explicitly. All Docker
and GPU activity still occurs on that Spark:

```bash
SPARK_HOST=kiwi scripts/launch-one-spark-exl3.sh
ssh kiwi docker logs -f ds4fv-exl3
```

The older text K2.1 checkpoint remains available with `MODEL_KIND=text`, but
it is not part of the final performance/quality evidence matrix. Any compatible
profile may opt into NVFP4; the backend fails closed when the required
DeepSeek-V4/B12x path is unavailable. FP8 is the only KV-cache format in the
v0.1.1 qualification matrix; NVFP4 remains an experimental user-selectable
option rather than a v0.1.1 capacity, quality, or performance claim.

## Safety boundary

Do not run this image, vLLM, or any GPU validation on a non-Spark workstation.
When the repository is cloned on a DGX Spark, the one-Spark launcher runs
locally there. From another machine, set `SPARK_HOST` so every Docker and GPU
action occurs on the selected Spark. Local workstation checks in this
repository are source-only and do not import vLLM or initialize CUDA.

## Pinned inputs

| Component | Immutable input |
| --- | --- |
| Runtime base | `vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce` |
| vLLM | `0.28.0` (`2cf0a6915ce544dc493a0990f2ea38d81601128a`) |
| Ray | `2.48.0` |
| B12x | `tpurtell/sparkinfer-glmrt@3fc8d1491d1313c0ca64b2b95772972b7f42ee9d` |
| Native comparison checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| Native Vision comparison checkpoint | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9` |
| Pure-K2 Vision EXL3 checkpoint | `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1@419697c409cb4157471bcaf68be07dbd151b0a40` |
| K2.2/D2 Vision EXL3 checkpoint | `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1@8aab722f04f7e8963af83de5acb16138474e0228` |

See [PROVENANCE.md](PROVENANCE.md) for package and checkpoint details and
[TUNING.md](TUNING.md) for the adaptation log and qualification status.

v0.1.1 keeps the pinned 0.28.0 base but fail-closed backports the later
DeepSeek fixes that affect this workload: layer-specific main/compressor RoPE
(#54815), FP32 SM121 router output (#54048), allocator-owned eager temporaries
(#52836), DeepSeek tool-stream and template corrections (#54838, #48922, and
#51262), termination-safe speculative structured output (#52805),
and the relevant DCP slot/cache-group invariants (#51031 and #54277).
InstantTensor is pinned to its buffered loader with I/O depth 128 so the large
Vision derivatives do not hit the direct-read tensor-boundary failure.

Check the public tag, platform, digest, and embedded source revision without
downloading image layers:

```bash
python3 scripts/check-release-image.py \
  ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1 \
  --expected-digest sha256:4c2c85052dac8f268a7fa15ec75d86cd1001c37cb96bb685eb91b889e6550511 \
  --expected-revision 6b940202b5ac9d38bb1af198c183f1ada513442a
```

## Build

Run this on a Spark, never on the workstation:

```bash
docker build --platform linux/arm64 --progress=plain \
  --build-arg RECIPE_COMMIT="$(git rev-parse HEAD)" \
  -t ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:native-dev .
```

The Dockerfile rejects non-`aarch64` builds and sets the CuTe target to
`sm_121a`. The final image contains no amd64 build stage.

## Native official-model evaluation (two Sparks)

This evaluator runs the official unquantized checkpoint with the same image
used above, allowing direct performance and quality comparisons against the
one-Spark quantized model. It is not the primary serving recipe. The launcher
defaults to `ostrich`
(`10.55.0.1`) plus `dodo` (`10.55.0.2`), uses both active RoCE HCAs at GID
index 3, and starts one Ray GPU worker per Spark. The same image tag and Hugging
Face cache path must exist on both nodes.

Check the fabric without starting containers:

```bash
scripts/launch-two-spark.sh --check-only
```

Launch the pinned native Vision checkpoint with TP2 across both Sparks. The
evaluation profile defaults to DCP1, so each rank retains a complete KV view:

```bash
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1 \
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
DS4FV_IMAGE=ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1 \
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

## Release benchmarks

The measured results are in [benchmarks/RESULTS.md](benchmarks/RESULTS.md), and
the frozen-image harness is documented in [benchmarks/README.md](benchmarks/README.md).
It measures pure-K2 and K2.2/D2 one-Spark Vision EXL3 checkpoints with FP8,
plus native Vision TP2+DCP1 with FP8 as the same-build official-model
comparison. The suites cover code-agent decode/concurrency and context
depth curves, unique 8K--128K prefill, the weighted semantic/structured blend,
tool use, 128K retrieval, role-specific Vision or prefix-cache checks, and a
post-long-context C4 soak. All three v0.1.1 suites use one exact candidate image
ID and the workstation's pinned 69-case/138-point tool evaluator.

## Status

Release `v0.1.1` is qualified for arm64 GB10/SM121. Its default one-Spark
K2.2/D2 FP8 profile passed the complete release gate on exact image
`sha256:d8a8d361adc3b81b7939fc487c97baa84e520201f1a31269b2dc0f100d94c3ee`,
including 128K retrieval/replay, Vision cache isolation, tool/structured
output, a 20-run C4 soak, and zero post-ready JIT. It measured
39.38/57.75/127.25 tok/s at C1/C2/C4, 1,321 prompt tok/s at 8K, and a
1,773,796-token physical KV pool.

The same image was fully measured with pure K2 on one Spark and official
Vision TP2+DCP1 on two Sparks. They provide the requested quantization and
official-checkpoint comparisons, but are not labeled as passing every strict
behavior gate: pure K2 scored 34/40 semantic contracts and missed one changed-
image digit without a cache collision; official Vision scored 36/40 semantic
contracts. Exact results and raw receipts are in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md). NVFP4 remains user-selectable
but is outside the v0.1.1 qualification matrix.

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

The later #53046 post-reasoning speculative-token validation backport is in
repository `main` after the frozen image revision. It is not claimed as part of
the v0.1.1 image above.
