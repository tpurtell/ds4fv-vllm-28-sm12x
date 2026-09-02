# Frozen-image release qualification

Release measurements are run only after one committed arm64 image has been
built on a DGX Spark and replicated byte-for-byte to every Spark used by the
service. Both the native Vision FP8 TP2+DCP2 suite and the one-Spark mixed
Vision EXL3 FP8/NVFP4 suites
must name the same full Docker image ID and 40-character recipe commit in every
receipt; a mutable tag is not release identity.

The workstation is an HTTP client only. It must never start this image, vLLM,
or GPU code.

## Release evidence profiles

| Role | Topology | KV cache | DSpark | Prefix cache | Vision limit |
| --- | --- | --- | --- | --- | --- |
| `native-vision` | two SM121 Sparks, TP2+DCP2, merged dual rail | FP8 | fixed greedy K3 | on | 16 images |
| `exl3-vision` | one SM121 Spark, mixed K2/K3 | FP8 | fixed greedy K3 | on | 16 images |
| `exl3-vision` matched variant | one SM121 Spark, mixed K2/K3 | NVFP4 DS-MLA | fixed greedy K3 | on | 16 images |

All profiles expose a 500,000 maximum model length and four scheduler slots.
Vision EXL3 uses a 2,048-token batch budget; the smaller chunk bounds transient
compressor state and measured slightly faster rather than trading away
prefill. NVFP4 uses the B12x compact sparse-MLA path and must prove a material
physical KV-capacity increase over the matched FP8 run. The stock adaptive
verifier is off. The service
must receive no unrelated traffic during a suite because DSpark acceptance is
read from process-wide Prometheus counter deltas.

## HTTP suite

[`run-release-suite.sh`](../scripts/run-release-suite.sh) executes the same
core workload for each role:

- Code-agent pure decode at C1, C2, and C4: 256 tokens per sequence, two
  warmups, five measured runs, fixed seed, and per-sequence first-to-last-token
  timing. It also measures C1 decode after 0, 8K, 32K, 64K, and 128K existing
  context with one warmup and three measured runs.
- Cold unique C1 prefill at exact 8K, 16K, 32K, 64K, and 128K lengths: each
  depth is independently warmed and then measured three times through TTFT.
- Five samples of all seven semantic workload categories. Normal JSON and
  constrained `response_format` JSON count at 0.5 each, every visible contract
  must pass, and the pure Orchid stream is retained separately as the
  low-entropy maximum-speed arm.
- One exact DeepSeek V4 `get_weather({"location":"Berlin"})` tool-call
  contract, followed by a cold exact-128K six-needle retrieval and a 20-request
  C4 post-long-context soak.
- Both Vision roles additionally read ordered numbered fixtures at 1, 4, and
  16 images, reject image 17 with HTTP 400, record real cache hits on an exact
  multimodal replay, and correctly read a changed-image request sharing the
  same text and cache salt. Every role also repeats one exact 128K text prompt
  and must record real prefix-cache hits on the second pass.

Example, after independently verifying the service is running the frozen
image on the Sparks:

```bash
ROLE=native-vision \
BASE_URL=http://10.55.0.1:8000 \
MODEL=deepseek-v4-flash-vision-exp-native \
IMAGE_ID=sha256:<64-hex-image-id> \
RECIPE_COMMIT=<40-hex-commit> \
scripts/run-release-suite.sh
```

Run it again with `ROLE=exl3-vision` against the primary one-Spark API, once
with `KV_CACHE_DTYPE=fp8` and once with `KV_CACHE_DTYPE=nvfp4_ds_mla`.
The runner refuses a non-empty output directory and marks its manifest passed
only after every command exits successfully.

## Non-HTTP release evidence

Before either HTTP run, retain Docker inspect output from every participating
Spark, prove all nodes resolve the identical image ID, record the exact model
revision and launch environment, and run the CUDA-hidden B12x/EXL3/Vision
contract smokes inside that image. Retain startup and post-suite logs from all
ranks, including model memory, KV capacity, graph capture, and any runtime
compilation; a release cannot claim post-ready JIT-free operation unless the
logs actually establish it.

The service container is not ready merely because vLLM's internal `/health`
endpoint responds. Docker health remains gated on the entrypoint's exact
`DS4FV release startup warmup complete; container is ready.` marker after the
real-path shape sweep succeeds. Run `scripts/audit-startup-jit.py` on the Spark
after a post-ready diagnostic and again after the suite; `post_ready_jit_count`
must remain zero.

The release gate is zero request failures, every semantic/tool/Vision/long-
context contract passing, a healthy completed soak, and no newly compiled
runtime shape after readiness. Performance must also exceed the corresponding
Mia image on matched workloads; the older native Vision reference points of
roughly 2,300 prompt tok/s, 58 decode tok/s, and 65 tok/s on the low-entropy
arm are comparison targets, not results from this image.
