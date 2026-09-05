# Frozen-image release qualification

Release measurements record a full Docker image ID and 40-character recipe
commit in every receipt; a mutable tag is not release identity. The v0.1.1
matrix reruns every profile on one exact post-0.28-corrected candidate because
the RoPE and router fixes can change model numerics. Exact provenance is
recorded in [RESULTS.md](RESULTS.md).

The workstation is an HTTP client only. It must never start this image, vLLM,
or GPU code.

## Published results

See [RESULTS.md](RESULTS.md) for the release tables and links to every retained
JSON receipt.

## Release evidence profiles

| Role | Topology | KV cache | DSpark | Prefix cache | Vision limit |
| --- | --- | --- | --- | --- | --- |
| Official Vision | two SM121 Sparks, TP2+DCP1, merged dual rail | FP8 | fixed greedy K3 | on | 16 images |
| Vision EXL3 pure K2 | one SM121 Spark | FP8 | fixed greedy K3 | on | 16 images |
| Vision EXL3 K2.2/D2 | one SM121 Spark, mixed K2/K3 | FP8 | fixed greedy K3 | on | 16 images |

All profiles expose a 500,000 maximum model length and four scheduler slots.
K2.2/D2 is the release-qualified default. The pure-K2 and official-model
profiles are complete comparison suites whose strict model-behavior misses are
reported in `RESULTS.md`; running every measurement does not relabel a failed
contract as passed.
Vision EXL3 uses a 2,048-token batch budget; the smaller chunk bounds transient
compressor state and measured slightly faster rather than trading away
prefill. FP8 is the only v0.1.1 qualified KV format; NVFP4 remains selectable
for experimentation but has no v0.1.1 release claim. The stock adaptive
verifier is off. The service
must receive no unrelated traffic during a suite because DSpark acceptance is
read from process-wide Prometheus counter deltas.

### Historical v0.1.0 auxiliary K6 content profile

The frozen one-Spark image was also measured with fixed greedy K6 as an
auxiliary content profile. This does not replace the qualified K3 default.
Each entry is the median decode tok/s from five samples; structured normal and
constrained arms retain half weight each in the aggregate score.

| KV / arm | K3 | K6 | K6 delta |
| --- | ---: | ---: | ---: |
| FP8 weighted content score | 33.03 | 32.41 | -1.88% |
| FP8 code | 40.89 | 40.66 | -0.55% |
| FP8 reasoning | 40.61 | 41.36 | +1.86% |
| FP8 creative prose | 24.87 | 21.27 | -14.47% |
| FP8 short response | 36.93 | 43.63 | +18.14% |
| FP8 exposition | 26.20 | 20.32 | -22.47% |
| FP8 structured JSON, normal | 37.68 | 39.03 | +3.59% |
| FP8 structured JSON, constrained | 37.45 | 41.49 | +10.80% |
| FP8 multilingual | 24.15 | 19.36 | -19.83% |
| FP8 Orchid | 50.77 | 70.29 | +38.45% |
| NVFP4 weighted content score | 32.85 | 31.38 | -4.46% |
| NVFP4 code | 41.35 | 39.97 | -3.32% |
| NVFP4 reasoning | 39.29 | 39.94 | +1.65% |
| NVFP4 creative prose | 23.71 | 19.81 | -16.45% |
| NVFP4 short response | 36.80 | 43.32 | +17.70% |
| NVFP4 exposition | 25.37 | 18.55 | -26.89% |
| NVFP4 structured JSON, normal | 37.58 | 39.00 | +3.78% |
| NVFP4 structured JSON, constrained | 38.26 | 37.18 | -2.81% |
| NVFP4 multilingual | 25.48 | 19.99 | -21.54% |
| NVFP4 Orchid | 50.67 | 69.95 | +38.05% |

FP8 K6 passed all 40 semantic contracts and all 10 structured samples. NVFP4
K6 passed structured 10/10 but scored 37/40 overall because three exposition
samples missed the exact formatting contract; its K3 release profile remains
40/40. The raw K6 and compact K3/K6 comparison JSON files live beside each
primary one-Spark manifest.

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
- The complete default `tool-eval-bench` `2.3.2.dev3+g5df1e9e0c` matrix:
  TC-01 through TC-69,
  thinking enabled, greedy temperature, one request at a time, and the stable
  138-point score contract. The runner records points, normalized score, and
  pass/partial/fail counts in both `tool-eval-bench.json` and the manifest.
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

Run it twice with `ROLE=exl3-vision` against the one-Spark APIs: once for the
pure-K2 served model and once for K2.2/D2. `run-release-suite.sh` recognizes
those two model names plus the official Vision name and rejects any non-FP8
v0.1.1 evidence run.
The runner refuses a non-empty output directory and marks its manifest passed
only after every command exits successfully. It resolves `tool-eval-bench`
from `PATH`, the adjacent `../tool-eval-bench` checkout through `uv`, or an
explicit `TOOL_EVAL_BENCH=/path/to/tool-eval-bench`. It fails before creating
release evidence unless that executable reports the pinned dev3 version above;
`TOOL_EVAL_REQUIRED_VERSION` is available only for an intentional future
matrix-version update. Run release qualification from the workstation so an
older Spark-local installation cannot silently score the model.

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
