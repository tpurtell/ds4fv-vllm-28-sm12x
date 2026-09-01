# Native Vision TP2 runtime validation

Date: 2026-09-01 (Asia/Taipei)

Nodes: `ostrich` (TP rank 0) and `dodo` (TP rank 1), both DGX Spark GB10 / SM121.
No vLLM process or GPU code ran on the image-build workstation.

## Qualified configuration

- Image: `ds4fv-vllm-28-sm12x:recipe-check`
  - `ostrich`: `sha256:3ad9d0c73f448ab8d82c944c970ae652dfe2f9a125aae856fd54cd2f43177edd`
  - `dodo`: `sha256:5c65b485c4b73d35c468344ad77435cf81720cf0bb140d2975a71cc1aef6c51c`
- Checkpoint:
  `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp@86f746b36186f0e567729a5c06a8c918caba82a9`
- Dense tensor parallelism: 2; experts: TP2; B12x linear and MoE backends.
- Dual RoCE HCAs: `=rocep1s0f0,roceP2p1s0f0`; GID index 3;
  `NCCL_IB_MERGE_NICS=1`; `NCCL_CROSS_NIC=2`.
- Maximum model length 131,072; maximum 8,192 batched tokens; four sequences;
  FP8 KV cache; prefix and multimodal processor caches disabled.

## Kernel qualification

The standalone B12x numeric test exercised TP2's exact local Vision envelope:
32 heads, a 512-wide primary sparse section, 512 compressed entries, 64-token
pages, and mixed valid lengths of `[384, 128]` and `[512, 257]`. Against the
pinned B12x independent PyTorch oracle, both Sparks produced:

```text
cosine similarity   0.99984747
output max abs      0.00061035
LSE max abs         0.00000191
```

The public B12x dispatcher selected the BF16-QK DSV4 multi-group kernel. The
full model subsequently crossed this same 512+512 prefill path during warmup
and live multimodal requests.

## Full-model startup

Both ranks loaded all 48 safetensor shards. Reported model allocation was
76.43 GiB per rank; rank load times were 161.26 seconds on `ostrich` and
130.18 seconds on `dodo`. B12x warmed 30 linear signatures and four dynamic
MoE variants, FlashInfer retained 24 decode autotune configurations, and vLLM
captured full plus piecewise CUDA graphs in six seconds (0.54/0.55 GiB).

Engine profile, KV-cache creation, and model warmup took 82.23 seconds. The
smaller per-rank KV budget was 22.29 GiB, yielding 434,414 total cache tokens
and 3.31x concurrency at the configured 131,072-token maximum. The API then
reported healthy and accepted requests without a traceback on either rank.

## End-to-end requests

The first request after startup used a 1x1 white PNG. It returned HTTP 200 for
209 prompt plus 64 completion tokens in 16.44 seconds, including one-time
inference JIT, and identified the image as blank/white. Repeating the request
on the warm path took 2.67 seconds.

A generated 128x128 fixture with a red left half and blue right half returned
HTTP 200 in 2.52 seconds for 217 prompt plus 60 completion tokens. The response
correctly stated that red was on the left and blue on the right. This provides
basic image-sensitivity evidence through preprocessing, the Vision tower,
aligner, visual routing, sparse prefill, and TP2 decode; it is not a full model
quality evaluation.

## Warm prefill baseline

`vllm bench serve` ran on `ostrich` against the live service, after one warmup,
with the random dataset, seed 20260901, 8,192 input tokens, one output token,
three requests, concurrency one, and prefix caching disabled:

```text
Successful requests                 3
Failed requests                     0
Benchmark duration                  11.24 s
Total input tokens                  24,576
Mean / median / P99 TTFT            3745.62 / 3751.01 / 3752.53 ms
Total token throughput              2187.25 tok/s
```

This is a native Vision TP2, dual-rail baseline. It is not comparable to the
one-Spark EXL3 mixed-K2/K3 Mia result: model format, worker count, and workload
differ. The future EXL3 qualification must compare mixed and uniform K2 under
identical prompts, concurrency, draft depth, and cache state, accepting no
more than 5% mixed-prefill loss.

## Remaining qualification

- Text-checkpoint startup and representative text generation.
- Image-logit/routing comparison against the checkpoint reference.
- Matched single-rail versus dual-rail throughput and stability runs.
- Longer-duration concurrency and failure-recovery testing.

## Expert-parallel follow-up

A second build retained all TP2 behavior and added the vLLM 0.28
released-source metadata adaptation for B12x EP2:

- `ostrich`: `sha256:b536981d2f9694bcb6d44fa46781737d1584ae0522d81167cda826e66e84dfdc`
- `dodo`: `sha256:9e819dcc26421e4f00936c3b5c2f3551d74ab558066c822e93eecc62f80ec395`

With dense layers TP2 and experts EP2, both ranks selected `B12xEPExperts`,
loaded all shards, warmed 30 linear signatures and 14 EP MoE launch variants,
and captured graphs in six seconds. Engine warmup took 89.64 seconds. The
smaller KV allocation was 22.71 GiB, producing 442,527 cache tokens and 3.38x
maximum-length concurrency.

The red-left/blue-right fixture returned HTTP 200 with the correct ordering;
the warm request took 3.14 seconds for 217 prompt plus 72 completion tokens.
The same standardized prefill benchmark produced:

```text
Successful requests                 3
Failed requests                     0
Benchmark duration                  12.62 s
Total input tokens                  24,576
Mean / median / P99 TTFT            4206.29 / 4229.36 / 4295.44 ms
Total token throughput              1947.72 tok/s
```

EP2 is therefore functional, but its warm prefill was about 10.95% below the
TP2 baseline and did not materially reduce rank memory. TP2 remains the native
two-Spark default; EP2 remains available with `MOE_MODE=ep` for workloads that
later demonstrate a benefit.
