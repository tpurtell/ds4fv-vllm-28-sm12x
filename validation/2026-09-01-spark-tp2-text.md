# Native text TP2 runtime validation

Date: 2026-09-01 (Asia/Taipei)

Nodes: `ostrich` (TP rank 0) and `dodo` (TP rank 1), both DGX Spark GB10 / SM121.
No vLLM process or GPU code ran on the image-build workstation.

## Configuration and startup

- Image: `ds4fv-vllm-28-sm12x:recipe-check`
  - `ostrich`: `sha256:b536981d2f9694bcb6d44fa46781737d1584ae0522d81167cda826e66e84dfdc`
  - `dodo`: `sha256:9e819dcc26421e4f00936c3b5c2f3551d74ab558066c822e93eecc62f80ec395`
- Checkpoint:
  `deepseek-ai/DeepSeek-V4-Flash-0731@9e165c30e2704aec5d9d593cce3eebd58bbef1cb`
- TP2 dense layers and experts; B12x linear and MoE backends; dual RoCE;
  FP8 KV cache; 131,072 maximum model length; prefix caching enabled.

Both ranks loaded all 48 safetensor shards and selected `B12xExperts`.
Reported model allocation was 75.51 GiB per rank; rank load times were 151.29
seconds on `ostrich` and 123.32 seconds on `dodo`. B12x warmed 30 linear
signatures and four dynamic MoE variants, and CUDA graph capture completed in
six seconds using 0.72/0.67 GiB.

Engine profile, cache creation, and warmup took 60.29 seconds. The smaller KV
budget was 24.08 GiB, yielding 470,084 cache tokens and 3.59x concurrency at
the configured maximum model length.

## Generation sanity check

The prompt asked the model to compute and verify 17 times 19. It returned HTTP
200 and correctly answered 323 using the `17 * 20 - 17` check. The first
101-prompt/65-completion-token request took 8.27 seconds with inference JIT;
the warm repeat took 2.44 seconds and was identical.

## Warm prefill baseline

An initial serving run was discarded because one main request triggered a late
shape JIT and later requests reused cached prefixes. After that JIT completed,
`vllm bench serve` used a fresh random seed (20260902), five distinct 8,192
input/one output requests, concurrency one, greedy sampling, and no benchmark
warmup requests. All per-request times were between about 3.63 and 3.72
seconds, and no new JIT appeared:

```text
Successful requests                 5
Failed requests                     0
Benchmark duration                  18.33 s
Total input tokens                  40,960
Mean / median / P99 TTFT            3665.79 / 3652.11 / 3721.54 ms
Total token throughput              2234.87 tok/s
```

This is a native text TP2, dual-rail baseline. It is not a one-Spark EXL3 or
mixed-K2/K3 result and must not be used in place of the equal-work EXL3
acceptance comparison.
