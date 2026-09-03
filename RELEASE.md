# v0.1.0 — DeepSeek V4 Flash/Vision for DGX Spark

First arm64-only release for NVIDIA GB10 / SM121. One image serves the native
DeepSeek V4 Flash/Vision checkpoints across two Sparks and the mixed K2/K3
Vision EXL3 checkpoint on one Spark.

## Defaults

- No-option one-Spark profile: Vision EXL3 K2.2/D2, FP8 KV, 500K model length,
  APC enabled, fixed greedy DSpark K3.
- Native two-Spark profile: Vision, TP2+DCP1, merged dual rail, FP8 KV, 500K
  model length, APC enabled, fixed greedy DSpark K3.
- NVFP4 DS-MLA is a qualified one-Spark option; DCP2 remains experimental and
  opt-in.

## Headline results

- Native Vision TP2+DCP1: 51.38 C1 decode tok/s, 137.22 aggregate C4 tok/s,
  2,081 prompt tok/s at 8K, and 71.15 tok/s on the Orchid low-entropy arm.
- Vision EXL3 FP8: 39.66/58.28/130.24 tok/s at C1/C2/C4.
- Vision EXL3 NVFP4: a 2,039,387-token exact-image KV pool versus 1,762,308
  with FP8, a 15.72% increase.
- Tool eval (`2.3.2.dev3+g5df1e9e0c`): 114/138 native FP8, 114/138 EXL3 FP8,
  and 117/138 EXL3 NVFP4.

The complete performance tables, test method, immutable candidate provenance,
and targeted final-digest qualification are in
[benchmarks/RESULTS.md](https://github.com/tpurtell/ds4fv-vllm-28-sm12x/blob/v0.1.0/benchmarks/RESULTS.md).

## Image

```text
ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.0
```

GHCR manifest digest:
`sha256:6401b9d020361fa97ad1ac192203fdc5ae38daba3e5625fd48d568e5f9288be8`.
The same manifest is available as `sha-93df141` and `latest`.

This image is intentionally `linux/arm64` only. Do not run it on the SM120
workstation; runtime belongs on SM121 DGX Sparks.
