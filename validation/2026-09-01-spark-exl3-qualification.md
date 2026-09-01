# One-Spark mixed EXL3 qualification

Date: 2026-09-01 (Asia/Taipei)

All CUDA execution occurred on DGX Spark GB10 / SM121 nodes. The local RTX
6000 SM120 host was used only for source editing and CPU-only checks; no vLLM
server or GPU workload ran locally.

## Qualified implementation

- EXL3 checkpoint:
  `wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3@7827301eed170e2a5e394f45a13cc66561c601ed`.
- B12x: `tpurtell/sparkinfer-glmrt@08e15b4e3467ac56fc9eab3500c11356c6068846`.
- The public one-grid mixed Trellis API receives all mixed K2/K3 routes in one
  cooperative launch. There is no per-expert compile or launch path.
- Selected large-prefill tile: `(64, 256, 64, 256)`. Native M32/M64 FC2 is
  retained when resources permit, with grouped M8 only as a resource-driven
  fallback.
- DSpark K5 uses FlashInfer's SM121 DSV4 K192 instantiation. The patched source
  has the versioned JIT name `sparse_mla_sm120_ds4fv_k192_v1`, preventing the
  unmodified packaged AOT module from taking precedence.

The final DSpark qualification image was built on `kiwi` as
`sha256:b1b6e355ffc3a959756f83a43def5a80a4ee279491dcd70744053a9246bfd04e`.
It is a development qualification image, not the frozen production-candidate
digest used for release benchmarking.

## Kernel and CPU-only checks

On `emu`, the changed B12x test files passed 128/128 GPU tests. The exact
8192x4096x2048, E256/top-6 microbenchmark produced these medians:

| Route mix | Median layer time |
|---|---:|
| Uniform K2 | 45.29331 ms |
| Balanced K2/K3 | 47.09480 ms |
| Mixed deficit | 3.98% |

The candidate also passed its CUDA-hidden EXL3 smoke, including all 46 mixed
layers, the public B12x-only path, K64 source selection, the FlashInfer H64/K192
capability, and the versioned adapted JIT module name.

## Matched full-model prefill gate

These target-only runs used `kiwi`, exact 8,192-token random inputs, one output
token, five prompts, concurrency one, temperature zero, and a discarded
fresh-seed warmup. Both model variants used the same B12x/EXL3 implementation;
the later FlashInfer-only DSpark patch does not affect this target-only gate.

| Model | Completed | Total input | Total throughput | Mean TTFT |
|---|---:|---:|---:|---:|
| Uniform K2 control | 5/5 | 40,960 | 1,334.90 tok/s | 6,137.37 ms |
| Mixed K2/K3 release quant | 5/5 | 40,960 | 1,284.72 tok/s | 6,377.03 ms |

The mixed full-model deficit is **3.76%**, passing the required `<5%` gate.
The raw result checksums on `kiwi` are
`fa314a93e6838351b005f59914aeedeffce4e5501d79cdd3f6acc90eb4d23039`
(uniform) and
`836e15d6cb1499aa367e3140595f84b37003281f8400be4dc5ed513e1039364d`
(mixed).

## Matched target-only and DSpark decode

Both retained serving runs used the mixed checkpoint on `kiwi`, exact random
256-token inputs and 128-token outputs, five prompts, concurrency one,
temperature zero, `ignore_eos`, seed 20260913, and one discarded seed-20260912
warmup. DSpark used native probabilistic sampling at fixed K5.

| Mode | Completed | Duration | Output throughput | Mean TTFT |
|---|---:|---:|---:|---:|
| Target only | 5/5 | 33.63 s | 19.03 tok/s | 420.99 ms |
| Native DSpark K5 | 5/5 | 22.40 s | 28.58 tok/s | 462.61 ms |

DSpark improves matched output throughput by **50.17%**. The retained DSpark
run drafted 1,050 tokens across 210 verification passes, accepted 433 draft
tokens (41.24%), and committed an average 3.06 tokens per target pass including
the target/bonus token. Its raw result checksum on `kiwi` is
`99da9c7f760093dcb08e76287ebe03f3108c7e13d0de08c0bf51e9dbbbb1ab3d`;
the target-only checksum is
`e0797cc31b254ab029991a00aaa07235e18b9b0ac9129696767989fb3baf2e04`.

## Release sequencing

These focused measurements are qualification gates, not release results. The
full Brandon-style matrices run only after a committed recipe is rebuilt and
its exact production-candidate digest is frozen: one complete native Vision
TP2 suite and one complete single-Spark mixed-EXL3 suite.
