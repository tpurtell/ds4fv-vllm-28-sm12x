# v0.1.1 — corrected DeepSeek V4 numerics on DGX Spark

This arm64-only release keeps the vLLM 0.28.0 base while backporting the
post-release DeepSeek correctness fixes needed by GB10 / SM121. The default is
the one-Spark Vision EXL3 K2.2/D2 model with FP8 KV, a 500K model length, APC,
and fixed greedy DSpark K3.

## Correctness changes

- [vLLM #54815](https://github.com/vllm-project/vllm/pull/54815): sparse
  SWA/main layers now use their trained plain theta-10000 RoPE while compressor
  layers retain theta-160000 YaRN. This is a numerical correction, so no
  v0.1.0 quality or performance result was inherited.
- [vLLM #54048](https://github.com/vllm-project/vllm/pull/54048): router GEMM
  output remains FP32 on family-120 CUDA, including SM121.
- [vLLM #52836](https://github.com/vllm-project/vllm/pull/52836): eager
  attention/indexer/compressor temporaries return to allocator ownership rather
  than a model-wide shared scratch pool.
- DeepSeek agent/API fixes from
  [#54838](https://github.com/vllm-project/vllm/pull/54838),
  [#48922](https://github.com/vllm-project/vllm/pull/48922), and
  [#51262](https://github.com/vllm-project/vllm/pull/51262), plus the applicable
  DCP slot/cache-group invariants from
  [#51031](https://github.com/vllm-project/vllm/pull/51031) and
  [#54277](https://github.com/vllm-project/vllm/pull/54277).
- [vLLM #52805](https://github.com/vllm-project/vllm/pull/52805): speculative
  grammar batches stop at termination instead of advancing beyond the grammar
  end state.
- InstantTensor now uses the buffered backend at I/O depth 128, avoiding the
  direct-I/O tensor-boundary failure without changing or re-uploading model
  payloads.

The post-0.28 DeepSeek PR audit was repeated through 2026-09-03. Later C128
contiguity work is already inherent in this base's full-width SM12x path;
BLHNC addressing changes target a different generic FlashInfer interface.
Remaining recent merges are Rust-renderer, DFlash, probabilistic/adaptive,
CI, other-platform, or post-0.28-regression-only changes and are not exercised
by these launch profiles.

## Rerun results

All three profiles were measured from one immutable image with FP8 KV and the
same fixed greedy K3 policy:

| Profile | C1 / C2 / C4 decode tok/s | 8K / 128K prefill tok/s | Weighted / Orchid tok/s | Semantic | Structured | Tool eval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vision EXL3 K2.2/D2, 1 Spark | 39.38 / 57.75 / 127.25 | 1,321 / 1,276 | 32.85 / 50.28 | 38/40 | 10/10 | 113/138 |
| Vision EXL3 pure K2, 1 Spark | 43.79 / 72.48 / 113.81 | 1,336 / 1,261 | 38.43 / 61.67 | 34/40 | 10/10 | 112/138 |
| Official Vision, TP2+DCP1 | 49.45 / 86.81 / 133.91 | 2,076 / 1,902 | 43.81 / 71.62 | 36/40 | 10/10 | 114/138 |

K2.2/D2 is the release-qualified default and passed its complete gate. The
pure-K2 and official-model rows are complete comparison results, not claims
that those checkpoints met every strict prompt-fidelity gate: pure K2 missed
the semantic floor and one changed-image read (with no cache collision), while
the official model missed the semantic floor. Raw receipts and exact failure
details are in
[benchmarks/RESULTS.md](https://github.com/tpurtell/ds4fv-vllm-28-sm12x/blob/v0.1.1/benchmarks/RESULTS.md).

## Image

```text
ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1
```

- OCI image ID:
  `sha256:d8a8d361adc3b81b7939fc487c97baa84e520201f1a31269b2dc0f100d94c3ee`
- GHCR manifest digest:
  `sha256:4c2c85052dac8f268a7fa15ec75d86cd1001c37cb96bb685eb91b889e6550511`
- Embedded recipe source:
  `6b940202b5ac9d38bb1af198c183f1ada513442a`

The same manifest is published as `sha-6b94020` and `latest`. This image is
intentionally `linux/arm64` only; vLLM/GPU runtime belongs on SM121 DGX Sparks,
not the SM120 workstation.

The public aliases can be checked without downloading layers:

```bash
python3 scripts/check-release-image.py \
  ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1 \
  --expected-digest sha256:4c2c85052dac8f268a7fa15ec75d86cd1001c37cb96bb685eb91b889e6550511 \
  --expected-revision 6b940202b5ac9d38bb1af198c183f1ada513442a
```

The #53046 post-reasoning speculative-token validation follow-up landed in
repository `main` after this frozen image was built and is not a v0.1.1 image
claim.
