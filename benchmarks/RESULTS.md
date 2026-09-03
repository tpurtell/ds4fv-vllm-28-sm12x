# v0.1.0 performance and qualification results

These results cover the release defaults: fixed greedy DSpark K3, prefix
caching enabled, a 500,000-token maximum model length, and at most four live
sequences. All throughput numbers are medians from the complete profile suites;
decode excludes time to first token and sums per-sequence rates at C2/C4.

## Release identity and evidence policy

The published arm64 image is
`sha256:dcafc6bf649d70a014ff4350eba85cd7e721dec0ecb9a24ea38bd58401ffe8bd`,
built from recipe commit `93df1414cd5aa558d7064706e8d37c93651c59c6`. Its
GHCR manifest digest is
`sha256:6401b9d020361fa97ad1ac192203fdc5ae38daba3e5625fd48d568e5f9288be8`.

The complete native suite was measured on its immediate predecessor
(`63b7e93`, image `sha256:7758b0...`); the only subsequent source changes were
the exhaustive startup shape warmup and termination-safe xgrammar token-batch
bookkeeping. The final image then passed the targeted delta package in
[`20260903T013200Z-final-delta-93df141`](20260903T013200Z-final-delta-93df141/).

The complete one-Spark FP8/NVFP4 suites were measured on candidate `6407692`
(image `sha256:83fdbf...`). Later shared-image changes either target native
DCP2 or harden common stream/tokenizer/grammar behavior. Both one-Spark cache
profiles were therefore started on the exact release image and given light
decode/API checks instead of duplicating the complete suites. Results below
remain tied to their original immutable receipts.

Those exact-image checks measured FP8 at 40.08 C1 and 127.70 C4 tok/s (+1.1%
and -1.9% versus its complete suite) and NVFP4 at 38.55 C1 and 119.78 C4
tok/s (-1.4% and +0.4%). Both stayed healthy with zero post-ready JIT.

## Code-agent decode throughput

| Profile | C1 tok/s | C2 tok/s | C4 tok/s |
| --- | ---: | ---: | ---: |
| Native Vision, TP2+DCP1, FP8 | 51.38 | 88.67 | 137.22 |
| Vision EXL3 K2.2, FP8 | 39.66 | 58.28 | 130.24 |
| Vision EXL3 K2.2, NVFP4 | 39.11 | 58.40 | 119.28 |

Native C1 stayed at 49.77, 49.26, 50.36, and 48.32 tok/s with 8K, 32K,
64K, and 128K of existing context. FP8 EXL3 measured 37.78, 36.35, 36.55,
and 34.55 tok/s at the same depths; NVFP4 measured 36.83, 36.83, 36.41, and
36.54 tok/s.

## Unique-prompt prefill

| Prompt tokens | Native Vision FP8 | EXL3 FP8 | EXL3 NVFP4 |
| ---: | ---: | ---: | ---: |
| 8K | 2,081 | 1,291 | 1,190 |
| 16K | 2,074 | 1,305 | 1,201 |
| 32K | 2,010 | 1,314 | 1,199 |
| 64K | 1,992 | 1,298 | 1,179 |
| 128K | 1,910 | 1,256 | 1,125 |

Values are effective prompt tokens per second through time to first token.
Prompts are unique between samples, so these are not prefix-cache hits.

## Content, structured output, and tool use

| Profile | Weighted content tok/s | Orchid tok/s | Semantic contracts | Structured contracts | Tool eval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native Vision FP8 | 44.74 | 71.15 | 39/40 | 10/10 | 114/138 |
| Vision EXL3 FP8 | 33.03 | 50.77 | 40/40 | 10/10 | 114/138 |
| Vision EXL3 NVFP4 | 32.85 | 50.67 | 40/40 | 10/10 | 117/138 |

The structured category gives the normal and constrained JSON arms weight 0.5
each. The full tool matrix is `tool-eval-bench 2.3.2.dev3+g5df1e9e0c`, run
locally from `dev3`; it contains TC-01 through TC-69 and has 138 possible
points. A targeted rerun of the native suite's 18 lost-point cases recovered
TC38, TC49, and TC53, showing a plausible one-shot range of 114--117 rather
than a deterministic 114 ceiling. It is not reported as a replacement full
score.

## KV capacity and cache correctness

| Profile | Physical KV tokens | 500K request equivalents |
| --- | ---: | ---: |
| Native Vision TP2+DCP1 FP8, full-suite startup | 1,184,262 | 2.37x |
| Vision EXL3 FP8, exact release image | 1,762,308 | 3.52x |
| Vision EXL3 NVFP4, exact release image | 2,039,387 | 4.08x |

NVFP4 adds 277,079 physical tokens, a 15.72% increase over FP8. Both retain
the production 500K request limit and APC. Exact 128K text replay, 1/4/16-image
ordering, exact multimodal replay, changed-image collision isolation, and the
17-image rejection contract all passed for every applicable profile.

## Reliability and final-image delta

Every complete suite passed its six-needle 128K retrieval, exact 128K prefix
replay, deterministic tool-call contract, and 20-request post-context C4 soak.
The final image additionally passed:

- the exact TC31 medium-prompt shape that previously caused late TileLang JIT;
- 12/12 repeated constrained-JSON requests and 5/5 tool calls;
- the full 1/4/16-image ladder plus image-17 rejection and Vision APC replay;
- 40 post-ready requests with zero JIT events, xgrammar/FSM warnings,
  tracebacks, or server errors.
- exact-image one-Spark FP8 and NVFP4 startup, matched light decode within 2%
  of the retained suite medians, and zero post-ready JIT in both modes.

The detailed method and release gates are in [README.md](README.md). Raw
receipts live in the three timestamped profile directories and the final delta
directory beside this file.
