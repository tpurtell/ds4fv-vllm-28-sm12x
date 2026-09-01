# One-Spark EXL3 greedy and stock-adaptive qualification

Date: 2026-09-02 (Asia/Taipei)

## Scope

This development qualification compares the locked one-Spark EXL3 default with
vLLM 0.28's stock adaptive verifier. It is not a test of ds4rt's request-local
online residual controller and is not a frozen production-candidate result.

- Spark: `kiwi`, DGX Spark GB10 / SM121
- Recipe commit: `51e5ebecfd461090e35c5f2069cc57c289fcf38b`
- Development image: `sha256:930c8cfb8ebdedcbb23528780b83a1d7327578ab11bdfc1a927b1548aa2c1589`
- Model: `wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3@7827301eed170e2a5e394f45a13cc66561c601ed`
- B12x: `1713e2acb8e810888e4be2545e4a31baf0667448`
- Common settings: greedy K5 drafter, FP8 KV, prefix cache on, B12x compressed MLA off, concurrency one
- Changed setting: `DSPARK_ADAPTIVE_VERIFICATION=0` versus `1`

The image passed the CUDA-hidden EXL3 and B12x contract smokes on the Spark.
All 46 projection-mixed layers entered the public one-grid B12x Trellis path;
startup, target and dSpark graph capture, live serving, and both benchmark arms
completed without server errors.

## Matched random decode gate

Both arms used one discarded seed-20260912 warmup followed by five exact
256-input/128-output requests at seed 20260913, temperature zero, `ignore_eos`,
and concurrency one.

| Metric | Fixed greedy K5 | Stock adaptive K5 | Adaptive delta |
| --- | ---: | ---: | ---: |
| Output throughput | 34.587 tok/s | 33.937 tok/s | -1.88% |
| Mean TTFT | 499.66 ms | 511.92 ms | +2.45% |
| Accepted draft tokens | 474 / 840 | 465 / 895 | — |
| Strict acceptance | 56.43% | 51.96% | -4.47 points |
| Tokens per verification pass | 3.821 | 3.598 | -5.85% |

The fixed greedy result is also 21.03% above the earlier retained
probabilistic-K5 result of 28.577 tok/s.

## Matched content suite

Each semantic arm ran five times at temperature zero with thinking disabled.
The structured-output category is split equally between ordinary JSON and
`response_format` constrained decoding, at weight 0.5 each.

| Content arm | Fixed greedy K5 | Stock adaptive K5 | Adaptive delta |
| --- | ---: | ---: | ---: |
| Code | 53.082 | 44.868 | -15.47% |
| Fable | 23.177 | 25.865 | +11.60% |
| Hello | 32.202 | 33.547 | +4.17% |
| Math | 41.259 | 37.654 | -8.74% |
| Multilingual | 29.100 | 29.860 | +2.61% |
| Structured JSON, normal | 42.677 | 39.544 | -7.34% |
| Structured JSON, constrained | 45.814 | 37.474 | -18.20% |
| Topic | 30.866 | 31.323 | +1.48% |
| Weighted seven-category score | 36.276 | 34.518 | -4.85% |

Both structured-output modes passed 5/5 contract checks in both arms. The
separate low-entropy Orchid stream reached 65.152 tok/s fixed and 56.759 tok/s
adaptive, but neither arm obeyed the prompt's requested 100-word stop: each
continued to the 1,500-token cap. That number is therefore retained only as a
speed showcase, not an instruction-compliance pass.

Raw reports:

- [Fixed greedy K5 random decode](2026-09-02-exl3-fixed-greedy-k5-decode.json)
- [Stock-adaptive K5 random decode](2026-09-02-exl3-stock-adaptive-k5-decode.json)
- [Fixed greedy K5 content suite](2026-09-02-exl3-fixed-greedy-k5-content.json)
- [Stock-adaptive K5 content suite](2026-09-02-exl3-stock-adaptive-k5-content.json)

## Decision

Fixed greedy K5 remains the one-Spark EXL3 default. vLLM's stock adaptive
verifier is valid and graph-safe, but it lost both matched aggregate gates and
regressed the code and structured-output arms most relevant to agentic work;
this result does not reject a future port of ds4rt's more sophisticated online
controller.
