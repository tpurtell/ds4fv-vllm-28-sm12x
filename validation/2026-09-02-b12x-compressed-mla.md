# B12x compressed sparse MLA qualification

## Scope

This is development-image evidence for the optional B12x compressed sparse MLA
decode adapter. It is not a production-candidate or release-suite result.

- Topology: two DGX Sparks (`emu` head and `kiwi` worker), native Vision TP2
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Decode profile: fixed native DSpark K6, greedy drafting, adaptive verification off
- Head image ID: `sha256:dc14cd973e3d2b2784a383b18ba2df285f405d8fe6cd7cf9b4c5da8d9ac0b4b2`
- Worker image ID: `sha256:24ab97d32c166b329b50a136d7e7dc0bbcdfd4568cdc1387faa574a0b3ef9413`
- B12x commit: `1713e2acb8e810888e4be2545e4a31baf0667448`

The B12x adapter reserved one 44,222,464-byte workspace per rank, entered the
real compressed-MLA path on both ranks, warmed it across all 43 eligible
layers, and completed target plus dSpark CUDA graph capture. The B12x policy
fix also kept K6's 192-entry SWA verification width on the supported split MLA
path rather than incorrectly promoting it to the single-pass kernel.

## Exact kernel result

The matched exact decode microbench compares B12x's fused single-softmax path
with vLLM's split implementation over the model's retained decode mix.

| Shape | B12x | vLLM split | Ratio |
| --- | ---: | ---: | ---: |
| SWA only | 12.29 us | 18.53 us | 0.6632 |
| SWA + 4 compressed candidates | 14.34 us | 20.48 us | 0.7000 |
| SWA + 128 compressed candidates | 57.41 us | 67.58 us | 0.8494 |
| Weighted | 34.27 us | 42.30 us | 0.8103 |

The isolated weighted kernel result is a 19.0% latency reduction.

## Full-model A/B

Each content arm ran five times against the same development image and launch
profile. Structured JSON is intentionally split between constrained and normal
generation at weight 0.5 each, so it contributes one category rather than two.

| Content arm | B12x on | B12x off | On delta |
| --- | ---: | ---: | ---: |
| Code | 61.989 | 61.440 | +0.89% |
| Fable | 29.663 | 29.594 | +0.23% |
| Hello | 66.292 | 65.904 | +0.59% |
| Math | 53.824 | 58.628 | -8.19% |
| Multilingual | 37.985 | 40.904 | -7.13% |
| Structured JSON, half-weight average | 47.663 | 49.374 | -3.47% |
| Topic | 40.227 | 37.344 | +7.72% |
| Weighted content score | 48.235 | 49.027 | -1.62% |
| Orchid low-entropy showcase | 118.602 | 110.267 | +7.56% |

Both structured-output modes passed their JSON contracts in all 5/5 samples.
The retained B12x-off image semantic smoke also passed with the expected answer
`1`.

Raw reports:

- [B12x enabled content report](2026-09-02-b12x-compressed-mla-on-content.json)
- [B12x disabled content report](2026-09-02-b12x-compressed-mla-off-content.json)
- [B12x disabled image smoke](2026-09-02-b12x-compressed-mla-off-vision.json)

## Decision

The fused kernel is real and materially faster in isolation, but that gain did
not survive the current fixed-K6 full-model content mix: enabled was 1.62%
slower overall despite improving code, topic, and Orchid. The adapter remains
available behind `DS4FV_USE_B12X_COMPRESSED_MLA=1`, while the qualified default
is off pending a controller/output-stability retest on the frozen production
candidate.
