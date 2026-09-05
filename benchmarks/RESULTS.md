# v0.1.1 performance and qualification results

These measurements replace the v0.1.0 numbers because the layer-specific RoPE
and SM121 router fixes change model numerics. Every row uses FP8 KV, prefix
caching, a 500,000-token maximum model length, four scheduler slots, fixed
greedy DSpark K3, and the same immutable arm64 candidate.

## Release identity

- OCI image ID:
  `sha256:d8a8d361adc3b81b7939fc487c97baa84e520201f1a31269b2dc0f100d94c3ee`
- GHCR manifest digest:
  `sha256:4c2c85052dac8f268a7fa15ec75d86cd1001c37cb96bb685eb91b889e6550511`
- Embedded recipe commit:
  `6b940202b5ac9d38bb1af198c183f1ada513442a`
- Tool evaluator: `tool-eval-bench 2.3.2.dev3+g5df1e9e0c`, complete
  69-scenario default matrix, 138 possible points

The one-Spark profiles ran on dodo (K2.2/D2) and ostrich (pure K2). Official
Vision performance/content/Vision measurements ran on emu+kiwi with TP2+DCP1.
Emu then suffered a host-level network outage during the first cold-128K
request; the unfinished 128K replay, soak, and tool matrix were rerun from
scratch on dodo+kiwi with the identical image, checkpoint revision, launch
configuration, and evaluator. No partial result from the interrupted request
is reported.

## Code-agent decode throughput

Decode excludes time to first token and sums per-sequence rates at C2/C4.
Values are medians of five measured runs.

| Profile | C1 tok/s | C2 tok/s | C4 tok/s |
| --- | ---: | ---: | ---: |
| Vision EXL3 K2.2/D2, 1 Spark | 39.38 | 57.75 | 127.25 |
| Vision EXL3 pure K2, 1 Spark | 43.79 | 72.48 | 113.81 |
| Official Vision, TP2+DCP1 | 49.45 | 86.81 | 133.91 |

### Decode after existing context

| Existing context | K2.2/D2 | Pure K2 | Official Vision |
| ---: | ---: | ---: | ---: |
| 0 | 38.89 | 41.17 | 50.51 |
| 8K | 36.61 | 43.24 | 52.79 |
| 32K | 36.92 | 42.19 | 52.26 |
| 64K | 36.57 | 40.72 | 48.30 |
| 128K | 34.52 | 39.93 | 47.37 |

## Unique-prompt prefill

Values are effective prompt tokens per second through time to first token.
Every sample uses a unique prompt, so these are not prefix-cache hits.

| Prompt tokens | K2.2/D2 | Pure K2 | Official Vision |
| ---: | ---: | ---: | ---: |
| 8K | 1,321 | 1,336 | 2,076 |
| 16K | 1,339 | 1,327 | 2,089 |
| 32K | 1,340 | 1,330 | 2,042 |
| 64K | 1,323 | 1,306 | 2,014 |
| 128K | 1,276 | 1,261 | 1,902 |

## Content, structured output, and tools

Normal JSON and constrained `response_format` JSON have weight 0.5 each, so
structured output does not dominate the seven-arm weighted content score.

| Profile | Weighted tok/s | Orchid tok/s | Semantic | Structured | Tool eval | Pass / partial / fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| K2.2/D2 | 32.85 | 50.28 | 38/40 | 10/10 | 113/138 (82/100) | 52 / 9 / 8 |
| Pure K2 | 38.43 | 61.67 | 34/40 | 10/10 | 112/138 (81/100) | 51 / 10 / 8 |
| Official Vision | 43.81 | 71.62 | 36/40 | 10/10 | 114/138 (83/100) | 52 / 10 / 7 |

K2.2/D2 met the 38/40 release floor. Its two misses were exposition samples
that omitted the required word “paging.” Pure K2 produced an overlong fable
once and answered all five bare “hi” samples with a long Chinese self-
introduction that hit the 32-token cap. Official Vision missed three fable
length/termination checks and one exposition keyword check. All profiles
passed all ten content-suite structured samples.

The canonical pure-K2 tool result above comes from the clean pinned evaluator.
An earlier 116/138 run used an adjacent checkout after its throughput module
had user changes; it is retained as
`tool-eval-bench-dirty-throughput-only.json` for diagnosis but is excluded
from release claims.

## KV capacity

| Profile | Model load | Available KV | Physical KV tokens | 500K equivalents |
| --- | ---: | ---: | ---: | ---: |
| K2.2/D2, 1 Spark | 87.03 GiB | 12.11 GiB | 1,773,796 | 3.55x |
| Pure K2, 1 Spark | 79.96 GiB | 19.74 GiB | 2,891,548 | 5.78x |
| Official Vision, fallback TP2+DCP1 limiting rank | 81.67 GiB/rank | 17.49 GiB | 1,083,545 | 2.17x |

The official performance host pair initially reported a more conservative
1,036,135-token pool (2.07x); the fallback pair's different free-memory floor
explains the capacity change. Both comfortably admit two 500K requests. These
are physical allocator capacities while the public per-request model limit
remains 500K.

## Correctness and reliability

- K2.2/D2 passed the deterministic tool call, 1/4/16-image ladder, image-17
  rejection, exact and changed-image Vision APC checks, cold six-needle 128K
  retrieval, exact 128K replay with 127,744 cache-hit tokens, and 20/20 C4
  soak. Its startup audit found zero post-ready JIT events across 476 requests.
- Pure K2 passed the tool call, image ladder/rejection, cold 128K retrieval,
  exact 128K text replay with 127,744 hit tokens, and 20/20 soak. Its exact
  Vision replay hit 512 tokens and changed images added zero hits, proving
  cache isolation; the changed fixture was nevertheless read as `5,5,7,8`
  instead of `5,6,7,8`, so this is recorded as a model-fidelity failure.
  Its audit found zero post-ready JIT events across 691 requests.
- Official Vision passed the tool/Vision checks on emu+kiwi and the clean
  fallback cold 128K retrieval, exact 128K replay with 127,744 hit tokens, and
  20/20 C4 soak on dodo+kiwi. The exact clean tool matrix scored 114/138, and
  the fallback head audit found zero post-ready JIT events across 249 requests.

The default K2.2/D2 profile therefore passes the complete release gate. Pure K2
and official Vision are published as transparent comparison measurements; they
are not described as passing every strict model-behavior gate.

## Raw receipts

- [K2.2/D2 one-Spark FP8](20260903T033652Z-vision-exl3-k2.2-fp8/)
- [Pure-K2 one-Spark FP8](20260903T033652Z-vision-exl3-k2-fp8/)
- [Official Vision TP2+DCP1 FP8](20260903T033652Z-official-vision-fp8/)

The harness and gate definitions are documented in [README.md](README.md).
