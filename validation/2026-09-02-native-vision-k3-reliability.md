# Native Vision K3 reliability isolation

## Frozen test identity

- Recipe commit: `d6b1afc0f631dde8b952b46b53d22bb0ba55b8c5`
- Image ID: `sha256:46471e9e07e3657ef4603f840e96d93ec99d7b7fd21a8fd8667145979e3b79f0`
- Topology: native Vision, TP2 on emu/kiwi, merged dual RoCE rails
- Common target settings: B12x MoE and linear backends, FP8 KV cache,
  8,192-token scheduler budget, four scheduler slots, prefix cache off

This was a diagnostic isolation on the frozen image, not a release suite. The
service was recreated between profiles and both Sparks were confirmed free of
GPU and Ray/vLLM processes before each launch.

## Reproduction and isolation

Fixed greedy K6 first crashed on a warm-cache restart with an asynchronous
illegal-memory-access report surfacing in the target indexer's BF16 projection.
A clean fresh-cache K6 start then passed startup, but a post-ready repeat
wedged on the 16-distinct-image arm while both GPUs stayed at roughly 96%
utilization and `/health` continued to return 200.

The same frozen K6 image was relaunched with `CUDA_LAUNCH_BLOCKING=1` and the
automatic request warmup disabled. Three standalone numbered-image requests
and ten complete 1/4/16-image cycles passed; cycle 11 passed its 1- and 4-image
requests and then timed out on the 16-image request. Both ranks again stayed
GPU-busy indefinitely. Synchronous launches therefore did not remove the
failure.

Holding the image, target model, B12x paths, Vision batching, TP2 topology, and
request sequence constant produced these controls:

| Profile | Complete 1/4/16 cycles | Requests passed | Result |
| --- | ---: | ---: | --- |
| Target-only | 30 | 90/90 | pass |
| Fixed greedy K3 | 30 | 90/90 | pass |
| Fixed greedy K6, synchronous diagnostic | 10 plus cycle 11's 1/4 arms | 35/36 | 16-image wedge |

This isolates the defect to speculative execution at K6 and, more narrowly,
to behavior introduced by its second pass through the checkpoint's three
predictor layers. It does not prove that rejection rollback is the corrupting
operation; K6 remains a correctness bug pending a kernel-level trace.

The retained head-rank evidence is the
[`K6 hang log`](2026-09-02-native-vision-k6-hang-head.log) and its
[`container inspection`](2026-09-02-native-vision-k6-hang-head-inspect.json),
the [`target-only stress log`](2026-09-02-native-vision-target-only-stress-head.log)
and its [`container inspection`](2026-09-02-native-vision-target-only-stress-head-inspect.json),
and the [`K3 stress log`](2026-09-02-native-vision-k3-stress-head.log) and its
[`container inspection`](2026-09-02-native-vision-k3-stress-head-inspect.json).

## Matched decode result

The same code-agent workload, seed, 256-token output, two warmups, and five
measured runs used for the K6 baseline gave:

| Concurrency | K3 median tok/s | K6 median tok/s | K3 change |
| ---: | ---: | ---: | ---: |
| C1 | 52.6301 | 48.0369 | +9.56% |
| C2 | 89.8646 | 80.4788 | +11.66% |
| C4 | 135.7304 | 126.8496 | +7.00% |

K3's median accepted-draft rates were 0.7160, 0.7033, and 0.7212 at C1/C2/C4,
with 3.1481, 3.1098, and 3.1636 median committed tokens per target pass. The
K6 C1 baseline committed only about 3.35 tokens per target pass despite
generating and verifying twice the draft depth, explaining why the second pass
did not pay for itself even before considering the reliability failure.

The retained structured decode receipt is
[`2026-09-02-native-vision-k3-decode.json`](2026-09-02-native-vision-k3-decode.json).
Native Vision therefore defaults to fixed greedy K3 for the next production
candidate. K6 remains an explicit, unqualified override.
