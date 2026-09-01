# Native Vision candidate rejected for late JIT

## Identity and disposition

- Image ID: `sha256:e88845cec1839ab25cac0e3d50b6643f803eba7da20e554de8692c817e44708e`
- Image recipe: `19d30c3c7953c26698d003a1876409ee485c3bf8`
- Topology: native Vision TP2/EP2 on `emu` and `kiwi`
- Profile: fixed greedy native DSpark K6
- Result: **rejected**, not release-qualified

Every functional HTTP gate completed: 40/40 content contracts, exact tool use,
ordered 1/4/16-image reads plus image-17 rejection, exact 128K six-needle
retrieval, and the 20-request C4 post-long-context soak. The suite nevertheless
failed the stricter readiness gate because its final audit found one compile
after the ready marker:

- Kernel: `BuildPrefillChunkMetadataKernel`
- Post-ready request count: 299
- Post-ready JIT count: 1
- Service state at audit: healthy

The first 128K warmup crossed the sparse-indexer logits budget and supplied an
ordinary nonzero `query_slice_start`. vLLM's explicit warmup declared
`WarmupIntRange(0, 2)`, covering only 0 and 1, so Triton compiled the remaining
scalar specialization at runtime. The next candidate expands that range to
`WarmupIntRange(0, 3)`, which adds representative value 2 before readiness
without paying for a 128K startup request.

## Useful but non-qualifying measurements

These numbers describe the rejected image and must not be published as release
results.

| Workload | Median result |
| --- | ---: |
| Code-agent C1 pure decode | 49.79 tok/s |
| Code-agent C2 aggregate pure decode | 82.04 tok/s |
| Code-agent C4 aggregate pure decode | 128.60 tok/s |
| Cold 8K prefill | 2,082.88 prompt tok/s |
| Cold 128K prefill | 1,921.94 prompt tok/s |
| Weighted seven-category content | 46.91 tok/s |
| Code content arm | 61.98 tok/s |
| Orchid low-entropy arm | 131.44 tok/s |
| Post-128K C4 soak | 121.35 tok/s |

Raw reports, Docker inspection, and both-rank logs are retained in
[`benchmarks/20260901T194423Z-native-vision`](../benchmarks/20260901T194423Z-native-vision/manifest.json).
