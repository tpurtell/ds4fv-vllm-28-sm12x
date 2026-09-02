# Native Vision TP2+DCP2 bring-up

Date: 2026-09-02

This is development bring-up evidence for the DCP-aware image, not the final
release benchmark suite. All model execution occurred on the SM121 Sparks;
the SM120 workstation was used only for orchestration and HTTP clients.

## Frozen development images

- Image: `ds4fv-vllm-28-sm12x:dcp2-attn-wip5`
- Image ID on both emu and kiwi:
  `sha256:933ed6fa4980c41263262e6935e77147e5d1ec9a4cbd7f97a1334ae3e150ade0`
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Revision: `86f746b36186f0e567729a5c06a8c918caba82a9`
- Runtime: TP2, DCP2, B12x TP MoE, FP8 KV, 500,000 model length, APC on,
  greedy DSpark K3, 8,192-token scheduler budget, four sequences, and 0.85 GPU
  memory utilization.

The corrected DSpark slot-mapping build is
`ds4fv-vllm-28-sm12x:dcp2-attn-wip6`, image ID
`sha256:737ea2c61f1fc55aca94b58fd1c0da8de5058b3bfe3f4a2494882f17d41d79bc`
on both emu and kiwi. Its runtime profile is otherwise identical to wip5.

## Contracts and capacity

The arm64 image passed the CUDA-hidden B12x, DCP sliding-window, and DSV4 DCP
source/shape contracts before launch. Both Sparks reported the same image ID.

The native service loaded all 48 target shards and all 48 DSpark shards, warmed
the live B12x compressed-attention path across 43 layers, and captured all
target and DSpark graph buckets. vLLM reported:

```text
GPU KV cache size: 2,257,743 tokens
Maximum concurrency for 500,000 tokens per request: 4.52x
```

The strict startup gate completed DSpark block shapes 8 through 256, greedy
C1/C2/N2/N4, 8K and chunk-crossing 9.5K prefills, structured output, the tool
parser, ordered 1/4/16-image requests, and four C4 soak passes. The container
then entered `running/healthy`; a post-ready deterministic request returned
the exact answer `391` for `17 * 23`.

## Performance gate failure

The first post-ready code-agent C1 measurement failed the release gate. Three
fixed greedy K3 runs produced 16.41, 16.56, and 16.79 tok/s while accepting
only 3.3--4.3% of drafted tokens. A depth-zero check was similarly bad at
17.39 tok/s and 5.7% acceptance. The retained DCP1 native Vision K3 baseline
accepts about 68% and reaches 52.63 tok/s on the release workload.

This does not implicate probabilistic drafting: the drafter is configured for
greedy token selection. “Accepted” and “rejected” are vLLM's verifier metrics;
the target verifier is correcting the bad DCP2 proposals, which explains why
the deterministic correctness check can pass while throughput collapses. The
development image is therefore bring-up evidence only and is not qualified for
release.

Source tracing found a proposer-specific DCP omission after this result.
DSpark's custom `prepare_dflash_inputs` kernel bypassed the normal
`BlockTables.compute_slot_mappings` path and mapped both recycled context KV
and its parallel query KV as if every rank owned every global token. The wip6
development image applies the canonical virtual-block owner and rank-local
offset transform to both slot paths.

## Corrected slot-mapping result

The wip6 image passed strict startup and reported 2,039,644 aggregate KV-cache
tokens (4.08 concurrent 500K requests). Startup traffic accepted 58.1--69.0%
of draft tokens. The isolated post-ready code-agent C1 gate accepted
60.1--73.8%, median 70.3%, which is effectively the same as DCP1's
64.4--73.3%, median 71.6%. This proves the proposer slot mapping is corrected.

Median C1 pure decode was 43.38 tok/s (41.44--47.15), still 17.6% below the
52.63 tok/s DCP1 reference despite equivalent draft acceptance. DCP2 is thus
functionally correct but not yet performance-qualified; the remaining work is
isolated to its per-layer attention communication/execution overhead.

The same source was rebuilt as wip7 solely to expose vLLM's DCP communication
selector. Image ID
`sha256:7bfca5a58adda2aa21162de3d3a8fa3b8feed35c239ada8aab56a81541df5fcb`
was identical on emu and kiwi. Switching from the default query/LSE
all-gathers plus output reduce-scatter (`ag_rs`) to packed output/LSE
all-to-all (`a2a`) raised isolated C1 only from 43.38 to 43.91 tok/s (+1.2%).
The A2A medians were 73.44 tok/s at C2 and 111.93 tok/s at C4, with normal
63--68% draft acceptance. A2A therefore reduces the collective count but does
not by itself close the roughly 18% decode gap.

A matched wip7 DCP1 control removed the old baseline's model-length and APC
differences: the same 500K FP8/APC image and scheduler profile reached medians
of 53.78 tok/s at C1, 89.52 at C2, and 143.77 at C4, with 70.3--76.9% draft
acceptance. Its pool was 1,036,598 tokens (2.07 concurrent 500K requests).
Against this control, corrected `ag_rs` DCP2 is 19.3% slower at C1, while A2A
is 18.4% slower and reduces the DCP2 pool from 2,039,644 to 1,955,732 tokens.
The release default therefore remains `ag_rs`; A2A is retained only as an
explicit diagnostic option.

vLLM's DCP-group query-replication primitive was then adapted to the separate
native DeepSeek-V4 attention class and tested with `ag_rs`. The corrected wip9
image (`sha256:cb76427a78eb1e32fc73db26ead179ecc3b83d14c72981520630e45546aa3ed2`)
passed strict startup, but its medians were only 41.90/71.95/104.98 tok/s at
C1/C2/C4, with 60.9--63.6% median acceptance. It also reduced the pool to
1,892,279 tokens (3.78x500K). Replicating all 43 Q projections therefore cost
more than the small query collectives it removed and was dropped from the
recipe rather than retained as an unqualified option.

## Retained rate-aware DCP2 result

The retained WIP19 overlay makes ownership follow storage cost: the dominant
C4 main and indexer caches remain DCP-sharded, while the fixed SWA window and
128:1 compressed caches are replicated. Those replicated layers use ordinary
TP2 attention and avoid DCP query/output exchange. C4 keeps the fused packed
AG/RS path and exact cross-rank LSE merge. A proposer-specific follow-up was
required because DSpark initially passed physical rank 1 to a logically
unitary replicated cache; using logical rank 0 for effective-DCP-width-one
groups restored the draft slots on kiwi.

The exact 500K/APC/FP8/greedy-K3 WIP19 images were
`sha256:712a437d30946545146880ae77ee7f1d5f2689b8e61e024f5b082bdf0f296a14`
on emu and
`sha256:ef99505f15d860adec832b639d59b7b85aea13bd12bf92631f7e948ad8a62550`
on kiwi. They passed DSpark block shapes 8 through 256, greedy C1/C2/N2/N4,
8K and 9.5K prefills, structured output, tool parsing, ordered 1/4/16-image
requests, and four C4 stabilization passes. The service reported 1,337,408
KV-cache tokens, or 2.67 concurrent 500K requests. This is 29.0% more capacity
than the matched DCP1 control, despite replication of the cheap cache families.

The matched three-run curve produced these medians:

| Profile | C1 | C2 | C4 |
| --- | ---: | ---: | ---: |
| Pure decode, tok/s | 47.11 | 76.91 | 116.68 |
| Draft acceptance | 65.5% | 65.7% | 64.7% |
| Target-pass cost | 62.97 ms | 77.56 ms | 105.70 ms |

A separate five-run C1 repeat varied from 41.34 to 48.37 tok/s as acceptance
moved from 53.4% to 68.7%, but target-pass cost stayed within 62.71--62.95 ms
with a 62.76 ms median. The matched DCP1 value is 59.92 ms, so retained DCP2
adds 4.7% target-side C1 overhead, down from 10.2% on the prior all-sharded
path. The original roughly 18% raw decline is therefore no longer a stable
kernel regression; final reported raw throughput will come from the broader
production-image content suite, where acceptance is averaged across workloads.

## Bugs closed during bring-up

The stock hybrid-cache coordinator rejected DCP sliding-window groups and
accounted their pages as replicated. DSV4's compressed metadata also mixed
uncompressed token ownership with C4/C128 record ownership, and its SM121
attention path had no cross-rank LSE merge.

After those were corrected, three duplicated vLLM setup paths were found to
omit DCP-local sequence lengths: ordinary DSpark proposals, eager/profile
dummy attention, and DSpark's private CUDA-graph capture builder. All now use
the existing persistent buffer plus vLLM's rank-local length kernel; the
attention implementation continues to fail closed if a future path omits the
metadata.

The development performance gate is complete. Final performance, content,
long-context, and 138-point tool qualification still require a committed
production-candidate image containing the retained rate-aware path.
