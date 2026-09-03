# Native Vision B12x shared-expert stream safety

Date: 2026-09-03

This is the sustained-concurrency qualification for the B12x-specific
shared-expert stream guard. All model execution occurred on the SM121 Sparks;
the SM120 workstation was used only for orchestration and HTTP clients.

## Failure reproduced

The otherwise startup-qualified WIP22 image completed four rounds of four
simultaneous Vision clients, then lost TP1 during round five. The failing
scheduler step combined one new four-image request (631 prompt tokens, 119
scheduled tokens) with three cached decode requests at four speculative tokens
each, for 131 scheduled tokens total.

The first reported Python failure was a later `torch.empty` in shared-expert
FP8 input quantization, but CUDA explicitly reported that the illegal address
was asynchronous. TP1 subsequently aborted from the CUDA caching allocator and
Ray shut down the executor. Sixteen completed WIP22 JSON receipts therefore
precede the failure; WIP22 is not reliability-qualified.

Two attempted synchronous reproductions were invalid because the client was
given an obsolete `--role` option and a bare shell `wait` masked the background
exit codes. They are not counted as evidence here.

## Isolation and correction

vLLM's B12x adapter already contains
`_b12x_moe_plan_supports_aux_stream_overlap()`, which conservatively returns
false because resident-grid plans can use device-wide barriers. The generic
MoE runner did not consume that backend restriction and could overlap the
dense shared expert on a separate CUDA stream with B12x.

As a causal control, WIP22 was relaunched with asynchronous CUDA execution
still enabled but `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`. It passed 80/80
Vision clients over 20 four-client rounds and 20/20 prefix replay/collision
cycles, with no CUDA, tokenizer-pool, or engine errors. Its code-agent medians
were 47.59 tok/s at C1 and 123.03 tok/s at C4.

The retained patch is narrower than that global control. It adds a backend
capability argument to `SharedExperts` and disables the auxiliary stream only
when `MoERunner` selected B12x; every other MoE backend retains the upstream
overlap policy. The diagnostic environment override remains available.

## WIP23 qualification

- Image: `ds4fv-vllm-28-sm12x:b12x-stream-safe-wip23`
- emu image ID:
  `sha256:e158ca95178f6f17a993e557099de76904205e1e6dc6044b861a9dc5375849d9`
- kiwi image ID:
  `sha256:487e4eb7ee80c970eba14c043de4d3755d430df7549ca02b6b042da24a8c752e`
- Model: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`
- Runtime: native Vision, TP2+DCP2, B12x, `ag_rs`, FP8 KV, APC enabled,
  fixed greedy DSpark K3, 500,000 model length, 8,192 scheduler tokens, four
  sequences, dual rail, asynchronous CUDA, and the global stream override off.

WIP23 passed full strict startup, then:

- 80/80 valid clients across 20 four-client mixed Vision rounds. Each round
  covered 1-, 4-, and 16-image requests plus the intentional 17-image
  rejection, and collected every background process status explicitly.
- 20/20 APC replay/collision cycles. Every identical replay added exactly 512
  prefix-cache hits, while every changed-image request added zero and returned
  the changed image ordering.
- Clean head and worker logs after the stress: no `Already borrowed`, illegal
  memory access, engine death, traceback, or error record.

The five-run code-agent medians were:

| Concurrency | Pure decode | Draft acceptance | Committed/pass |
| ---: | ---: | ---: | ---: |
| C1 | 47.92 tok/s | 67.86% | 3.036 |
| C4 | 124.96 tok/s | 71.78% | 3.153 |

C1 ranged from 46.25 to 49.98 tok/s. Its derived median target-pass time was
63.35 ms, only 0.94% above the retained WIP19 DCP2 reference of 62.76 ms. The
WIP23 benchmark receipt had SHA-256
`7d65ae8603d2d6a46916e27d3063283cc21505b337866d23289fc9f311fbef0e`.

## Warm-restart failure

The required warm restart did not pass. WIP23 again completed DSpark shape
warmup, C1/C2/N2/N4, 8K and 9.5K prefills, structured output, and tool parsing,
then reported an asynchronous illegal memory access on the first one-image
request. The failing scheduler step contained one new 164-token multimodal
request and no cached requests.

This result narrows but does not close the fault: disabling the B12x
shared-expert auxiliary stream removes the original sustained mixed-load
trigger, but another asynchronous producer remains in the Vision path. WIP23
is therefore a diagnostic image, not a production candidate.

## Attention projection stream isolation

The exact warm-restart profile was repeated from the same WIP23 image with
only `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0`. This disables DeepSeek V4's
outer parallel input-GEMM fanout while leaving its later indexer/compressor
streams, asynchronous scheduling/CUDA, and the scoped B12x shared-expert guard
unchanged.

That isolation completed the full strict startup, including the previously
failing one-image request, four- and sixteen-image requests, and all four C4
passes. It implicated the outer attention overlap, but did not by itself
exclude the later streams or a stochastic shared-expert failure.

WIP24 added recursive `Tensor.record_stream()` calls for auxiliary results
returned to the caller stream. With the normal 1,024-token threshold restored,
it passed strict startup and 80/80 mixed Vision clients. It nevertheless lost
TP0 to another asynchronous illegal address during prefix replay cycle 18/20;
17 complete cycles passed first. The exact replay had a 512-token prefix hit,
and the first visible error surfaced in a later MoE gate conversion before the
CUDA allocator aborted while inserting stream events. Result lifetime tracking
alone is therefore insufficient, and WIP24 is not a release candidate.

The same WIP24 image was then relaunched with the hard
`VLLM_DISABLE_SHARED_EXPERTS_STREAM=1` control and all attention overlap left
enabled. It passed strict startup, 80/80 mixed Vision clients, and 20/20 APC
cycles. Every exact replay added 512 prefix hits and every changed-image
request added zero. Its clean-log check found no CUDA, allocator, or engine
error, and an exact warm process restart passed the complete strict startup
again.

The surviving control points back to construction-time B12x detection. The
runner's original predicate inspected a lazily initialized MoE kernel object;
the configured `--moe-backend b12x` identity is already authoritative when
`SharedExperts` is constructed. WIP25 tests that stable predicate with the
hard override off and excludes WIP24's generic result-lifetime patch.

## Minimal WIP25 result

- Image: `ds4fv-vllm-28-sm12x:b12x-configured-stream-wip25`
- emu image ID:
  `sha256:0df4fbdd18f8381b2569c776ee87fa65b8f38bcc92294932ebd5e0090456727d`
- kiwi image ID:
  `sha256:6d292df60e0d11833e2f1cb8baa43d81e2e2564610ff7ea270307ad71dc06f9e`
- Diagnostic runtime: the same TP2+DCP2 FP8/APC/greedy-K3 profile used above,
  with normal attention overlap and `VLLM_DISABLE_SHARED_EXPERTS_STREAM=0`.

WIP25 passed full strict startup, 80/80 mixed Vision clients, and 20/20 APC
replay/collision cycles. Every exact replay added 512 prefix hits and every
changed-image request added zero. Head and worker logs were clean after the
stress. The same image then passed a complete warm process restart, including
all image-count and C4 gates that had failed on WIP23.

This is the retained correction: B12x configuration disables only the unsafe
shared-expert auxiliary stream before lazy kernel initialization. The generic
WIP24 result-lifetime experiment is excluded from the production recipe.
