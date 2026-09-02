# Native prefill rank-hang diagnosis and fix

## Disposition

The rank-asymmetric native Vision wedge is a FlashInfer sparse-MLA prefill
kernel bug on the Spark path, not a DSpark acceptance-policy problem. Recipe
commit `693dd2267172ff0e8fb05c78592545d275ed3991` routes every native 128/512-wide
single- or dual-cache prefill through B12x's nonpersistent unified multi-group
kernel while retaining FlashInfer's default split decode path.

The exact arm64 validation image was
`sha256:a45c31dd522f23cb2ff7706217ae50a76218c5d89d1cb0329517e2ca07a4e777`
on GB10/SM121 Sparks `emu` and `kiwi`. This is source-fix qualification, not a
published release-suite result; the documentation-bearing final candidate must
still be rebuilt and frozen by digest before both complete suites run.

## Direct root cause

The original `a180cfb` candidate could leave both GPUs near full utilization
while `/health` still returned 200 and request broadcast stopped making
progress. The decisive run used target-only execution, startup request warmup
disabled, full eager execution, and `CUDA_LAUNCH_BLOCKING=1` so that asynchronous
CUDA failures could not obscure the active call sites.

`py-spy` captured the two tensor-parallel workers in different phases:

- TP0 had returned from attention and was waiting in `ncclAllReduce` through
  `tensor_model_parallel_all_reduce` from the following B12x output projection.
- TP1 had not returned from
  `flashinfer.mla._sparse_mla_sm120._paged_attention`, reached through
  `_trtllm_batch_decode_sparse_mla_sm120` and vLLM's `_forward_prefill`.

That divergence explains the apparently healthy deadlock: TP0 waited for the
next collective, while TP1 remained inside the preceding FlashInfer kernel.
The retained raw stacks are the [TP0 trace](2026-09-02-native-prefill-hang-tp0-stack.txt)
and [TP1 trace](2026-09-02-native-prefill-hang-tp1-stack.txt).

Controls reproduced the wedge with target-only execution, `n=1`, greedy
sampling, both tested FlashInfer autotune tactics, ordinary and breakable CUDA
graphs, full eager execution, and with the persistent sparse-TopK path disabled.
The persistent-TopK fallback candidate still wedged in cycle 2; eager and
breakable-graph variants failed in cycles 4 and 3 respectively. These controls
rule out DSpark verification, stochastic sampling, graph capture, and the
indexer TopK implementation as the primary cause.

## Fix and numeric qualification

When `linear_backend=b12x` and the native SWA index width is 128 or 512,
`DeepseekV4FlashInferSM120Attention._forward_prefill` now calls the shared B12x
unified prefill dispatcher. The adapter accepts both the text/SWA single-cache
form and Vision's optional extra cache. Unsupported shapes remain fail-closed
inside B12x rather than silently falling back to the faulty path.

The exact 32-head, 512-main plus 512-extra GPU test passed against B12x's
independent PyTorch oracle:

| Metric | Result |
| --- | ---: |
| Output cosine | 0.99984747 |
| Maximum output absolute error | 0.00061035 |
| Maximum LSE absolute error | 0.00000191 |

The CUDA-hidden Vision, B12x, and mixed EXL3 integration tests also passed from
the exact image. FlashInfer's split sparse-MLA decode remains enabled by
default, and the separately qualified B12x compressed decode adapter remains
opt-in.

## Native regression

The exact former trigger—four target-only, greedy, 128-token decodes followed
by repeated numbered-image requests—completed with a 28.204 tok/s median and
30/30 full 1/4/16-image cycles (90/90 exact answers). Both ranks remained
healthy with zero restarts or OOMs. The [trigger receipt](2026-09-02-native-prefill-fix-target-only-trigger.json)
contains the four decode runs.

The production fixed-greedy K3 lifecycle then passed:

- complete gated startup with one readiness marker;
- 30/30 full 1/4/16-image cycles, or 90/90 exact answers;
- exact 128,000-token six-needle retrieval, with 66.174-second TTFT and
  approximately 1,934.29 prompt tok/s;
- another 10/10 Vision cycles immediately after the 128K request;
- a warm-cache restart, its complete startup gate, and another exact
  1/4/16-image probe;
- zero post-ready JIT in both lifecycles, with both ranks healthy and no
  restarts or OOMs.

The previous rejected candidate measured 1,921.94 prompt tok/s on its cold
128K case, so the fixed candidate is about 0.64% faster on that retained
comparison. This removes evidence of a material prefill regression, though it
does not close the separate optimization gap to older approximately 2,300
tok/s observations. Retained receipts include the [128K result](2026-09-02-native-prefill-fix-k3-long-context.json),
[first-lifecycle JIT audit](2026-09-02-native-prefill-fix-k3-first-jit-audit.json),
and [restart JIT audit](2026-09-02-native-prefill-fix-k3-restart-jit-audit.json).

## EXL3 proportional regression

The same exact image loaded all 46 projection-mixed K2/K3 layers through the
public batched B12x route and passed the complete K5 startup gate. It then
passed exact 128K six-needle retrieval and a 20-run C4 post-long-context soak:

| Metric | Current | Previous `a180cfb` | Change |
| --- | ---: | ---: | ---: |
| C4 median aggregate decode | 152.988 tok/s | 156.378 tok/s | -2.17% |
| Median accepted draft rate | 0.71150 | 0.70306 | +0.00844 |
| Median committed tokens/target pass | 4.5575 | 4.5153 | +0.0422 |

The decode change stays within the required 5% proportional gate, all 20 runs
completed, the service stayed healthy, and the audit found zero post-ready
JIT. See the [128K result](2026-09-02-native-prefill-fix-exl3-long-context.json),
[C4 soak](2026-09-02-native-prefill-fix-exl3-c4-soak.json), and
[JIT audit](2026-09-02-native-prefill-fix-exl3-jit-audit.json).

The machine-readable summary is
[`2026-09-02-native-prefill-hang-fix.json`](2026-09-02-native-prefill-hang-fix.json).
