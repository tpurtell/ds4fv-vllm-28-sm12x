# Adaptation and tuning log

Each entry describes one major adaptation in one or two sentences. A path is
not called qualified until its Spark-side correctness and performance evidence
is retained in this repository.

## 2026-09-01

- **Spark-only target:** The image is hard-pinned to `linux/arm64` and GB10's
  `sm_121a`; SM120 is not used as an alias because it identifies a different
  host GPU. amd64 support is deferred until the EXL3 quant is complete.

- **Reproducible vLLM base:** The recipe uses the official vLLM 0.28.0 arm64
  image by digest. This avoids inheriting an unversioned development vLLM while
  keeping Torch 2.13, CUDA 13, and CUTLASS DSL 4.6.2 intact.

- **Current B12x:** `tpurtell/sparkinfer-glmrt` was merged with upstream B12x
  through `139e0404`, tested on a Spark without initializing CUDA, and pinned at
  `1713e2ac`. Kernel paths will be enabled individually only after model-shape
  qualification; installing B12x does not imply every projection uses it.

- **Native MXFP4 MoE port:** The newer B12x TP and replicated-input EP adapters
  were extracted from pinned vLLM lineage, adapted to the current public `b12x`
  API, and integrated into vLLM 0.28 without replacing its newer MXFP4 fixes.
  B12x takes ownership of packed weights and releases vLLM's source tensors;
  the EP workspace adapter therefore treats a source-derived zero expert count
  as that release sentinel while rejecting every nonzero ownership mismatch.
  Full-model TP2 and EP2 both start and serve correctly on the two Sparks.

- **Native Vision architecture:** The checkpoint's text-only architecture
  declaration cannot represent its vision tower, five image token types,
  visual MoE bias, or bidirectional image spans. The image adds an explicit
  multimodal architecture and fails closed if the required Vision fields are
  absent.

- **Visual MoE routing:** Image-block positions receive native Vision features
  while their raw virtual IDs above `vocab_size` are preserved for DeepSeek
  routing. Text tokens keep hash/noaux routing, while image tokens select
  experts with `bias_vl` exactly as the reference implementation.

- **Image-block attention:** Text keeps the trained 128-token SWA semantics,
  while an image query can see its complete image block in both directions.
  The physical SWA cache span is widened for the maximum 384-token block; the
  compressed DeepSeek indexer remains causal, matching the checkpoint code.
  FlashInfer owns its supported text/decode shapes; Vision's 512+512 dual-cache
  prefill is routed narrowly through B12x's BF16-QK multi-group kernel. The
  exact TP2 envelope reached 0.99984747 cosine against B12x's independent
  PyTorch oracle and passed full-model startup plus live image requests.

- **Native output projection:** DeepSeek-V4's two-stage output projection is
  packed and executed through B12x, with an explicit TP all-reduce and a
  DeepGEMM fallback for other backends. Qualified decode tilers are pinned to
  `(16, 64)` for M1 and `(16, 128)` for M2..8 to preserve the packed WO-B
  layout.

- **Compressed sparse MLA decode:** A shared-workspace adapter now exposes
  B12x's exact SWA-plus-compressed single-softmax kernel on the real Vision
  decode hot path. It was 19.0% faster in the weighted exact kernel microbench,
  but `DS4FV_USE_B12X_COMPRESSED_MLA=1` reduced the matched full content score
  from 49.027 to 48.235 tok/s (-1.62%); the adapter remains opt-in while the
  qualified default uses vLLM's split path.

- **Vision sentinel validation:** The checkpoint's five image sentinel IDs
  sit immediately above the text vocabulary. Only multimodal requests for the
  exact Vision architecture receive the wider validator bound; text requests
  and every other model retain the ordinary vocabulary limit.

- **Two-rail networking:** Each Spark exposes two active RoCEv2 rails, with GID
  index 3 on `rocep1s0f0/1` and `roceP2p1s0f0/1`. On the same five warmed text
  TP2 prompts, merged dual rail with cross-NIC 2 reached 2,234.87 tok/s versus
  2,131.70 tok/s for `rocep1s0f0` alone: a 4.84% gain and 4.62% lower mean
  TTFT. The two-HCA profile is now the qualified default.

- **Mixed EXL3 Trellis:** Large packed route blocks had forced FC2 through
  grouped M8 subtiles; the whole-tile scheduler now keeps K64/N256 and uses
  native M32/M64 FC2 with a resource-driven M8 fallback. On GB10 at the exact
  8192x4096x2048, E256/top-6 geometry, mixed K2/K3 measured 47.095 ms versus
  45.293 ms uniform K2 (+3.98%); matched one-Spark 8K prefill measured
  1,284.72 tok/s versus 1,334.90 tok/s uniform K2 (-3.76%), passing both 5%
  gates.

- **Native DSpark on EXL3:** Draft depth five widens DeepSeek V4's SWA route
  from 128 to 192 entries, so the image adds FlashInfer's legal three-chunk
  SM121 DSV4 instantiations and uses a versioned JIT module instead of the
  package's stale unadapted AOT binary. Fixed greedy K5 reached 34.59 tok/s on
  the matched 256-in/128-out C1 gate, 21.03% above the earlier probabilistic-K5
  result and 1.88% above stock adaptive K5.

- **Native DSpark launch profile:** Native Vision defaults to qualified K6,
  covering two passes through its three next-token predictor layers, while
  native text and the one-Spark EXL3 role default to K5. Greedy drafting is the
  speed-first default, with probabilistic drafting, target-only controls, and
  explicit draft-depth tuning retained as overrides; the multimodal wrapper
  carries vLLM's EAGLE3 interface so auxiliary target hidden states are
  delegated to the underlying native DeepSeek decoder.

- **Adaptive DSpark distinction:** The SM121 patch makes vLLM's stock adaptive
  verifier graph-safe for DeepSeek V4, but it remains opt-in after losing to
  fixed greedy depth on both Vision and EXL3; on EXL3 it was 4.85% lower on the
  weighted content score. These results do not qualify or reject ds4rt's more
  capable controller, which adds request-local online shared/per-position
  confidence residuals and online context/row cost learning before a global
  search over the measured jagged verification curve.

- **Content benchmark scoring:** The retained speed suite scores seven content
  categories, with normal and `response_format` structured JSON weighted 0.5
  each so constrained decoding cannot double-count the category. Orchid is a
  separate maximum-speed arm; schema v3 asks for a pure continuous stream to
  the output limit and validates purity/minimum repetitions instead of an exact
  early stop, while every visible semantic arm now has a release contract.

- **Release API contracts:** Both launch roles explicitly enable the native
  DeepSeek V4 tokenizer, reasoning parser, and automatic tool parser; Vision
  also enforces the qualified 16-image ceiling. The frozen-image suite checks
  exact tool arguments plus 1/4/16-image ordering and image-17 rejection.

- **Frozen-image benchmark harness:** The release runner records an immutable
  image ID plus the OCI-baked recipe commit across code-agent decode/depth,
  cold 8K--128K prefill, semantic content, 128K retrieval, prefix/Vision, and
  C4 soak receipts; structured normal and constrained arms retain half weight.

- **Native Vision TP2 baseline:** The full 48-shard model now starts on two
  SM121 Sparks and serves image-sensitive requests. A standardized warm
  8,192-input/1-output, concurrency-one serving run produced 2,187.25 total
  tok/s with 3.746-second mean TTFT; this is retained only as a native dual-rail
  baseline and is not an EXL3 mixed-layer comparison.

- **Native Vision EP2:** Dense TP2 plus MoE EP2 warms 14 B12x launch variants,
  serves the image-sensitivity fixture, and produces 1,947.72 tok/s with
  4.206-second mean TTFT on the same 8,192+1 workload. That is 10.95% below
  TP2 without a material rank-memory reduction, so native TP2 remains the
  default and EP2 is an opt-in mode.

- **Native text TP2:** The pinned 48-shard text checkpoint starts with B12x
  linear/MoE, serves a correct generation sanity check, and produces 2,234.87
  tok/s with 3.666-second mean TTFT over five fresh 8,192+1 requests. Its
  retained record excludes an earlier run contaminated by late JIT and prefix
  reuse.

## Qualification queue

- **Passed:** Build the pinned image and pass B12x plus Vision
  import/API/processor tests on a Spark with CUDA hidden. Evidence is retained
  in `validation/2026-09-01-spark-no-gpu.md`.
- **Passed:** Load the native Vision checkpoint across two Sparks with TP2
  experts, cross the wide dual-cache warmup, serve text and image requests,
  establish basic image sensitivity, and retain a warm prefill baseline in
  `validation/2026-09-01-spark-tp2-vision.md`.
- **Passed:** Start dense TP2 plus MoE EP2, complete EP warmup/graph capture,
  serve the image-sensitivity fixture, and benchmark the matched prefill case.
- **Passed:** Load the pinned native text checkpoint with TP2, serve a correct
  generation sanity check, and retain a clean long-prefill baseline.
- **Passed:** Compare one and two HCAs on the same warmed five-prompt workload;
  retain merged dual rail as the 4.84%-faster default. An eight-request,
  concurrency-four stress point also completed without failures.
- Qualify Vision logits/routing against the reference rather than relying only
  on semantic image fixtures.
- Extend reliability evidence beyond the retained concurrency-four stress
  point and exercise recovery from a worker or fabric interruption.
- **Passed:** Qualify one-grid mixed K2/K3 against uniform K2 at production
  geometry; the selected K64/N256 kernel is 3.98% slower and stays below the
  5% microbenchmark gate.
- **Passed:** The matched one-Spark EXL3 full-model comparison held prompts,
  concurrency, cache state, and target-only execution constant; mixed K2/K3
  was 3.76% below uniform K2 and passed the 5% prefill gate.
- **Passed:** Native DSpark K5 starts on the mixed EXL3 model and improves the
  earlier matched C1 output throughput by 50.17% over target-only decode; the
  newly qualified greedy default raises that result another 21.03% to 34.59
  tok/s. Fixed K5 also beat stock adaptive by 1.88% on random decode and 4.85%
  on the weighted content suite; evidence is retained in
  `validation/2026-09-02-exl3-greedy-adaptive.md`.
- **Passed:** Prove the shared B12x compressed-MLA adapter enters the real
  Vision hot path, then retain it as opt-in because its 19.0%-faster isolated
  kernel was 1.62% slower on the matched full-model content score. Evidence is
  retained in `validation/2026-09-02-b12x-compressed-mla.md`.
- **Passed:** Source-check and minimally exercise the frozen-image benchmark
  clients without running a premature full suite; the development-only smoke
  is retained in `validation/2026-09-02-release-harness-smoke.md`.
- Freeze one committed production-candidate digest, then run the separate
  native Vision TP2 and one-Spark EXL3 release suites against that exact image.
