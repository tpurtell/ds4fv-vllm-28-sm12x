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
  `a1367713`. Kernel paths will be enabled individually only after model-shape
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

- **Vision sentinel validation:** The checkpoint's five image sentinel IDs
  sit immediately above the text vocabulary. Only multimodal requests for the
  exact Vision architecture receive the wider validator bound; text requests
  and every other model retain the ordinary vocabulary limit.

- **Two-rail networking:** Each Spark exposes two active RoCEv2 rails, with GID
  index 3 on `rocep1s0f0/1` and `roceP2p1s0f0/1`. The launch profile will use
  both HCAs and retain measured single-rail/dual-rail evidence before the final
  default is frozen.

- **Mixed EXL3 Trellis:** The roughly 1300-to-800 prefill regression seen with
  mixed K2/K3 layers in the Mia image is not an acceptable baseline. Current
  B12x already carries ds4rt's projection-mixed direct small-M routing; the
  vLLM binding must preserve it, and mixed prefill must remain within 5% of an
  equal-work uniform-K2 run before becoming the default.

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
- Qualify Vision logits/routing against the reference rather than relying only
  on semantic image fixtures.
- Measure TP2 and TP2+EP2 over both RoCE rails, then publish reliability and
  benchmark artifacts.
- Add the one-Spark EXL3 profile only after the calibrated v3 checkpoint is
  complete; compare mixed K2/K3 against uniform K2 with identical prompts,
  batch/concurrency, draft depth, and cache state, accepting at most 5% prefill
  loss.
