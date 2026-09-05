# Provenance

This recipe is intentionally reproducible from immutable inputs. Floating tags
may be used as human-friendly aliases after qualification, but they are not
used as build inputs.

## v0.1.1 release image

- OCI image ID:
  `sha256:d8a8d361adc3b81b7939fc487c97baa84e520201f1a31269b2dc0f100d94c3ee`
- GHCR manifest digest:
  `sha256:4c2c85052dac8f268a7fa15ec75d86cd1001c37cb96bb685eb91b889e6550511`
- Recipe source embedded in the image:
  `6b940202b5ac9d38bb1af198c183f1ada513442a`
- Platform: `linux/arm64`; CUDA target: `sm_121a`
- Registry aliases: `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.1`,
  `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:sha-6b94020`, and
  `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:latest`

The exact image ID was loaded on `emu`, `kiwi`, `ostrich`, and `dodo`. Three
complete measurement suites cover official Vision TP2+DCP1 FP8, pure-K2
Vision EXL3 FP8, and K2.2/D2 Vision EXL3 FP8. K2.2/D2 passed the complete
release gate; the other two are retained as transparent checkpoint-comparison
results with their strict model-behavior misses intact.

The pinned 0.28.0 source snapshot predates or omits upstream vLLM fixes #52805,
#52836, #54048, #54815, #54838, #48922, #51262, #51031, and #54277. Narrow
fail-closed source transforms apply their relevant structured-spec-decode,
eager-temporary ownership, SM121 router dtype, layer-specific DeepSeek-V4 RoPE,
agent/tool parsing, and DCP cache/slot invariants. The OCI backport label records
the post-v0.1.0 additions; #52805 was already present in the candidate lineage.
InstantTensor uses `INSTANTTENSOR_BACKEND=BUFFERED` and
`INSTANTTENSOR_IO_DEPTH=128`, avoiding the direct-I/O tensor-boundary failure
without modifying or re-uploading either 80+ GiB EXL3 checkpoint.

The #53046 post-reasoning speculative-token validation patch was committed to
repository `main` after the frozen v0.1.1 image revision. It is intentionally
excluded from the release-image claims above.

## v0.1.0 release image

- OCI image ID:
  `sha256:dcafc6bf649d70a014ff4350eba85cd7e721dec0ecb9a24ea38bd58401ffe8bd`
- GHCR manifest digest:
  `sha256:6401b9d020361fa97ad1ac192203fdc5ae38daba3e5625fd48d568e5f9288be8`
- Recipe source embedded in the image:
  `93df1414cd5aa558d7064706e8d37c93651c59c6`
- Platform: `linux/arm64`; CUDA target: `sm_121a`
- Retained registry aliases: `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:v0.1.0`
  and `ghcr.io/tpurtell/ds4fv-vllm-28-sm12x:sha-93df141`. The floating
  `latest` alias now follows v0.1.1.

The same full image ID was verified on `emu`, `kiwi`, `ostrich`, and `dodo`.
Performance provenance and the final-digest delta policy are recorded in
[benchmarks/RESULTS.md](benchmarks/RESULTS.md).

## Runtime

- Base image: `vllm/vllm-openai@sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce`
- Architecture: `linux/arm64` only
- vLLM: `0.28.0`, source commit `2cf0a6915ce544dc493a0990f2ea38d81601128a`
- PyTorch: `2.13.0+cu130`
- CUDA user-space: 13.x from the pinned base
- FlashInfer in the base: `0.6.16.post3`
- CUTLASS DSL: `4.6.2`
- Ray: `2.48.0` with its default runtime extras (the official arm64 base does
  not include Ray)
- B12x: `tpurtell/sparkinfer-glmrt@3fc8d1491d1313c0ca64b2b95772972b7f42ee9d`
- B12x vLLM MoE adapter lineage:
  `local-inference-lab/vllm@30038602b71395f481ef4a6edfe4fcf8551d9c15`

The B12x revision is the fork merge containing upstream B12x through
`139e04048bc3bb4f7210c99e7184d8d2f0e345e7`. Its non-GPU planning and policy
tests passed in the official arm64 vLLM container on a Spark (142 passed, 3
skipped) before it was pinned here.

Only the two native FP4 MoE adapter files are inherited from the older vLLM
fork. Their source SHA256 digests are checked before installation, their
package imports are translated to the current `b12x` namespace, and narrowly
anchored integration edits are applied to vLLM 0.28 rather than replacing its
newer MXFP4 oracle or runner.

One additional EP compatibility delta is applied after verifying those source
digests: B12x releases vLLM's source expert tensors after packing, while vLLM
0.28 later derives a workspace expert count from the released tensor's shape.
The adapter recognizes zero only as that sentinel and continues to validate
the prepared allocation for every nonzero ownership count.

The later EXL3 adapter will retain the mixed-Trellis direct-route lineage from
ds4rt (`359e8055`) and the corresponding current B12x implementation rather
than reproducing Mia's projection-by-projection mixed-layer fallback.

## Models

- Native text: `deepseek-ai/DeepSeek-V4-Flash-0731` at
  `9e165c30e2704aec5d9d593cce3eebd58bbef1cb`.
- Native vision: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at
  `86f746b36186f0e567729a5c06a8c918caba82a9`.
- Text EXL3: `wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3`
  at `7827301eed170e2a5e394f45a13cc66561c601ed`.
- Vision EXL3: `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1`
  at `8aab722f04f7e8963af83de5acb16138474e0228`.
- Vision EXL3 pure K2:
  `wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1` at
  `419697c409cb4157471bcaf68be07dbd151b0a40`.

The experimental Vision checkpoint declares `DeepseekV4ForCausalLM` despite
shipping `vision.*`, `aligner.*`, image sentinel embeddings, and visual-routing
biases. The recipe therefore forces the in-tree
`DeepseekV4VisionForConditionalGeneration` architecture and does not enable
`--trust-remote-code`.

The Vision EXL3 derivatives retain that incomplete base architecture metadata,
so their first-class launch profiles apply the same architecture override plus
the pinned native Vision dimensions. K2.2/D2's 43 target layers use
projection-mixed K2/K3 Trellis routes; the pure-K2 profile provides the matched
uniform quantization comparison. The three DSpark draft layers remain uniform
K2 in both profiles.
The profile also preserves the checkpoint's `hash_moe`/`moe` layer taxonomy by
registering those DeepSeek-V4-specific values with Transformers' strict layer
type validator. The pinned base combines vLLM 0.28 with a newer Transformers
validator split, so the compatibility helper updates the legacy combined tuple
and the separate attention/MLP tuples.
The Vision wrapper also handles the derivative's dual text-weight namespaces:
ordinary tensors retain native `layers.*` names, while its quantized expert
payloads are already stored under `model.layers.*`; both delegate exactly once
to the wrapped DeepSeek-V4 language model.

The later upstream revision `31ea11185e11ccafad1c385104188a9e3b648ad6`
changes only README/evaluation metadata relative to the pinned cached payload.
Git object IDs for `config.json`, `model.safetensors.index.json`, and
`inference/image_processor.py` are identical at both revisions.
