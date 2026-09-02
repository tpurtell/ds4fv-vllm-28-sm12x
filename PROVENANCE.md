# Provenance

This recipe is intentionally reproducible from immutable inputs. Floating tags
may be used as human-friendly aliases after qualification, but they are not
used as build inputs.

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
- B12x: `tpurtell/sparkinfer-glmrt@1713e2acb8e810888e4be2545e4a31baf0667448`
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

The experimental Vision checkpoint declares `DeepseekV4ForCausalLM` despite
shipping `vision.*`, `aligner.*`, image sentinel embeddings, and visual-routing
biases. The recipe therefore forces the in-tree
`DeepseekV4VisionForConditionalGeneration` architecture and does not enable
`--trust-remote-code`.

The Vision EXL3 derivative retains that incomplete base architecture metadata,
so its first-class launch profile applies the same architecture override plus
the pinned native Vision dimensions. Its 43 target layers use projection-mixed
K2/K3 Trellis routes; the three DSpark draft layers remain uniform K2.
The profile also preserves the checkpoint's `hash_moe`/`moe` layer taxonomy by
registering those DeepSeek-V4-specific values with Transformers' strict layer
type validator.

The later upstream revision `31ea11185e11ccafad1c385104188a9e3b648ad6`
changes only README/evaluation metadata relative to the pinned cached payload.
Git object IDs for `config.json`, `model.safetensors.index.json`, and
`inference/image_processor.py` are identical at both revisions.
