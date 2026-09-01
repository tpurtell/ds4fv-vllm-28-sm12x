# Spark no-GPU build and contract validation

Date: 2026-09-01 (Asia/Taipei)

Hosts: `ostrich` and `dodo` (both `aarch64`), with `CUDA_VISIBLE_DEVICES`
empty and no GPU device passed to any test container.

## Image build

- Base digest:
  `sha256:2a7cde230b59f3ce6cab33dd245ba6bee41aa87b38c9fe84f966ff24016813ce`
- Validation tag: `ds4fv-vllm-28-sm12x:recipe-check`
- Image IDs:
  - `ostrich`: `sha256:b536981d2f9694bcb6d44fa46781737d1584ae0522d81167cda826e66e84dfdc`
  - `dodo`: `sha256:9e819dcc26421e4f00936c3b5c2f3551d74ab558066c822e93eecc62f80ec395`
- Result: all 12 Dockerfile stages completed independently on both Sparks,
  and each combined patched vLLM package passed `compileall`. The image IDs
  differ because fresh package/archive layers retain build metadata; the
  pinned runtime contracts below were therefore checked on both images.

The resulting OCI labels retained the vLLM commit, arm64/SM121 target, B12x
commit, B12x adapter lineage, and Ray version.

## Runtime package and adapter contracts

Both built images reported Ray `2.48.0` and B12x `1.3.0`. The B12x smoke test
then verified on each Spark:

- the public plan/prepare/bind/run APIs used by the adapter;
- vLLM's explicit `b12x` linear and MoE backend configuration;
- native MXFP4 backend mapping and unrounded DeepSeek-V4 dimensions;
- TP2 and replicated-input EP2 policy envelopes;
- the EP2 released-source metadata sentinel used after B12x takes ownership of
  packed expert allocations;
- the native B12x output-projection API and qualified M1/M2..8 WO-B tilers;
- the exact Vision 512+512 dual-cache dispatch and vLLM index-shape adapter;
- removal of the legacy `sparkinfer` namespace and fork-only env attribute.

Result on both images: `Spark no-GPU B12x MoE, native O-projection, and wide
dual-prefill smoke test passed`.

## Vision contracts

Against the checkpoint's retained `inference/image_processor.py`, the Vision
smoke test verified exact resize, image-block type/permutation, and patch
construction parity for square, portrait, landscape, and extreme-aspect
inputs. It also checked the multimodal registry, visual MoE routing parameters,
MM-prefix support, the 384-token image bound, and FlashInfer sparse dispatch
entries for semantic window 128 and physical window 512 over head counts
8/16/32/64/128. The validator also checked that only multimodal inputs for the
exact Vision architecture may carry the checkpoint's five sentinel IDs above
the text vocabulary.

Result on both images: `Spark no-GPU Vision smoke test passed`.

The warnings about `vllm._C` and the CUDA runtime are expected in this test:
CUDA was intentionally hidden. No full model was loaded, no kernel was
compiled or launched, and no performance claim is made by this artifact.

## Fabric inspection

Read-only host inspection on `ostrich` and `dodo` found both active HCAs on
both nodes:

| Node | HCA | Netdev | IPv4 | RoCEv2 GID index |
| --- | --- | --- | --- | --- |
| ostrich | `rocep1s0f0` | `enp1s0f0np0` | `10.55.0.1/24` | 3 |
| ostrich | `roceP2p1s0f0` | `enP2p1s0f0np0` | `10.55.0.5/24` | 3 |
| dodo | `rocep1s0f0` | `enp1s0f0np0` | `10.55.0.2/24` | 3 |
| dodo | `roceP2p1s0f0` | `enP2p1s0f0np0` | `10.55.0.6/24` | 3 |

This validates the launcher's exact dual-HCA selector and GID default. It does
not by itself qualify throughput. The runtime baseline retained separately in
`2026-09-01-spark-tp2-vision.md` used `NCCL_CROSS_NIC=2` and merged both rails;
an otherwise-identical single-rail comparison remains required.
