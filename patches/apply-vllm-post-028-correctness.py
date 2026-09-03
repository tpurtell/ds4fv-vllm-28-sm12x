#!/usr/bin/env python3
"""Backport post-v0.28 DeepSeek-V4 numerical correctness fixes.

The pinned vLLM 0.28 wheel predates three upstream fixes that directly affect
the DGX Spark profile:

* vllm-project/vllm#54815: construct main/compressor RoPE independently.
* vllm-project/vllm#54048: retain FP32 router output on family-120 CUDA.
* vllm-project/vllm#52836: stop reusing model-wide eager scratch buffers.

The wheel already contains the compiled non-``out`` FP8 Q/KV operator, so the
Python-side #52836 backport can restore caching-allocator ownership without
rebuilding vLLM's C++ extension.  The recipe's NVFP4 operator still accepts an
explicit output, which is allocated per invocation on the calling stream.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_exact(
    path: Path,
    old: str,
    new: str,
    label: str,
    *,
    expected: int = 1,
) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != expected:
        raise RuntimeError(f"{path}: {label} expected {expected} anchors, found {count}")
    path.write_text(source.replace(old, new))


def patch_rope(root: Path) -> None:
    path = root / "models/deepseek_v4/common/rope.py"
    replace_exact(
        path,
        '''    rope_parameters = config.rope_parameters
    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if rope_parameters["rope_type"] != "default":
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
''',
        '''    rope_parameters = config.rope_parameters
    # Newer checkpoints nest per-layer-type rope dicts ({"main", "compress"});
    # older ones ship a single flat dict shared by all layer types.
    if isinstance(rope_parameters.get("main"), dict) and isinstance(
        rope_parameters.get("compress"), dict
    ):
        key = "compress" if compress_ratio > 1 else "main"
        rope_parameters = dict(rope_parameters[key])
    else:
        rope_parameters = dict(rope_parameters)

    rope_parameters["rope_theta"] = (
        config.compress_rope_theta if compress_ratio > 1 else config.rope_theta
    )
    if compress_ratio > 1 and rope_parameters["rope_type"] != "default":
        # YaRN applies only to compressor (CSA/HCA) layers.
        rope_parameters["rope_type"] = (
            "deepseek_yarn"
            if rope_parameters.get("apply_yarn_scaling", True)
            else "deepseek_llama_scaling"
        )
    else:
        # Sliding-window layers use plain RoPE.  The DeepSeek class with factor
        # one is numerically identical and retains the FP32 cache expected by
        # the fused kernels.
        rope_parameters["rope_type"] = "deepseek_yarn"
        rope_parameters["factor"] = 1.0
        rope_parameters["original_max_position_embeddings"] = max_position_embeddings
''',
        "#54815 layer-specific RoPE construction",
    )


def patch_router(root: Path) -> None:
    path = root / "model_executor/layers/fused_moe/router/gate_linear.py"
    replace_exact(
        path,
        '''        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk.
        # cuBLAS on CUDA (SM90+, via allow_specialized_router_gemm); hipBLASLt on
        # ROCm, which supports the same out_dtype epilogue.
        self._router_gemm_no_bias = not bias
        self.allow_cublas_router_gemm = (
            (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
''',
        '''        # Fused bf16 x bf16 -> fp32 GEMM eligibility. torch.mm's out_dtype
        # epilogue folds the fp32 cast into the GEMM, removing the standalone
        # bf16->fp32 copy kernel that otherwise runs before grouped_topk. This is
        # the plain cuBLAS (CUDA) / hipBLASLt (ROCm) out_dtype epilogue, so it
        # applies on any CUDA-alike device. In particular, it covers family-120
        # Blackwell (GB10 / DGX Spark), which the specialized-kernel gate omits.
        self._router_gemm_no_bias = not bias
        self._router_gemm_cublas_capable = (
            current_platform.is_cuda() or current_platform.is_rocm()
        ) and self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
''',
        "#54048 CUDA router GEMM capability",
    )
    replace_exact(
        path,
        '''        if (
            not self.allow_cublas_router_gemm
            and (
                self.allow_specialized_router_gemm
                or (current_platform.is_rocm() and self._router_gemm_no_bias)
            )
            and out_dtype == torch.float32
        ):
''',
        '''        if (
            not self.allow_cublas_router_gemm
            and self._router_gemm_cublas_capable
            and out_dtype == torch.float32
        ):
''',
        "#54048 dynamic router output dtype",
    )


def patch_attention_scratch(root: Path) -> None:
    path = root / "models/deepseek_v4/attention.py"
    replace_exact(
        path,
        '    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool\n',
        "",
        "#52836 attention scratch import",
    )
    replace_exact(
        path,
        '        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,\n',
        "",
        "#52836 attention constructor arguments",
        expected=2,
    )
    replace_exact(
        path,
        "        self.eager_scratch_pool = eager_scratch_pool\n",
        "",
        "#52836 attention scratch members",
        expected=2,
    )
    replace_exact(
        path,
        "                eager_scratch_pool=eager_scratch_pool,\n",
        "",
        "#52836 nested attention scratch arguments",
        expected=2,
    )
    replace_exact(
        path,
        "            eager_scratch_pool=eager_scratch_pool,\n",
        "",
        "#52836 indexer compressor scratch argument",
    )
    replace_exact(
        path,
        '''            if self.eager_scratch_pool is not None:
                q_out = self.eager_scratch_pool.q_out(q.shape[0])
            else:
                q_out = torch.empty(
                    (q.shape[0], self.padded_heads, self.head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
''',
        '''            q_out = torch.empty(
                (q.shape[0], self.padded_heads, self.head_dim),
                dtype=q.dtype,
                device=q.device,
            )
''',
        "#52836 allocator-owned NVFP4 Q output",
    )
    replace_exact(
        path,
        '''            if self.eager_scratch_pool is not None:
                q_out = self.eager_scratch_pool.q_out(q.shape[0])
                torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(
                    q,
                    kv,
                    q_out,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.padded_heads,
                    self.eps,
                    swa_metadata.block_size,
                )
                return q_out
''',
        "",
        "#52836 allocator-owned FP8 Q output",
    )
    replace_exact(
        path,
        '''    def _global_topk_output_buffers(
        self, topk_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.compress_ratio != 4 or self.eager_scratch_pool is None:
            return None
        return self.eager_scratch_pool.global_topk_outputs(topk_indices)

''',
        "",
        "#52836 global top-k scratch accessor",
    )
    replace_exact(
        path,
        '''            outputs = None
            if self.eager_scratch_pool is not None and self.use_fp4_kv:
                outputs = self.eager_scratch_pool.indexer_q_outputs(q.shape[0])
''',
        "",
        "#52836 indexer scratch selection",
    )
    replace_exact(
        path,
        "                output_buffers=outputs,\n",
        "",
        "#52836 indexer scratch call",
    )


def patch_cache_ops(root: Path) -> None:
    cache_utils = root / "models/deepseek_v4/common/ops/cache_utils.py"
    replace_exact(
        cache_utils,
        "    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,\n",
        "",
        "#52836 global top-k output API",
    )
    replace_exact(
        cache_utils,
        '''    if output_buffers is None:
        global_topk_indices = torch.empty_like(topk_indices)
        topk_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        global_topk_indices, topk_lens = output_buffers
        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
''',
        '''    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
''',
        "#52836 allocator-owned global top-k outputs",
    )

    fused_q = root / "models/deepseek_v4/common/ops/fused_indexer_q.py"
    replace_exact(
        fused_q,
        "    output_buffers: tuple[torch.Tensor, ...] | None = None,\n",
        "",
        "#52836 fused indexer output API",
    )
    replace_exact(
        fused_q,
        '''    if output_buffers is None:
        index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    else:
        expected_num_buffers = 3 if use_fp4 else 2
        assert len(output_buffers) == expected_num_buffers
        index_weights_out = output_buffers[-1]
        assert index_weights_out.shape == index_weights.shape
''',
        '''    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)
''',
        "#52836 allocator-owned index weights",
    )
    replace_exact(
        fused_q,
        '''        packed_shape = (num_tokens, num_index_q_heads, index_q_head_dim // 2)
        scale_shape = (num_tokens, num_index_q_heads, num_scale_blocks)
        if output_buffers is None:
            index_q_packed = torch.empty(
                packed_shape,
                dtype=torch.uint8,
                device=index_q.device,
            )
            index_q_scale = torch.empty(
                scale_shape,
                dtype=torch.uint8,
                device=index_q.device,
            )
        else:
            index_q_packed, index_q_scale, _ = output_buffers
        assert index_q_packed.shape == packed_shape
        assert index_q_scale.shape == scale_shape
''',
        '''        index_q_packed = torch.empty(
            (num_tokens, num_index_q_heads, index_q_head_dim // 2),
            dtype=torch.uint8,
            device=index_q.device,
        )
        index_q_scale = torch.empty(
            (num_tokens, num_index_q_heads, num_scale_blocks),
            dtype=torch.uint8,
            device=index_q.device,
        )
''',
        "#52836 allocator-owned FP4 indexer outputs",
    )
    replace_exact(
        fused_q,
        '''    if output_buffers is None:
        index_q_fp8 = torch.empty_like(index_q, dtype=fp8_dtype)
    else:
        index_q_fp8, _ = output_buffers
        assert index_q_fp8.shape == index_q.shape
''',
        '''    index_q_fp8 = torch.empty_like(index_q, dtype=fp8_dtype)
''',
        "#52836 allocator-owned FP8 indexer output",
    )


def patch_compressor(root: Path) -> None:
    path = root / "models/deepseek_v4/compressor.py"
    replace_exact(
        path,
        "from typing import TYPE_CHECKING, Any, ClassVar, cast\n",
        "from typing import Any, ClassVar, cast\n",
        "#52836 compressor typing import",
    )
    replace_exact(
        path,
        '''if TYPE_CHECKING:
    from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool


''',
        "",
        "#52836 compressor scratch import",
    )
    replace_exact(
        path,
        '        eager_scratch_pool: "DeepseekV4EagerScratchPool | None" = None,\n',
        "",
        "#52836 compressor constructor argument",
    )
    replace_exact(
        path,
        "        self.eager_scratch_pool = eager_scratch_pool\n",
        "",
        "#52836 compressor scratch member",
    )
    replace_exact(
        path,
        '''            if not self.overlap and self.eager_scratch_pool is not None:
                extra_kwargs["compress_scratch"] = (
                    self.eager_scratch_pool.compressor_scratch(num_actual)
                )
''',
        "",
        "#52836 compressor scratch selection",
    )

    cutedsl = root / "models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py"
    replace_exact(
        cutedsl,
        "    compress_scratch: torch.Tensor | None = None,\n",
        "",
        "#52836 CuTe compressor scratch API",
    )
    replace_exact(
        cutedsl,
        '''        if compress_scratch is None:
            compressed_kv = torch.empty(
                (num_actual, head_dim),
                dtype=torch.float32,
                device=state_cache.device,
            )
        else:
            assert compress_scratch.shape == (num_actual, head_dim)
            compressed_kv = compress_scratch
''',
        '''        compressed_kv = torch.empty(
            (num_actual, head_dim),
            dtype=torch.float32,
            device=state_cache.device,
        )
''',
        "#52836 allocator-owned compressor temporary",
    )


def patch_backends(root: Path) -> None:
    flashinfer = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_exact(
        flashinfer,
        '''                        output_buffers=self._global_topk_output_buffers(
                            self.topk_indices_buffer[:num_decode_tokens]
                        ),
''',
        "",
        "#52836 FlashInfer decode top-k outputs",
    )
    replace_exact(
        flashinfer,
        "                    output_buffers=self._global_topk_output_buffers(local_topk_indices),\n",
        "",
        "#52836 FlashInfer prefill top-k outputs",
    )

    flashmla = root / "models/deepseek_v4/nvidia/flashmla.py"
    replace_exact(
        flashmla,
        '''                    output_buffers=self._global_topk_output_buffers(
                        self.topk_indices_buffer[:num_decode_tokens]
                    ),
''',
        "",
        "#52836 FlashMLA top-k outputs",
    )


def patch_model(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/model.py"
    replace_exact(
        path,
        "from vllm.models.deepseek_v4.eager_scratch import DeepseekV4EagerScratchPool\n",
        "",
        "#52836 model scratch import",
    )
    replace_exact(
        path,
        "        eager_scratch_pool: DeepseekV4EagerScratchPool | None = None,\n",
        "",
        "#52836 decoder scratch argument",
    )
    replace_exact(
        path,
        "            eager_scratch_pool=eager_scratch_pool,\n",
        "",
        "#52836 decoder attention scratch argument",
    )
    replace_exact(
        path,
        '''        padded_heads = _select_dsv4_attn_cls(vllm_config).get_padded_num_q_heads(
            config.num_attention_heads // get_tensor_model_parallel_world_size()
        )
        self.eager_scratch_pool: DeepseekV4EagerScratchPool | None = None
        if not vllm_config.parallel_config.use_ubatching:
            # TODO: support dbo if needed
            # this requires the buffer to have ubatch dim
            self.eager_scratch_pool = DeepseekV4EagerScratchPool(
                vllm_config.scheduler_config.max_num_batched_tokens,
                padded_heads,
                config.head_dim,
                config.index_n_heads,
                config.index_head_dim,
                config.index_topk,
                current_platform.device_type,
            )

''',
        "",
        "#52836 model-wide scratch allocation",
    )
    replace_exact(
        path,
        "                eager_scratch_pool=self.eager_scratch_pool,\n",
        "",
        "#52836 layer scratch argument",
    )

    scratch = root / "models/deepseek_v4/eager_scratch.py"
    if not scratch.is_file():
        raise RuntimeError(f"{scratch}: #52836 expected the pre-revert scratch module")
    scratch.unlink()


def verify(root: Path) -> None:
    dsv4 = root / "models/deepseek_v4"
    references = []
    for path in dsv4.rglob("*.py"):
        text = path.read_text()
        if "eager_scratch" in text or "_global_topk_output_buffers" in text:
            references.append(str(path))
    if references:
        raise RuntimeError(f"#52836 left scratch references: {references}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1])
    patch_rope(root)
    patch_router(root)
    patch_attention_scratch(root)
    patch_cache_ops(root)
    patch_compressor(root)
    patch_backends(root)
    patch_model(root)
    verify(root)
    print("Applied vLLM post-0.28 DeepSeek-V4 correctness backports")


if __name__ == "__main__":
    main()
