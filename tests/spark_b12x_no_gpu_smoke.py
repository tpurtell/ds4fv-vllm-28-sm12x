#!/usr/bin/env python3
"""Spark-side B12x/vLLM contract smoke test with CUDA hidden."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from b12x.attention import compressed_sparse_mla
import b12x.attention._shared.mla.prefill as prefill_dispatch
import b12x.attention._shared.mla.prefill_mg as prefill_mg
from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat
from b12x.gemm import wo_projection
from b12x.gemm._shared import wo_mxfp8 as wo_projection_impl
from b12x.moe import ep_moe, fused_moe
from vllm.config.kernel import KernelConfig
from vllm.model_executor.layers.fused_moe.b12x_ep_moe import B12xEPExperts
from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    backend_to_kernel_cls,
    map_mxfp4_backend,
    mxfp4_round_up_hidden_size_and_intermediate_size,
)
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferSM120Attention,
)


def require_public_api(module: object, names: tuple[str, ...]) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"{module!r} is missing public B12x API: {missing}")


def main() -> None:
    if os.uname().machine != "aarch64":
        raise RuntimeError("this smoke test is Spark/arm64-only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be hidden for the no-GPU smoke test")

    require_public_api(
        fused_moe,
        (
            "Caps",
            "plan",
            "plan_execution",
            "plan_weights",
            "prepare_weights",
            "bind",
            "run",
        ),
    )
    require_public_api(
        ep_moe,
        ("Caps", "plan", "prepare_expert_map", "bind", "run"),
    )
    require_public_api(
        compressed_sparse_mla,
        ("Caps", "plan", "bind", "run", "split_chunks_for_contract"),
    )
    require_public_api(wo_projection, ("pack_weights", "run_inv_rope"))
    assert wo_projection_impl._wo_b_fused_mma_tiler(1) == (16, 64)
    assert wo_projection_impl._wo_b_fused_mma_tiler(2) == (16, 128)
    assert wo_projection_impl._wo_b_fused_mma_tiler(8) == (16, 128)
    assert wo_projection_impl._wo_b_fused_mma_tiler(9) is None
    fused_wo_b_source = inspect.getsource(
        wo_projection_impl.wo_b_dense_gemm_fused_quant_mxfp8
    )
    assert "mma_tiler_mn=_wo_b_fused_mma_tiler(expected_m)" in fused_wo_b_source

    config = KernelConfig(moe_backend="b12x", linear_backend="b12x")
    assert config.moe_backend == "b12x"
    assert config.linear_backend == "b12x"
    assert map_mxfp4_backend("b12x") == [Mxfp4MoeBackend.B12X]
    assert backend_to_kernel_cls(Mxfp4MoeBackend.B12X) == [
        B12xEPExperts,
        B12xExperts,
    ]
    assert mxfp4_round_up_hidden_size_and_intermediate_size(
        Mxfp4MoeBackend.B12X, 14400, 1536
    ) == (14400, 1536)

    tp = SimpleNamespace(
        use_ep=False,
        ep_size=1,
        use_all2all_kernels=False,
        enable_eplb=False,
    )
    assert B12xExperts._supports_parallel_config(tp)

    ep = SimpleNamespace(
        use_ep=True,
        ep_size=2,
        tp_size=1,
        dp_size=1,
        pcp_size=1,
        sp_size=1,
        use_all2all_kernels=False,
        enable_eplb=False,
    )
    assert B12xEPExperts._supports_parallel_config(ep)

    b12x_source = inspect.getsource(B12xExperts)
    module_source = Path(inspect.getfile(B12xExperts)).read_text()
    assert "plan_b12x_fp4_moe_weights" in b12x_source
    assert 'VLLM_B12X_MOE_FORCE_MODELOPT_PREP")' in module_source
    assert "envs.VLLM_B12X_MOE_FORCE_MODELOPT_PREP" not in module_source
    assert "sparkinfer." not in module_source

    ep_workspace_source = inspect.getsource(B12xEPExperts.workspace_shapes)
    assert "source_local_num_experts not in (0, prepared.num_experts)" in (
        ep_workspace_source
    )
    assert "released-parameter sentinel" in ep_workspace_source

    o_proj_source = inspect.getsource(DeepseekV4FlashInferSM120Attention._o_proj)
    b12x_o_proj_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._b12x_o_proj
    )
    post_load_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention.process_b12x_o_proj_weights_after_loading
    )
    assert "self._b12x_o_proj_enabled" in o_proj_source
    assert "deep_gemm_fp8_o_proj" in o_proj_source
    assert "wo_projection.run_inv_rope" in b12x_o_proj_source
    assert "tensor_model_parallel_all_reduce" in b12x_o_proj_source
    assert "wo_projection.pack_weights" in post_load_source

    decode_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._forward_decode
    )
    b12x_decode_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._b12x_compressed_mla_decode
    )
    reserve_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._reserve_empty_forward_workspace
    )
    assert "self._b12x_compressed_mla_enabled" in decode_source
    assert "compressed_sparse_mla.bind" in b12x_decode_source
    assert "scratch_views.bind" in b12x_decode_source
    assert "compressed_sparse_mla.run" in b12x_decode_source
    assert "out=output" in b12x_decode_source
    assert "self._get_b12x_compressed_mla_workspace" in reserve_source

    # Preserve vLLM's aggregate packed-cache page stride while exposing the
    # per-layer FP8 payload as B12x's rank-2 byte view. This is a CPU-only view
    # contract check; no attention kernel is launched.
    packed_stride = 1_039_680
    cache_backing = torch.empty(4 * packed_stride, dtype=torch.uint8)
    vllm_cache = torch.as_strided(
        cache_backing,
        size=(4, 64, 1, 584),
        stride=(packed_stride, 584, 584, 1),
    )
    b12x_cache = DeepseekV4FlashInferSM120Attention._as_b12x_sparse_cache(
        vllm_cache
    )
    assert b12x_cache.shape == (4, 64 * 584)
    assert b12x_cache.stride() == (packed_stride, 1)

    # The public B12x dispatcher must route the exact Vision prefill envelope
    # (TP2 local heads=32, primary SWA width=512, compressed width=512) to the
    # BF16-QK MG dual kernel. Replace only the CUDA launcher so this remains a
    # no-GPU contract test.
    calls: list[dict[str, object]] = []
    real_mg = prefill_mg.run_unified_prefill_mg

    def fake_mg(**kwargs):
        calls.append(kwargs)

    prefill_mg.run_unified_prefill_mg = fake_mg
    try:
        q = torch.empty((2, 32, 512), dtype=torch.bfloat16)
        main_cache = torch.empty((4, 37440), dtype=torch.uint8)
        main_indices = torch.zeros((2, 512), dtype=torch.int32)
        extra_cache = torch.empty((4, 37440), dtype=torch.uint8)
        extra_indices = torch.zeros((2, 512), dtype=torch.int32)
        output, lse = prefill_dispatch.run_unified_prefill(
            q=q,
            kv_cache=main_cache,
            topk_indices=main_indices,
            sm_scale=512**-0.5,
            page_block_size=64,
            stride_kv_block=37440,
            extra_kv_cache=extra_cache,
            extra_indices=extra_indices,
            extra_page_block_size=64,
            stride_extra_kv_block=37440,
        )
    finally:
        prefill_mg.run_unified_prefill_mg = real_mg
    assert output.shape == (2, 32, 512)
    assert lse.shape == (2, 32)
    assert len(calls) == 1
    assert calls[0]["compute_mode"] == ComputeMode.BF16
    assert calls[0]["model_type"] == ModelType.DSV4
    assert calls[0]["scale_format"] == ScaleFormat.UE8M0_BYTE
    assert calls[0]["extra_kv_cache"] is extra_cache
    assert calls[0]["extra_page_block_size"] == 64

    # vLLM carries one shared-KV-head dimension in its sparse index tensors;
    # the adapter must remove that singleton before entering B12x's 2-D API.
    adapted: dict[str, object] = {}
    real_prefill = prefill_dispatch.run_unified_prefill

    def fake_prefill(**kwargs):
        adapted.update(kwargs)

    prefill_dispatch.run_unified_prefill = fake_prefill
    try:
        fake_attention = SimpleNamespace(
            scale=512**-0.5,
            attn_sink=torch.empty((32,), dtype=torch.float32),
            _get_workspace=lambda _device: torch.empty((4096,), dtype=torch.uint8),
        )
        DeepseekV4FlashInferSM120Attention._b12x_wide_dual_prefill(
            fake_attention,
            q=torch.empty((2, 32, 512), dtype=torch.bfloat16),
            swa_kv_cache=torch.empty((4, 64, 1, 584), dtype=torch.uint8),
            swa_indices=torch.zeros((2, 1, 512), dtype=torch.int32),
            swa_lengths=torch.full((2,), 128, dtype=torch.int32),
            extra_kv_cache=torch.empty((4, 64, 1, 584), dtype=torch.uint8),
            extra_indices=torch.zeros((2, 1, 512), dtype=torch.int32),
            extra_lengths=torch.full((2,), 512, dtype=torch.int32),
            output=torch.empty((2, 32, 512), dtype=torch.bfloat16),
        )
    finally:
        prefill_dispatch.run_unified_prefill = real_prefill
    assert adapted["topk_indices"].shape == (2, 512)
    assert adapted["extra_indices"].shape == (2, 512)

    wide_prefill_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._b12x_wide_dual_prefill
    )
    forward_prefill_source = inspect.getsource(
        DeepseekV4FlashInferSM120Attention._forward_prefill
    )
    assert "run_unified_prefill" in wide_prefill_source
    assert "extra_page_block_size" in wide_prefill_source
    assert "int(swa_indices_chunk.shape[-1]) == 512" in forward_prefill_source
    assert "use_b12x_wide_dual" in forward_prefill_source

    print(
        "Spark no-GPU B12x MoE, native O-projection, compressed decode, and "
        "wide dual-prefill smoke test passed"
    )


if __name__ == "__main__":
    main()
