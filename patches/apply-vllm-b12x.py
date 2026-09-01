#!/usr/bin/env python3
"""Install and integrate the pinned B12x MXFP4 MoE adapters into vLLM 0.28.

The two adapter modules come from an immutable vLLM fork revision and are
verified byte-for-byte before their namespace is updated from ``sparkinfer``
to the current ``b12x`` package.  All changes to the official vLLM source are
exactly anchored; version drift therefore fails the image build closed.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


ADAPTER_COMMIT = "30038602b71395f481ef4a6edfe4fcf8551d9c15"
ADAPTER_FILES = {
    "model_executor/layers/fused_moe/b12x_moe.py": (
        "06318bea3fa342f231496a16997992a92c737c4ed657b13bdc11f1c79b02bc6d"
    ),
    "model_executor/layers/fused_moe/b12x_ep_moe.py": (
        "97619dfabe6e1a34017a328cf353a1a54988dce50c4b6040e4e6eb32d25599ae"
    ),
}
MARKER = "B12X native MXFP4 MoE backend (Spark SM12x)"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label} expected exactly one source anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


def read_adapter(relative: str, source_tree: Path | None, source_base: str) -> bytes:
    if source_tree is not None:
        return (source_tree / "vllm" / relative).read_bytes()
    url = f"{source_base.rstrip('/')}/vllm/{relative}"
    with urllib.request.urlopen(url) as response:
        return response.read()


def install_adapters(root: Path, source_tree: Path | None, source_base: str) -> None:
    for relative, expected_sha256 in ADAPTER_FILES.items():
        payload = read_adapter(relative, source_tree, source_base)
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"adapter {relative} SHA256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )

        source = payload.decode()
        source = source.replace("sparkinfer.", "b12x.")
        if relative.endswith("/b12x_moe.py"):
            old = "envs.VLLM_B12X_MOE_FORCE_MODELOPT_PREP"
            if source.count(old) != 1:
                raise RuntimeError(
                    f"adapter {relative}: expected one legacy env probe, "
                    f"found {source.count(old)}"
                )
            source = source.replace(
                old,
                '_env_flag("VLLM_B12X_MOE_FORCE_MODELOPT_PREP")',
                1,
            )
            source = source.replace("import vllm.envs as envs\n", "", 1)
        if "sparkinfer." in source:
            raise RuntimeError(f"adapter {relative}: legacy namespace remains")

        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)


def patch_b12x_wo_projection(root: Path) -> None:
    path = root / "gemm/_shared/wo_mxfp8.py"
    replace_once(
        path,
        "def _wo_a_fused_mma_tiler(expected_m: int | None, rank: int) -> tuple[int, int] | None:\n"
        "    if expected_m is not None and 1 <= expected_m <= 8 and rank <= 1536:\n"
        "        return (16, 64)\n"
        "    return None\n\n\n"
        "def wo_a_dense_gemm_fused_quant_mxfp8(\n",
        "def _wo_a_fused_mma_tiler(expected_m: int | None, rank: int) -> tuple[int, int] | None:\n"
        "    if expected_m is not None and 1 <= expected_m <= 8 and rank <= 1536:\n"
        "        return (16, 64)\n"
        "    return None\n\n\n"
        "def _wo_b_fused_mma_tiler(expected_m: int | None) -> tuple[int, int] | None:\n"
        "    # The production tile-major WO-B pack is compatible only with the\n"
        "    # qualified 16-row decode plans. Pin those plans here so the generic\n"
        "    # low-SM 32x64 override cannot select a layout-incompatible kernel.\n"
        "    if expected_m == 1:\n"
        "        return (16, 64)\n"
        "    if expected_m is not None and 2 <= expected_m <= 8:\n"
        "        return (16, 128)\n"
        "    return None\n\n\n"
        "def wo_a_dense_gemm_fused_quant_mxfp8(\n",
        "B12x WO-B fused decode tile policy",
    )
    replace_once(
        path,
        "        rhs_values_tiled=(\n"
        "            wo_b_hgr.values_tiled\n"
        "            if expected_m is not None and 1 <= expected_m <= 8\n"
        "            else None\n"
        "        ),\n"
        "        a_inner_span=inner_span,\n",
        "        rhs_values_tiled=(\n"
        "            wo_b_hgr.values_tiled\n"
        "            if expected_m is not None and 1 <= expected_m <= 8\n"
        "            else None\n"
        "        ),\n"
        "        a_inner_span=inner_span,\n"
        "        mma_tiler_mn=_wo_b_fused_mma_tiler(expected_m),\n",
        "B12x WO-B fused decode tile selection",
    )


def patch_b12x_wide_dual_prefill(root: Path) -> None:
    path = root / "attention/_shared/mla/prefill.py"
    replace_once(
        path,
        "  * DSV4 dual-cache (extra/indexed tokens): topk==128, heads % 8 == 0,\n"
        "    pbs_extra in {2, 64} (BF16-QK), using the same head partitioning.\n",
        "  * DSV4 dual-cache (extra/indexed tokens): topk in {128, 512},\n"
        "    heads % 8 == 0, pbs_extra in {2, 64} (BF16-QK), using the same\n"
        "    head partitioning. The 512-wide primary section is required by\n"
        "    DeepSeek-V4 Vision's mixed-modal SWA prefix.\n",
        "B12x wide dual-cache prefill documentation",
    )
    replace_once(
        path,
        "        if model_type == ModelType.DSV4 and int(topk) == 128:\n"
        "            return _run_partitioned_mg(\n",
        "        # The MG dual kernel derives num_main_tiles from the runtime\n"
        "        # primary width. Qualify the 512-wide DSV4 Vision contract in\n"
        "        # addition to FlashInfer's original 128-wide text contract.\n"
        "        if model_type == ModelType.DSV4 and int(topk) in (128, 512):\n"
        "            return _run_partitioned_mg(\n",
        "B12x wide DSV4 dual-cache dispatch",
    )
    replace_once(
        path,
        '            "DSV4 topk==128 with heads divisible by 8 is supported. "\n',
        '            "DSV4 topk in {128, 512} with heads divisible by 8 is supported. "\n',
        "B12x wide dual-cache dispatch error",
    )
    replace_once(
        path,
        '        "DSV4 dual-cache topk==128 with heads%8==0 and pbs_extra in {2, 64}; "\n',
        '        "DSV4 dual-cache topk in {128, 512} with heads%8==0 and "\n'
        '        "pbs_extra in {2, 64}; "\n',
        "B12x wide dual-cache supported-shape summary",
    )


def patch_kernel_config(root: Path) -> None:
    path = root / "config/kernel.py"
    replace_once(
        path,
        '    "deep_gemm_mega_moe",\n    "cutlass",\n',
        '    "deep_gemm_mega_moe",\n    "b12x",\n    "cutlass",\n',
        "B12x MoE backend literal",
    )
    replace_once(
        path,
        '    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels\n'
        '    - "cutlass": Use vLLM CUTLASS kernels\n',
        '    - "deep_gemm_mega_moe": Use DeepGEMM mega MoE kernels\n'
        '    - "b12x": Use native B12X FP4 MoE kernels on SM12x\n'
        '    - "cutlass": Use vLLM CUTLASS kernels\n',
        "B12x MoE backend documentation",
    )


def patch_mxfp4_oracle(root: Path) -> None:
    path = root / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    replace_once(
        path,
        'class Mxfp4MoeBackend(Enum):\n    NONE = "None"\n',
        'class Mxfp4MoeBackend(Enum):\n'
        '    NONE = "None"\n'
        f'    # {MARKER}\n'
        '    B12X = "B12X"\n',
        "B12x MXFP4 backend enum",
    )
    replace_once(
        path,
        "def backend_to_kernel_cls(\n"
        "    backend: Mxfp4MoeBackend,\n"
        ") -> list[type[mk.FusedMoEExperts]]:\n"
        "    if backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "def backend_to_kernel_cls(\n"
        "    backend: Mxfp4MoeBackend,\n"
        ") -> list[type[mk.FusedMoEExperts]]:\n"
        "    if backend == Mxfp4MoeBackend.B12X:\n"
        "        from vllm.model_executor.layers.fused_moe.b12x_ep_moe import (\n"
        "            B12xEPExperts,\n"
        "        )\n"
        "        from vllm.model_executor.layers.fused_moe.b12x_moe import B12xExperts\n"
        "\n"
        "        return [B12xEPExperts, B12xExperts]\n"
        "\n"
        "    elif backend == Mxfp4MoeBackend.DEEPGEMM_MXFP4:\n",
        "B12x expert implementations",
    )
    replace_once(
        path,
        '    mapping: dict[str, list[Mxfp4MoeBackend]] = {\n'
        '        "deep_gemm": [Mxfp4MoeBackend.DEEPGEMM_MXFP4],\n',
        '    mapping: dict[str, list[Mxfp4MoeBackend]] = {\n'
        '        "b12x": [Mxfp4MoeBackend.B12X],\n'
        '        "deep_gemm": [Mxfp4MoeBackend.DEEPGEMM_MXFP4],\n',
        "B12x backend mapping",
    )
    replace_once(
        path,
        "    \"\"\"Round up hidden_size and intermediate_size based on backend requirements.\"\"\"\n"
        "    if backend == Mxfp4MoeBackend.EMULATION:\n",
        "    \"\"\"Round up hidden_size and intermediate_size based on backend requirements.\"\"\"\n"
        "    if backend == Mxfp4MoeBackend.B12X:\n"
        "        return hidden_size, intermediate_size\n"
        "    if backend == Mxfp4MoeBackend.EMULATION:\n",
        "B12x native dimensions",
    )
    replace_once(
        path,
        '    \"\"\"Convert loaded weights into backend-specific kernel format.\n\n'
        '    Supports DeepGEMM, FlashInfer, TRTLLM MXFP8, Triton and Marlin backends.\n'
        '    \"\"\"\n'
        '    is_gfx1250 = False\n',
        '    \"\"\"Convert loaded weights into backend-specific kernel format.\n\n'
        '    Supports DeepGEMM, FlashInfer, TRTLLM MXFP8, Triton and Marlin backends.\n'
        '    \"\"\"\n'
        '    if mxfp4_backend == Mxfp4MoeBackend.B12X:\n'
        '        return (\n'
        '            w13_weight.data,\n'
        '            w2_weight.data,\n'
        '            w13_weight_scale.data,\n'
        '            w2_weight_scale.data,\n'
        '            w13_bias,\n'
        '            w2_bias,\n'
        '        )\n\n'
        '    is_gfx1250 = False\n',
        "B12x source-format preservation",
    )
    replace_once(
        path,
        "            gemm1_clamp_limit=swiglu_limit,\n"
        "        )\n"
        "    elif mxfp4_backend in (\n"
        "        Mxfp4MoeBackend.MARLIN,\n",
        "            gemm1_clamp_limit=swiglu_limit,\n"
        "        )\n"
        "    elif mxfp4_backend in (\n"
        "        Mxfp4MoeBackend.B12X,\n"
        "        Mxfp4MoeBackend.MARLIN,\n",
        "B12x W4A16 quantization config",
    )


def patch_mxfp4_method(root: Path) -> None:
    path = root / "model_executor/layers/quantization/mxfp4.py"
    replace_once(
        path,
        "            self.moe_kernel = make_mxfp4_moe_kernel(\n"
        "                moe_quant_config=self.moe_quant_config,\n"
        "                moe_config=self.moe,\n"
        "                mxfp4_backend=self.mxfp4_backend,\n"
        "                experts_cls=self.experts_cls,\n"
        "                routing_tables=layer._expert_routing_tables(),\n"
        "            )\n\n"
        "    def process_weights_after_loading(self, layer):\n",
        "            self.moe_kernel = make_mxfp4_moe_kernel(\n"
        "                moe_quant_config=self.moe_quant_config,\n"
        "                moe_config=self.moe,\n"
        "                mxfp4_backend=self.mxfp4_backend,\n"
        "                experts_cls=self.experts_cls,\n"
        "                routing_tables=layer._expert_routing_tables(),\n"
        "            )\n"
        "            if self.mxfp4_backend == Mxfp4MoeBackend.B12X:\n"
        "                self.moe_kernel.fused_experts.process_weights_after_loading(\n"
        "                    layer\n"
        "                )\n\n"
        "    def process_weights_after_loading(self, layer):\n",
        "B12x eager weight preparation",
    )


def patch_moe_runner(root: Path) -> None:
    path = root / "model_executor/layers/fused_moe/runner/moe_runner.py"
    replace_once(
        path,
        "direct_register_custom_op(\n"
        '    op_name="moe_forward_shared",\n'
        "    op_func=_moe_forward_shared,\n"
        "    fake_impl=_moe_forward_shared_fake,\n"
        "    tags=(torch.Tag.needs_fixed_stride_order,),\n"
        ")\n\n\n"
        "def _unpack(\n",
        "direct_register_custom_op(\n"
        '    op_name="moe_forward_shared",\n'
        "    op_func=_moe_forward_shared,\n"
        "    fake_impl=_moe_forward_shared_fake,\n"
        "    tags=(torch.Tag.needs_fixed_stride_order,),\n"
        ")\n\n\n"
        "# B12x owns prepared expert allocations and can alias the input/output\n"
        "# buffers. Separate opaque ops preserve that mutation contract through\n"
        "# torch.compile without changing other MoE backends.\n"
        "direct_register_custom_op(\n"
        '    op_name="b12x_moe_forward",\n'
        "    op_func=_moe_forward,\n"
        '    mutates_args=["hidden_states"],\n'
        "    fake_impl=_moe_forward_fake,\n"
        "    tags=(torch.Tag.needs_fixed_stride_order,),\n"
        ")\n\n\n"
        "direct_register_custom_op(\n"
        '    op_name="b12x_moe_forward_shared",\n'
        "    op_func=_moe_forward_shared,\n"
        '    mutates_args=["hidden_states"],\n'
        "    fake_impl=_moe_forward_shared_fake,\n"
        "    tags=(torch.Tag.needs_fixed_stride_order,),\n"
        ")\n\n\n"
        "def _unpack(\n",
        "B12x opaque forward operations",
    )
    replace_once(
        path,
        "        return (\n"
        "            torch.ops.vllm.moe_forward\n"
        "            if self._shared_experts is None\n"
        "            else torch.ops.vllm.moe_forward_shared\n"
        "        )\n\n"
        "    @property\n"
        "    def shared_experts(self) -> SharedExperts | None:\n",
        "        if self._uses_b12x_moe_kernel:\n"
        "            return (\n"
        "                torch.ops.vllm.b12x_moe_forward\n"
        "                if self._shared_experts is None\n"
        "                else torch.ops.vllm.b12x_moe_forward_shared\n"
        "            )\n\n"
        "        return (\n"
        "            torch.ops.vllm.moe_forward\n"
        "            if self._shared_experts is None\n"
        "            else torch.ops.vllm.moe_forward_shared\n"
        "        )\n\n"
        "    @property\n"
        "    def shared_experts(self) -> SharedExperts | None:\n",
        "B12x opaque forward selection",
    )
    replace_once(
        path,
        "    @property\n"
        "    def _quant_method(self) -> FusedMoEMethodBase:\n"
        "        return self.routed_experts.quant_method\n\n"
        "    def apply_routed_input_transform(\n",
        "    @property\n"
        "    def _quant_method(self) -> FusedMoEMethodBase:\n"
        "        return self.routed_experts.quant_method\n\n"
        "    @property\n"
        "    def _uses_b12x_moe_kernel(self) -> bool:\n"
        "        moe_kernel = getattr(self._quant_method, \"moe_kernel\", None)\n"
        "        fused_experts = getattr(moe_kernel, \"fused_experts\", None)\n"
        "        if fused_experts is None:\n"
        "            return False\n\n"
        "        from vllm.model_executor.layers.fused_moe.b12x_moe import (\n"
        "            B12xExperts,\n"
        "        )\n\n"
        "        return isinstance(fused_experts, B12xExperts)\n\n"
        "    def apply_routed_input_transform(\n",
        "B12x kernel detection",
    )


def patch_b12x_o_projection(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        "    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:\n"
        "        return deep_gemm_fp8_o_proj(\n"
        "            o,\n"
        "            positions,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            self.wo_a,\n"
        "            self.wo_b,\n"
        "            n_groups=self.n_local_groups,\n"
        "            heads_per_group=self.n_local_heads // self.n_local_groups,\n"
        "            nope_dim=self.nope_head_dim,\n"
        "            rope_dim=self.rope_head_dim,\n"
        "            o_lora_rank=self.o_lora_rank,\n"
        "            einsum_recipe=self._einsum_recipe,\n"
        "            tma_aligned_scales=self._tma_aligned_scales,\n"
        "        )\n\n"
        "    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:\n"
        "        super().__init__(vllm_config, *args, **kwargs)\n",
        "    @staticmethod\n"
        "    def _b12x_weight_scale(linear: torch.nn.Module) -> torch.Tensor:\n"
        "        scale = getattr(linear, \"weight_scale_inv\", None)\n"
        "        if scale is None:\n"
        "            scale = linear.weight_scale\n"
        "        return scale\n\n"
        "    def process_b12x_o_proj_weights_after_loading(self) -> None:\n"
        "        if not self._b12x_o_proj_enabled:\n"
        "            return\n"
        "        if not (\n"
        "            getattr(self.wo_a, \"b12x_block_fp8_linear\", False)\n"
        "            and getattr(self.wo_b, \"b12x_block_fp8_linear\", False)\n"
        "        ):\n"
        "            raise RuntimeError(\n"
        "                \"B12x DeepSeek V4 O projection requires B12x-processed \"\n"
        "                \"WO-A and WO-B weights\"\n"
        "            )\n"
        "        from b12x.gemm import wo_projection\n\n"
        "        group_width = (\n"
        "            (self.n_local_heads // self.n_local_groups)\n"
        "            * (self.nope_head_dim + self.rope_head_dim)\n"
        "        )\n"
        "        self._b12x_o_proj_weights = wo_projection.pack_weights(\n"
        "            self.wo_a.weight,\n"
        "            self._b12x_weight_scale(self.wo_a),\n"
        "            self.wo_b.weight,\n"
        "            self._b12x_weight_scale(self.wo_b),\n"
        "            groups=self.n_local_groups,\n"
        "            group_width=group_width,\n"
        "            rank=self.o_lora_rank,\n"
        "            hidden=self.hidden_size,\n"
        "        )\n\n"
        "    def _b12x_o_proj(\n"
        "        self, o: torch.Tensor, positions: torch.Tensor\n"
        "    ) -> torch.Tensor:\n"
        "        from b12x.gemm import wo_projection\n"
        "        from vllm.distributed import (\n"
        "            get_tensor_model_parallel_world_size,\n"
        "            tensor_model_parallel_all_reduce,\n"
        "        )\n\n"
        "        weights = self._b12x_o_proj_weights\n"
        "        if weights is None:\n"
        "            raise RuntimeError(\n"
        "                \"B12x DeepSeek V4 O-projection weights were not packed \"\n"
        "                \"after loading\"\n"
        "            )\n"
        "        output = wo_projection.run_inv_rope(\n"
        "            o,\n"
        "            positions,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            weights,\n"
        "            heads_per_group=self.n_local_heads // self.n_local_groups,\n"
        "            nope_dim=self.nope_head_dim,\n"
        "            rope_dim=self.rope_head_dim,\n"
        "            expected_m=int(o.shape[0]),\n"
        "        )\n"
        "        if get_tensor_model_parallel_world_size() > 1:\n"
        "            output = tensor_model_parallel_all_reduce(output)\n"
        "        return output\n\n"
        "    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:\n"
        "        if self._b12x_o_proj_enabled:\n"
        "            return self._b12x_o_proj(o, positions)\n"
        "        return deep_gemm_fp8_o_proj(\n"
        "            o,\n"
        "            positions,\n"
        "            self.rotary_emb.cos_sin_cache,\n"
        "            self.wo_a,\n"
        "            self.wo_b,\n"
        "            n_groups=self.n_local_groups,\n"
        "            heads_per_group=self.n_local_heads // self.n_local_groups,\n"
        "            nope_dim=self.nope_head_dim,\n"
        "            rope_dim=self.rope_head_dim,\n"
        "            o_lora_rank=self.o_lora_rank,\n"
        "            einsum_recipe=self._einsum_recipe,\n"
        "            tma_aligned_scales=self._tma_aligned_scales,\n"
        "        )\n\n"
        "    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:\n"
        "        super().__init__(vllm_config, *args, **kwargs)\n"
        "        self._b12x_o_proj_enabled = (\n"
        "            vllm_config.kernel_config.linear_backend == \"b12x\"\n"
        "        )\n"
        "        self._b12x_o_proj_weights = None\n",
        "B12x native DeepSeek V4 output projection",
    )

    replace_once(
        path,
        "    def _forward_prefill(\n"
        "        self,\n"
        "        q: torch.Tensor,\n"
        "        compressed_k_cache: torch.Tensor | None,\n",
        "    def _b12x_wide_dual_prefill(\n"
        "        self,\n"
        "        *,\n"
        "        q: torch.Tensor,\n"
        "        swa_kv_cache: torch.Tensor,\n"
        "        swa_indices: torch.Tensor,\n"
        "        swa_lengths: torch.Tensor,\n"
        "        extra_kv_cache: torch.Tensor,\n"
        "        extra_indices: torch.Tensor,\n"
        "        extra_lengths: torch.Tensor | None,\n"
        "        output: torch.Tensor,\n"
        "    ) -> None:\n"
        "        # FlashInfer's SM12x dual-cache prefill dispatcher fixes the\n"
        "        # primary SWA width at 128. Vision stores a 512-wide physical\n"
        "        # SWA index row so image-prefix tokens can remain visible. B12x's\n"
        "        # qualified MG kernel accepts that wider primary section while\n"
        "        # preserving one softmax across SWA + compressed candidates.\n"
        "        from b12x.attention._shared.mla.prefill import run_unified_prefill\n"
        "\n"
        "        lse_numel = int(q.shape[0]) * int(q.shape[1])\n"
        "        workspace = self._get_workspace(q.device)\n"
        "        lse_out = workspace[: lse_numel * 4].view(torch.float32).view(\n"
        "            int(q.shape[0]), int(q.shape[1])\n"
        "        )\n"
        "        run_unified_prefill(\n"
        "            q=q,\n"
        "            kv_cache=swa_kv_cache,\n"
        "            topk_indices=swa_indices.reshape(int(q.shape[0]), -1),\n"
        "            topk_length=swa_lengths,\n"
        "            sm_scale=self.scale,\n"
        "            page_block_size=int(swa_kv_cache.shape[1]),\n"
        "            attn_sink=self.attn_sink,\n"
        "            output=output,\n"
        "            lse_out=lse_out,\n"
        "            extra_kv_cache=extra_kv_cache,\n"
        "            extra_indices=extra_indices.reshape(int(q.shape[0]), -1),\n"
        "            extra_topk_length=extra_lengths,\n"
        "            extra_page_block_size=int(extra_kv_cache.shape[1]),\n"
        "        )\n"
        "\n"
        "    def _forward_prefill(\n"
        "        self,\n"
        "        q: torch.Tensor,\n"
        "        compressed_k_cache: torch.Tensor | None,\n",
        "B12x wide dual-cache prefill helper",
    )
    replace_once(
        path,
        "            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:\n"
        "                raise RuntimeError(\n"
        '                    "Compressed sparse MLA prefill requires compressed sparse indices."\n'
        "                )\n"
        "            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
        "                query=q_chunk,\n"
        "                swa_kv_cache=swa_kv_paged,\n"
        "                workspace_buffer=self._get_workspace(q.device),\n"
        "                sparse_indices=swa_indices_chunk,\n"
        "                compressed_kv_cache=extra_kv_paged,\n"
        "                out=output[query_start:query_end],\n"
        "                bmm1_scale=self.scale,\n"
        "                sinks=self.attn_sink,\n"
        '                kv_layout="NHD",\n'
        "                swa_topk_lens=swa_lens_chunk,\n"
        "                extra_sparse_indices=extra_sparse_indices_chunk,\n"
        "                extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
        "            )\n",
        "            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:\n"
        "                raise RuntimeError(\n"
        '                    "Compressed sparse MLA prefill requires compressed sparse indices."\n'
        "                )\n"
        "            output_chunk = output[query_start:query_end]\n"
        "            use_b12x_wide_dual = (\n"
        "                self._b12x_o_proj_enabled\n"
        "                and extra_kv_paged is not None\n"
        "                and extra_sparse_indices_chunk is not None\n"
        "                and int(swa_indices_chunk.shape[-1]) == 512\n"
        "            )\n"
        "            if use_b12x_wide_dual:\n"
        "                self._b12x_wide_dual_prefill(\n"
        "                    q=q_chunk,\n"
        "                    swa_kv_cache=swa_kv_paged,\n"
        "                    swa_indices=swa_indices_chunk,\n"
        "                    swa_lengths=swa_lens_chunk,\n"
        "                    extra_kv_cache=extra_kv_paged,\n"
        "                    extra_indices=extra_sparse_indices_chunk,\n"
        "                    extra_lengths=extra_sparse_lengths_chunk,\n"
        "                    output=output_chunk,\n"
        "                )\n"
        "            else:\n"
        "                flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
        "                    query=q_chunk,\n"
        "                    swa_kv_cache=swa_kv_paged,\n"
        "                    workspace_buffer=self._get_workspace(q.device),\n"
        "                    sparse_indices=swa_indices_chunk,\n"
        "                    compressed_kv_cache=extra_kv_paged,\n"
        "                    out=output_chunk,\n"
        "                    bmm1_scale=self.scale,\n"
        "                    sinks=self.attn_sink,\n"
        '                    kv_layout="NHD",\n'
        "                    swa_topk_lens=swa_lens_chunk,\n"
        "                    extra_sparse_indices=extra_sparse_indices_chunk,\n"
        "                    extra_sparse_topk_lens=extra_sparse_lengths_chunk,\n"
        "                )\n",
        "B12x Vision wide dual-cache prefill selection",
    )

    loader_path = root / "model_executor/model_loader/utils.py"
    replace_once(
        loader_path,
        "    # Initialize post-load attention weights for any attention layer and MM\n"
        "    # encoder. NOTE: Happens after other modules so we can easily decompress\n"
        "    # weights.\n",
        "    # Pack B12x's fused DeepSeek V4 output projection only after both linear\n"
        "    # quant methods have finalized their checkpoint weights. Doing this here\n"
        "    # also keeps the one-time conversion outside torch.compile and CUDA graphs.\n"
        "    for _, module in model.named_modules():\n"
        "        b12x_o_proj_post_load = getattr(\n"
        "            module, \"process_b12x_o_proj_weights_after_loading\", None\n"
        "        )\n"
        "        if b12x_o_proj_post_load is not None:\n"
        "            with device_loading_context(module, target_device):\n"
        "                b12x_o_proj_post_load()\n"
        "            release_device_memory_under_pressure(target_device)\n\n"
        "    # Initialize post-load attention weights for any attention layer and MM\n"
        "    # encoder. NOTE: Happens after other modules so we can easily decompress\n"
        "    # weights.\n",
        "B12x output-projection post-load packing",
    )


def patch_warmup(root: Path) -> None:
    path = root / "model_executor/warmup/b12x_warmup.py"
    replace_once(
        path,
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\n"
        "from vllm.model_executor.layers.fused_moe.b12x_moe import (\n"
        "    warmup_b12x_moe_dynamic,\n"
        ")\n",
        "B12x MoE warmup import",
    )
    replace_once(
        path,
        "    for name, warmup in providers:\n"
        "        warmed = warmup(model, **warmup_kwargs)\n"
        "        if warmed:\n"
        "            logger.info_once(\n"
        '                "Warmed up %d B12X %s linear GEMM signatures.",\n'
        "                warmed,\n"
        "                name,\n"
        "            )\n",
        "    for name, warmup in providers:\n"
        "        warmed = warmup(model, **warmup_kwargs)\n"
        "        if warmed:\n"
        "            logger.info_once(\n"
        '                "Warmed up %d B12X %s linear GEMM signatures.",\n'
        "                warmed,\n"
        "                name,\n"
        "            )\n\n"
        "    warmed_moe = warmup_b12x_moe_dynamic(\n"
        "        model,\n"
        "        max_tokens=max_tokens,\n"
        "        token_counts=cudagraph_capture_sizes,\n"
        "    )\n"
        "    if warmed_moe:\n"
        "        logger.info_once(\n"
        '            "Warmed up %d B12X dynamic MoE signatures.", warmed_moe\n'
        "        )\n",
        "B12x MoE warmup invocation",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    parser.add_argument("--b12x-root", required=True, type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-tree", type=Path)
    source.add_argument(
        "--source-base",
        default=(
            "https://raw.githubusercontent.com/local-inference-lab/vllm/"
            f"{ADAPTER_COMMIT}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.vllm_root.resolve()
    b12x_root = args.b12x_root.resolve()
    oracle = root / "model_executor/layers/fused_moe/oracle/mxfp4.py"
    if not oracle.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")
    if MARKER in oracle.read_text():
        raise RuntimeError(f"B12x patch already applied to {root}")
    if not (b12x_root / "gemm/_shared/wo_mxfp8.py").is_file():
        raise RuntimeError(f"not a B12x package root: {b12x_root}")

    source_base = getattr(args, "source_base", None)
    if source_base is None:
        source_base = ""
    patch_b12x_wo_projection(b12x_root)
    patch_b12x_wide_dual_prefill(b12x_root)
    install_adapters(root, args.source_tree, source_base)
    patch_kernel_config(root)
    patch_mxfp4_oracle(root)
    patch_mxfp4_method(root)
    patch_moe_runner(root)
    patch_b12x_o_projection(root)
    patch_warmup(root)


if __name__ == "__main__":
    main()
