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
        if relative.endswith("/b12x_ep_moe.py"):
            old = '''        if prepared.num_experts != int(local_num_experts):
            raise ValueError(
                "B12X EP local expert metadata does not match prepared weights: "
                f"metadata={int(local_num_experts)}, "
                f"prepared={prepared.num_experts}"
            )
'''
            new = '''        # The parent B12x owner releases the source tensors after packing.
        # vLLM 0.28 subsequently derives this workspace argument from
        # ``w1.shape[0]``, so zero is its released-parameter sentinel rather
        # than an EP ownership count. The prepared allocation is authoritative;
        # a nonzero disagreement still fails closed.
        source_local_num_experts = int(local_num_experts)
        if source_local_num_experts not in (0, prepared.num_experts):
            raise ValueError(
                "B12X EP local expert metadata does not match prepared weights: "
                f"metadata={source_local_num_experts}, "
                f"prepared={prepared.num_experts}"
            )
'''
            if source.count(old) != 1:
                raise RuntimeError(
                    f"adapter {relative}: expected one released-source EP "
                    f"metadata guard, found {source.count(old)}"
                )
            source = source.replace(old, new, 1)
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
        "    def _b12x_prefill(\n"
        "        self,\n"
        "        *,\n"
        "        q: torch.Tensor,\n"
        "        swa_kv_cache: torch.Tensor,\n"
        "        swa_indices: torch.Tensor,\n"
        "        swa_lengths: torch.Tensor,\n"
        "        extra_kv_cache: torch.Tensor | None,\n"
        "        extra_indices: torch.Tensor | None,\n"
        "        extra_lengths: torch.Tensor | None,\n"
        "        output: torch.Tensor,\n"
        "    ) -> None:\n"
        "        # Use B12x's nonpersistent SM121 MG kernel for every native\n"
        "        # DSV4 prefill shape. FlashInfer's sparse paged-attention kernel\n"
        "        # can hang on only one TP rank after sustained decode, leaving\n"
        "        # its peer blocked in the following output-projection all-reduce.\n"
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
        "            extra_indices=(\n"
        "                extra_indices.reshape(int(q.shape[0]), -1)\n"
        "                if extra_indices is not None\n"
        "                else None\n"
        "            ),\n"
        "            extra_topk_length=extra_lengths,\n"
        "            extra_page_block_size=(\n"
        "                int(extra_kv_cache.shape[1])\n"
        "                if extra_kv_cache is not None\n"
        "                else None\n"
        "            ),\n"
        "        )\n"
        "\n"
        "    def _forward_prefill(\n"
        "        self,\n"
        "        q: torch.Tensor,\n"
        "        compressed_k_cache: torch.Tensor | None,\n",
        "B12x native prefill helper",
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
        "            use_b12x_prefill = (\n"
        "                self._b12x_o_proj_enabled\n"
        "                and int(swa_indices_chunk.shape[-1]) in (128, 512)\n"
        "            )\n"
        "            if use_b12x_prefill:\n"
        "                self._b12x_prefill(\n"
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
        "B12x native prefill selection",
    )

    loader_path = root / "model_executor/model_loader/utils.py"
    replace_once(
        loader_path,
        "    # Initialize post-load attention weights for any attention layer and MM\n"
        "    # encoder. NOTE: Happens after other modules so we can easily decompress\n"
        "    # weights.\n",
        "    # Pack B12x's fused DeepSeek V4 output projection only after both linear\n"
        "    # quant methods have finalized their checkpoint weights. Reserve the one\n"
        "    # shared compressed-MLA workspace in the same post-load phase so memory\n"
        "    # profiling accounts for it and graph capture never allocates it late.\n"
        "    for _, module in model.named_modules():\n"
        "        b12x_o_proj_post_load = getattr(\n"
        "            module, \"process_b12x_o_proj_weights_after_loading\", None\n"
        "        )\n"
        "        if b12x_o_proj_post_load is not None:\n"
        "            with device_loading_context(module, target_device):\n"
        "                b12x_o_proj_post_load()\n"
        "            release_device_memory_under_pressure(target_device)\n\n"
        "        b12x_compressed_mla_reserve = getattr(\n"
        "            module, \"_get_b12x_compressed_mla_workspace\", None\n"
        "        )\n"
        "        if (\n"
        "            b12x_compressed_mla_reserve is not None\n"
        "            and getattr(module, \"_b12x_compressed_mla_enabled\", False)\n"
        "        ):\n"
        "            with device_loading_context(module, target_device):\n"
        "                b12x_compressed_mla_reserve(target_device)\n\n"
        "    # Initialize post-load attention weights for any attention layer and MM\n"
        "    # encoder. NOTE: Happens after other modules so we can easily decompress\n"
        "    # weights.\n",
        "B12x post-load projection packing and decode workspace reservation",
    )


def patch_b12x_compressed_mla_decode(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        "from typing import TYPE_CHECKING, ClassVar, cast\n\n"
        "import torch\n",
        "import os\n"
        "from typing import TYPE_CHECKING, ClassVar, cast\n\n"
        "import torch\n",
        "B12x compressed MLA environment import",
    )
    replace_once(
        path,
        "from vllm.forward_context import get_forward_context\n",
        "from vllm.forward_context import get_forward_context\n"
        "from vllm.logger import init_logger\n",
        "B12x compressed MLA logger import",
    )
    replace_once(
        path,
        "if TYPE_CHECKING:\n"
        "    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata\n\n"
        "_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024\n"
        "_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}\n",
        "if TYPE_CHECKING:\n"
        "    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata\n\n"
        "logger = init_logger(__name__)\n\n"
        "_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024\n"
        "_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}\n"
        "# One capture-stable B12x allocation is shared by every sequential DSV4\n"
        "# attention layer. The key includes the complete capacity contract so a\n"
        "# second model in the same process cannot accidentally reuse undersized\n"
        "# storage. Binding views are cached after their first materialization to\n"
        "# avoid replaying scratch-control fills in every layer.\n"
        "_b12x_compressed_mla_workspace_by_key: dict[\n"
        "    tuple[torch.device, tuple[int, ...]], tuple[object, torch.Tensor]\n"
        "] = {}\n"
        "_b12x_compressed_mla_scratch_by_key: dict[\n"
        "    tuple[torch.device, tuple[int, ...]], object\n"
        "] = {}\n",
        "B12x shared compressed MLA workspace state",
    )
    replace_once(
        path,
        "def _required_sm120_sparse_topk(vllm_config: VllmConfig, window_size: int) -> int:\n"
        "    \"\"\"Return the SM120 DSV4 SWA specialization needed by this model.\"\"\"\n"
        "    if not vllm_config.attention_config.use_non_causal:\n"
        "        return window_size\n"
        "    speculative_config = vllm_config.speculative_config\n"
        "    if speculative_config is None:\n"
        "        return window_size\n"
        "    return get_dspark_swa_index_width(\n"
        "        window_size,\n"
        "        speculative_config.num_speculative_tokens,\n"
        "    )\n\n\n"
        "class DeepseekV4FlashInferMLASparseBackend",
        "def _required_sm120_sparse_topk(vllm_config: VllmConfig, window_size: int) -> int:\n"
        "    \"\"\"Return the SM120 DSV4 SWA specialization needed by this model.\"\"\"\n"
        "    if not vllm_config.attention_config.use_non_causal:\n"
        "        return window_size\n"
        "    speculative_config = vllm_config.speculative_config\n"
        "    if speculative_config is None:\n"
        "        return window_size\n"
        "    return get_dspark_swa_index_width(\n"
        "        window_size,\n"
        "        speculative_config.num_speculative_tokens,\n"
        "    )\n\n\n"
        "def _use_b12x_compressed_mla_decode() -> bool:\n"
        "    raw = os.getenv(\"DS4FV_USE_B12X_COMPRESSED_MLA\", \"0\").strip().lower()\n"
        "    if raw in (\"1\", \"true\", \"yes\", \"on\"):\n"
        "        return True\n"
        "    if raw in (\"0\", \"false\", \"no\", \"off\"):\n"
        "        return False\n"
        "    raise ValueError(\n"
        "        \"DS4FV_USE_B12X_COMPRESSED_MLA must be a boolean value, \"\n"
        "        f\"got {raw!r}\"\n"
        "    )\n\n\n"
        "def _b12x_compressed_mla_limits(\n"
        "    vllm_config: VllmConfig,\n"
        "    *,\n"
        "    semantic_window: int,\n"
        "    physical_window: int,\n"
        "    padded_heads: int,\n"
        ") -> tuple[int, ...]:\n"
        "    speculative_config = vllm_config.speculative_config\n"
        "    num_spec = (\n"
        "        int(speculative_config.num_speculative_tokens)\n"
        "        if speculative_config is not None\n"
        "        else 0\n"
        "    )\n"
        "    scheduled_decode_rows = (\n"
        "        int(vllm_config.scheduler_config.max_num_seqs) * (num_spec + 1)\n"
        "    )\n"
        "    graph_rows = int(\n"
        "        vllm_config.compilation_config.max_cudagraph_capture_size or 0\n"
        "    )\n"
        "    max_rows = max(1, scheduled_decode_rows, graph_rows)\n\n"
        "    max_swa_width = max(\n"
        "        1,\n"
        "        int(semantic_window),\n"
        "        int(physical_window),\n"
        "        _required_sm120_sparse_topk(vllm_config, semantic_window),\n"
        "    )\n"
        "    hf_config = vllm_config.model_config.hf_config\n"
        "    ratios = {int(value) for value in getattr(hf_config, \"compress_ratios\", ())}\n"
        "    indexed_widths = [0]\n"
        "    if 4 in ratios:\n"
        "        indexed_widths.append(int(getattr(hf_config, \"index_topk\", 0)))\n"
        "    if 128 in ratios:\n"
        "        c128_width = (\n"
        "            int(vllm_config.model_config.max_model_len) + 127\n"
        "        ) // 128\n"
        "        # vLLM's C128 metadata rows use 128-entry alignment.\n"
        "        c128_width = ((c128_width + 127) // 128) * 128\n"
        "        indexed_widths.append(c128_width)\n"
        "    max_indexed_width = max(indexed_widths)\n"
        "    max_chunks = (max_swa_width + 63) // 64\n"
        "    if max_indexed_width:\n"
        "        max_chunks += (max_indexed_width + 63) // 64\n"
        "    return (\n"
        "        int(padded_heads),\n"
        "        max_rows,\n"
        "        max_swa_width,\n"
        "        max_indexed_width,\n"
        "        max(1, max_chunks),\n"
    "    )\n\n\n"
        "def _get_b12x_compressed_mla_workspace(\n"
        "    device: torch.device, limits: tuple[int, ...]\n"
        ") -> tuple[tuple[torch.device, tuple[int, ...]], object, torch.Tensor]:\n"
        "    from b12x.attention import compressed_sparse_mla\n\n"
        "    device = torch.device(device)\n"
        "    if device.type == \"cuda\" and device.index is None:\n"
        "        device = torch.device(\"cuda\", torch.cuda.current_device())\n"
        "    key = (device, limits)\n"
        "    workspace = _b12x_compressed_mla_workspace_by_key.get(key)\n"
        "    if workspace is None:\n"
        "        heads, max_rows, max_swa, max_indexed, max_chunks = limits\n"
        "        plan = compressed_sparse_mla.plan(\n"
        "            compressed_sparse_mla.Caps(\n"
        "                device=device,\n"
        "                num_q_heads=heads,\n"
        "                max_q_rows=max_rows,\n"
        "                max_width=max_swa + max_indexed,\n"
        "                max_batch=max_rows,\n"
        "                max_chunks_per_row=max_chunks,\n"
        "                decode_row_capacity=max_rows,\n"
        "                swa_width=max_swa,\n"
        "                indexed_width=max_indexed,\n"
        "                # Runtime pages differ across SWA/C4/C128; page sizes\n"
        "                # are validated from each live cache by run().\n"
        "                swa_page_size=1,\n"
        "                indexed_page_size=1,\n"
        "                mode=\"decode\",\n"
        "                use_cuda_graph=True,\n"
        "            )\n"
        "        )\n"
        "        scratch_spec = plan.scratch_specs()[0]\n"
        "        scratch = torch.empty(\n"
        "            scratch_spec.shape,\n"
        "            dtype=scratch_spec.dtype,\n"
        "            device=scratch_spec.device,\n"
        "        )\n"
        "        workspace = (plan, scratch)\n"
        "        _b12x_compressed_mla_workspace_by_key[key] = workspace\n"
        "        logger.info_once(\n"
        "            \"Reserved shared B12x compressed MLA decode workspace: \"\n"
        "            \"rows=%d heads=%d swa=%d indexed=%d chunks=%d bytes=%d\",\n"
        "            max_rows,\n"
        "            heads,\n"
        "            max_swa,\n"
        "            max_indexed,\n"
        "            max_chunks,\n"
        "            scratch_spec.nbytes,\n"
        "        )\n"
        "    return key, workspace[0], workspace[1]\n\n\n"
        "class DeepseekV4FlashInferMLASparseBackend",
        "B12x compressed MLA shared planning",
    )
    replace_once(
        path,
        "        self._b12x_o_proj_enabled = (\n"
        "            vllm_config.kernel_config.linear_backend == \"b12x\"\n"
        "        )\n"
        "        self._b12x_o_proj_weights = None\n"
        "        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config\n",
        "        self._b12x_o_proj_enabled = (\n"
        "            vllm_config.kernel_config.linear_backend == \"b12x\"\n"
        "        )\n"
        "        self._b12x_o_proj_weights = None\n"
        "        self._b12x_compressed_mla_enabled = (\n"
        "            self._b12x_o_proj_enabled\n"
        "            and self.use_fp8_ds_mla_layout\n"
        "            and _use_b12x_compressed_mla_decode()\n"
        "        )\n"
        "        self._b12x_compressed_mla_limits = (\n"
        "            _b12x_compressed_mla_limits(\n"
        "                vllm_config,\n"
        "                semantic_window=self.window_size,\n"
        "                physical_window=self.swa_cache_window_size,\n"
        "                padded_heads=self.padded_heads,\n"
        "            )\n"
        "            if self._b12x_compressed_mla_enabled\n"
        "            else None\n"
        "        )\n"
        "        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config\n",
        "B12x compressed MLA model qualification",
    )
    replace_once(
        path,
        "    def _reserve_empty_forward_workspace(self) -> None:\n"
        "        self._get_workspace(\n"
        "            torch.device(\"cuda\", torch.accelerator.current_device_index())\n"
        "        )\n\n"
        "    def _forward_sparse_impl(\n",
        "    def _get_b12x_compressed_mla_workspace(\n"
        "        self, device: torch.device\n"
        "    ) -> tuple[tuple[torch.device, tuple[int, ...]], object, torch.Tensor]:\n"
        "        limits = self._b12x_compressed_mla_limits\n"
        "        if limits is None:\n"
        "            raise RuntimeError(\"B12x compressed MLA decode is not enabled\")\n"
        "        return _get_b12x_compressed_mla_workspace(device, limits)\n\n"
        "    def _reserve_empty_forward_workspace(self) -> None:\n"
        "        device = torch.device(\n"
        "            \"cuda\", torch.accelerator.current_device_index()\n"
        "        )\n"
        "        self._get_workspace(device)\n"
        "        if self._b12x_compressed_mla_enabled:\n"
        "            self._get_b12x_compressed_mla_workspace(device)\n\n"
        "    def _forward_sparse_impl(\n",
        "B12x compressed MLA workspace reservation",
    )
    replace_once(
        path,
        "    def _forward_decode(\n"
        "        self,\n"
        "        q: torch.Tensor,\n"
        "        kv_cache: torch.Tensor | None,\n"
        "        swa_metadata: \"DeepseekSparseSWAMetadata\",\n"
        "        attn_metadata: DeepseekV4FlashMLAMetadata | None,\n"
        "        swa_only: bool,\n"
        "        output: torch.Tensor,\n"
        "    ) -> None:\n",
        "    @staticmethod\n"
        "    def _as_b12x_sparse_cache(kv_cache: torch.Tensor) -> torch.Tensor:\n"
        "        if kv_cache.dtype == torch.float8_e4m3fn:\n"
        "            kv_cache = kv_cache.view(torch.uint8)\n"
        "        if kv_cache.dtype != torch.uint8:\n"
        "            raise TypeError(\n"
        "                \"B12x compressed MLA cache must use uint8/FP8 storage, \"\n"
        "                f\"got {kv_cache.dtype}\"\n"
        "            )\n"
        "        if kv_cache.dim() == 4:\n"
        "            if int(kv_cache.shape[-2]) != 1:\n"
        "                raise ValueError(\n"
        "                    \"B12x compressed MLA expects one shared KV head, \"\n"
        "                    f\"got cache shape {tuple(kv_cache.shape)}\"\n"
        "                )\n"
        "            kv_cache = kv_cache.squeeze(-2)\n"
        "        if kv_cache.dim() == 2:\n"
        "            return kv_cache\n"
        "        if kv_cache.dim() != 3:\n"
        "            raise ValueError(\n"
        "                \"B12x compressed MLA cache must be rank 2, 3, or 4, \"\n"
        "                f\"got {tuple(kv_cache.shape)}\"\n"
        "            )\n"
        "        if (\n"
        "            int(kv_cache.stride(-1)) != 1\n"
        "            or int(kv_cache.stride(-2)) != int(kv_cache.shape[-1])\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"B12x compressed MLA requires a token-contiguous packed cache; \"\n"
        "                f\"got shape={tuple(kv_cache.shape)} \"\n"
        "                f\"stride={tuple(kv_cache.stride())}\"\n"
        "            )\n"
        "        return torch.as_strided(\n"
        "            kv_cache,\n"
        "            size=(\n"
        "                int(kv_cache.shape[0]),\n"
        "                int(kv_cache.shape[1]) * int(kv_cache.shape[2]),\n"
        "            ),\n"
        "            stride=(int(kv_cache.stride(0)), 1),\n"
        "        )\n\n"
        "    def _b12x_compressed_mla_decode(\n"
        "        self,\n"
        "        *,\n"
        "        q: torch.Tensor,\n"
        "        swa_cache: torch.Tensor,\n"
        "        swa_indices: torch.Tensor,\n"
        "        swa_lengths: torch.Tensor,\n"
        "        indexed_cache: torch.Tensor | None,\n"
        "        indexed_indices: torch.Tensor | None,\n"
        "        indexed_lengths: torch.Tensor | None,\n"
        "        output: torch.Tensor,\n"
        "    ) -> None:\n"
        "        from b12x.attention import compressed_sparse_mla\n"
        "        from b12x.attention._shared.mla.compressed_reference import (\n"
        "            COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,\n"
        "        )\n\n"
        "        swa_cache = self._as_b12x_sparse_cache(swa_cache)\n"
        "        if int(swa_cache.shape[1]) % COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN:\n"
        "            raise ValueError(\n"
        "                \"B12x SWA cache page is not an integral compressed-MLA page\"\n"
        "            )\n"
        "        swa_page_size = (\n"
        "            int(swa_cache.shape[1]) // COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN\n"
        "        )\n"
        "        indexed_page_size = None\n"
        "        if indexed_cache is not None:\n"
        "            indexed_cache = self._as_b12x_sparse_cache(indexed_cache)\n"
        "            if (\n"
        "                int(indexed_cache.shape[1])\n"
        "                % COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN\n"
        "            ):\n"
        "                raise ValueError(\n"
        "                    \"B12x indexed cache page is not an integral \"\n"
        "                    \"compressed-MLA page\"\n"
        "                )\n"
        "            indexed_page_size = (\n"
        "                int(indexed_cache.shape[1])\n"
        "                // COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN\n"
        "            )\n\n"
        "        key, plan, scratch = self._get_b12x_compressed_mla_workspace(q.device)\n"
        "        scratch_views = _b12x_compressed_mla_scratch_by_key.get(key)\n"
        "        binding_args = dict(\n"
        "            q=q,\n"
        "            swa_indices=swa_indices,\n"
        "            swa_lengths=swa_lengths,\n"
        "            indexed_indices=indexed_indices,\n"
        "            indexed_lengths=indexed_lengths,\n"
        "        )\n"
        "        if scratch_views is None:\n"
        "            binding = compressed_sparse_mla.bind(\n"
        "                plan, scratch=scratch, **binding_args\n"
        "            )\n"
        "            scratch_views = binding.scratch\n"
        "            _b12x_compressed_mla_scratch_by_key[key] = scratch_views\n"
        "        else:\n"
        "            binding = scratch_views.bind(**binding_args)\n"
        "        compressed_sparse_mla.run(\n"
        "            binding=binding,\n"
        "            swa_k_cache=swa_cache,\n"
        "            swa_page_size=swa_page_size,\n"
        "            indexed_k_cache=indexed_cache,\n"
        "            indexed_page_size=indexed_page_size,\n"
        "            attn_sink=self.attn_sink,\n"
        "            sm_scale=self.scale,\n"
        "            expected_num_q_heads=self.padded_heads,\n"
        "            out=output,\n"
        "        )\n"
        "        logger.info_once(\n"
        "            \"Using shared B12x compressed sparse MLA decode on SM12x.\"\n"
        "        )\n\n"
        "    def _forward_decode(\n"
        "        self,\n"
        "        q: torch.Tensor,\n"
        "        kv_cache: torch.Tensor | None,\n"
        "        swa_metadata: \"DeepseekSparseSWAMetadata\",\n"
        "        attn_metadata: DeepseekV4FlashMLAMetadata | None,\n"
        "        swa_only: bool,\n"
        "        output: torch.Tensor,\n"
        "    ) -> None:\n",
        "B12x compressed MLA decode helpers",
    )
    replace_once(
        path,
        "        flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
        "            query=q,\n"
        "            swa_kv_cache=swa_cache,\n"
        "            workspace_buffer=self._get_workspace(q.device),\n"
        "            sparse_indices=swa_indices,\n"
        "            compressed_kv_cache=extra_cache,\n"
        "            out=output,\n"
        "            bmm1_scale=self.scale,\n"
        "            sinks=self.attn_sink,\n"
        "            kv_layout=\"NHD\",\n"
        "            swa_topk_lens=swa_lens,\n"
        "            extra_sparse_indices=extra_sparse_indices,\n"
        "            extra_sparse_topk_lens=extra_sparse_lengths,\n"
        "        )\n",
        "        if self._b12x_compressed_mla_enabled:\n"
        "            self._b12x_compressed_mla_decode(\n"
        "                q=q,\n"
        "                swa_cache=swa_cache,\n"
        "                swa_indices=swa_indices,\n"
        "                swa_lengths=swa_lens,\n"
        "                indexed_cache=extra_cache,\n"
        "                indexed_indices=extra_sparse_indices,\n"
        "                indexed_lengths=extra_sparse_lengths,\n"
        "                output=output,\n"
        "            )\n"
        "        else:\n"
        "            flashinfer_trtllm_batch_decode_sparse_mla_dsv4(\n"
        "                query=q,\n"
        "                swa_kv_cache=swa_cache,\n"
        "                workspace_buffer=self._get_workspace(q.device),\n"
        "                sparse_indices=swa_indices,\n"
        "                compressed_kv_cache=extra_cache,\n"
        "                out=output,\n"
        "                bmm1_scale=self.scale,\n"
        "                sinks=self.attn_sink,\n"
        "                kv_layout=\"NHD\",\n"
        "                swa_topk_lens=swa_lens,\n"
        "                extra_sparse_indices=extra_sparse_indices,\n"
        "                extra_sparse_topk_lens=extra_sparse_lengths,\n"
        "            )\n",
        "B12x compressed MLA decode dispatch",
    )


def patch_warmup(root: Path) -> None:
    path = root / "model_executor/warmup/b12x_warmup.py"
    replace_once(
        path,
        "    model = worker.get_model()\n"
        "    max_tokens = worker.scheduler_config.max_num_batched_tokens\n",
        "    model = worker.get_model()\n"
        "    device = torch.device(\n"
        "        \"cuda\", torch.accelerator.current_device_index()\n"
        "    )\n"
        "    b12x_compressed_mla_layers = 0\n"
        "    for module in model.modules():\n"
        "        if not getattr(module, \"_b12x_compressed_mla_enabled\", False):\n"
        "            continue\n"
        "        module._get_b12x_compressed_mla_workspace(device)\n"
        "        b12x_compressed_mla_layers += 1\n"
        "    max_tokens = worker.scheduler_config.max_num_batched_tokens\n",
        "B12x compressed MLA pre-capture warmup",
    )
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
        "        )\n\n"
        "    if b12x_compressed_mla_layers:\n"
        "        # Execute the real model attention path after the shared linear\n"
        "        # and MoE warmups but before CUDA graph capture. This materializes\n"
        "        # the shared binding views and JITs the three DeepSeek layer\n"
        "        # regimes against live packed KV-cache strides.\n"
        "        worker.model_runner._dummy_run(\n"
        "            num_tokens=min(16, max_tokens),\n"
        "            skip_eplb=True,\n"
        "            is_profile=True,\n"
        "            force_attention=True,\n"
        "            create_mixed_batch=True,\n"
        "        )\n"
        "        logger.info_once(\n"
        "            \"Warmed shared B12x compressed MLA decode across %d layers.\",\n"
        "            b12x_compressed_mla_layers,\n"
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
    patch_b12x_compressed_mla_decode(root)
    patch_warmup(root)


if __name__ == "__main__":
    main()
