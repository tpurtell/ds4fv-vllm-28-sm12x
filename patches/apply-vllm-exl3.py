#!/usr/bin/env python3
"""Port the immutable EXL3 backend onto vLLM 0.28 and B12x.

The source image contributes only ``exl3.py``.  Every edit below is anchored
against that file or the pinned vLLM 0.28 package so version drift fails the
image build instead of producing a partially patched runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label} expected exactly one source anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


def replace_span(
    path: Path,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> None:
    source = path.read_text()
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(
            f"{path}: {label} requires unique span anchors; "
            f"start={source.count(start)}, end={source.count(end)}"
        )
    begin = source.index(start)
    finish = source.index(end, begin)
    path.write_text(source[:begin] + replacement + source[finish:])


def patch_registry(root: Path) -> None:
    path = root / "model_executor/layers/quantization/__init__.py"
    replace_once(
        path,
        '    "deepseek_v4_fp8",\n    "online",',
        '    "deepseek_v4_fp8",\n    "exl3",\n    "online",',
        "EXL3 quantization literal",
    )
    replace_once(
        path,
        "    from .experts_int8 import ExpertsInt8Config\n",
        "    from .experts_int8 import ExpertsInt8Config\n"
        "    from .exl3 import Exl3Config\n",
        "EXL3 config import",
    )
    replace_once(
        path,
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "humming": HummingConfig,',
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "exl3": Exl3Config,\n'
        '        "humming": HummingConfig,',
        "EXL3 config mapping",
    )


def patch_deepseek_model(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/model.py"
    replace_once(
        path,
        '            prefix=f"{prefix}.experts",\n'
        "            scoring_func=self.scoring_func,",
        '            prefix=f"{prefix}.experts",\n'
        '            ckpt_names=("w1", "w2", "w3"),\n'
        "            scoring_func=self.scoring_func,",
        "DeepSeek V4 expert checkpoint names",
    )
    replace_once(
        path,
        "        self.config = config\n"
        '        expert_dtype = getattr(config, "expert_dtype", "fp4")',
        "        self.config = config\n"
        "        self.quant_config = vllm_config.quant_config\n"
        '        expert_dtype = getattr(config, "expert_dtype", "fp4")',
        "retain target quantization config",
    )
    replace_once(
        path,
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        '        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])',
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        normalize = getattr(\n"
        "            self.quant_config, \"normalize_standard_weight_name\", None\n"
        "        )\n"
        "        if normalize is not None:\n"
        "            weights = ((normalize(name), weight) for name, weight in weights)\n"
        '        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])',
        "normalize standard target expert names",
    )


def patch_dspark(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/dspark.py"
    replace_once(
        path,
        "        for name, loaded_weight in weights:\n"
        "            mapped = self._remap_dspark_name(name)",
        "        normalize = getattr(\n"
        "            self.quant_config, \"normalize_standard_weight_name\", None\n"
        "        )\n"
        "        for name, loaded_weight in weights:\n"
        "            original_name = name\n"
        "            if normalize is not None:\n"
        "                name = normalize(name)\n"
        "            mapped = self._remap_dspark_name(name)",
        "normalize standard DSpark expert names",
    )
    replace_once(
        path,
        "                        loaded_weight,\n"
        "                        name_mapped,\n"
        "                        shard_id=shard_id,\n",
        "                        loaded_weight,\n"
        "                        original_name,\n"
        "                        shard_id=shard_id,\n",
        "retain DSpark checkpoint name in loader diagnostics",
    )


def patch_mtp(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/mtp.py"
    replace_once(
        path,
        "        for name, loaded_weight in weights:\n"
        "            mtp_layer_idx = _find_mtp_layer_idx(name)",
        "        normalize = getattr(\n"
        "            self.quant_config, \"normalize_standard_weight_name\", None\n"
        "        )\n"
        "        for name, loaded_weight in weights:\n"
        "            original_name = name\n"
        "            if normalize is not None:\n"
        "                name = normalize(name)\n"
        "            mtp_layer_idx = _find_mtp_layer_idx(name)",
        "normalize standard MTP expert names",
    )
    replace_once(
        path,
        "                            loaded_weight,\n"
        "                            name_mapped,\n"
        "                            shard_id=expert_shard_id,\n",
        "                            loaded_weight,\n"
        "                            original_name,\n"
        "                            shard_id=expert_shard_id,\n",
        "retain MTP checkpoint name in loader diagnostics",
    )


STANDARD_CONFIG_METHODS = '''    def _configure_standard_deepseek(
        self, hf_config: PretrainedConfig | None
    ) -> None:
        """Use B12x for projection-mixed standard DeepSeek EXL3 experts."""

        if hf_config is None or getattr(hf_config, "model_type", None) != "deepseek_v4":
            return
        if self.codebook != "mcg" or not self.tensor_storage:
            return
        standard_expert = re.compile(
            r"^(?:model\\.layers\\.\\d+|mtp\\.\\d+)\\.mlp\\.experts\\.\\d+\\."
            r"(?:gate_proj|up_proj|down_proj)$"
        )
        if not all(standard_expert.fullmatch(name) for name in self.tensor_storage):
            return

        num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", 0))
        pattern = re.compile(
            r"^(?:model\\.layers\\.(?P<layer>\\d+)|mtp\\.(?P<mtp>\\d+))"
            r"\\.mlp\\.experts\\.(?P<expert>\\d+)\\."
            r"(?P<projection>gate_proj|up_proj|down_proj)$"
        )
        families: dict[tuple[int, int], dict[str, int]] = {}
        for name, entry in self.tensor_storage.items():
            match = pattern.fullmatch(name)
            assert match is not None
            layer_index = (
                int(match.group("layer"))
                if match.group("layer") is not None
                else num_hidden_layers + int(match.group("mtp"))
            )
            raw_bits = entry.get("bits_per_weight", self.bits)
            try:
                bits = float(raw_bits)
            except (TypeError, ValueError):
                raise ValueError(
                    f"standard EXL3 tensor {name} has invalid "
                    f"bits_per_weight={raw_bits!r}"
                ) from None
            if bits != int(bits) or int(bits) not in (2, 3, 4, 5, 6):
                raise ValueError(
                    f"standard EXL3 tensor {name} requires an integral "
                    f"K2-K6 bitrate, got {raw_bits!r}"
                )
            key = (layer_index, int(match.group("expert")))
            projections = families.setdefault(key, {})
            projection = match.group("projection")
            if projection in projections:
                raise ValueError(f"duplicate standard EXL3 tensor metadata: {name}")
            projections[projection] = int(bits)

        required = ("gate_proj", "up_proj", "down_proj")
        by_layer: dict[int, dict[int, tuple[int, int, int]]] = {}
        for (layer_index, expert_id), projections in families.items():
            if set(projections) != set(required):
                raise ValueError(
                    "standard EXL3 expert family is incomplete: "
                    f"layer={layer_index}, expert={expert_id}, "
                    f"projections={sorted(projections)}"
                )
            by_layer.setdefault(layer_index, {})[expert_id] = tuple(
                projections[name] for name in required
            )

        for layer_index, experts in by_layer.items():
            expected = list(range(max(experts) + 1))
            if sorted(experts) != expected:
                raise ValueError(
                    f"standard EXL3 layer {layer_index} has a sparse expert map"
                )
            projection_bitrates = tuple(experts[index] for index in expected)
            tiers = sorted({bit for rates in projection_bitrates for bit in rates})
            if len(tiers) > 1 and (
                len(tiers) != 2 or tiers[1] != tiers[0] + 1
            ):
                raise ValueError(
                    "projection-mixed EXL3 requires two consecutive tiers, "
                    f"layer={layer_index}, tiers={tiers}"
                )
            self.standard_projection_bits_by_layer[layer_index] = projection_bitrates

        self.standard_fused_moe = True
        # Keep the source spelling for file lookup and add NVIDIA DSV4 aliases
        # for routed-expert construction and codebook validation.
        aliases: dict[str, Any] = {}
        for name, entry in self.tensor_storage.items():
            alias = self.normalize_standard_weight_name(name)
            if alias.startswith("mtp."):
                match = re.match(r"^mtp\\.(\\d+)\\.(.*)$", alias)
                assert match is not None
                alias = (
                    f"model.layers.{num_hidden_layers + int(match.group(1))}."
                    f"{match.group(2)}"
                )
            aliases[alias] = entry
        self.tensor_storage.update(aliases)

    def standard_layer_projection_bitrates(
        self, layer_name: str, num_experts: int
    ) -> tuple[tuple[int, int, int], ...]:
        match = re.search(r"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)", layer_name)
        if match is None:
            raise ValueError(
                f"cannot resolve standard EXL3 layer index from {layer_name!r}"
            )
        layer_index = int(match.group(1))
        try:
            rates = self.standard_projection_bits_by_layer[layer_index]
        except KeyError as exc:
            raise ValueError(
                f"standard EXL3 bitrate map has no layer {layer_index}"
            ) from exc
        if len(rates) != num_experts:
            raise ValueError(
                "standard EXL3 expert count does not match metadata: "
                f"layer={layer_index}, metadata={len(rates)}, model={num_experts}"
            )
        return rates

'''


PROJECTION_PREPARE_METHOD = '''    def _prepare_mixed_rank_sliced_weights(self, layer: RoutedExperts) -> None:
        """Prepare all mixed expert projections for one public B12x plan."""

        api = _load_b12x_fused_moe()
        from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights

        num_experts = int(layer.local_num_experts)
        hidden_size = int(layer.exl3_hidden_size)
        intermediate_size = int(layer.exl3_intermediate_size_per_partition)
        rates = tuple(layer.exl3_projection_bitrates)
        if len(rates) != num_experts or any(len(entry) != 3 for entry in rates):
            raise ValueError(
                "projection-mixed EXL3 bitrate geometry does not match experts"
            )
        tier_bits = tuple(sorted({bit for entry in rates for bit in entry}))
        if len(tier_bits) != 2 or tier_bits[1] != tier_bits[0] + 1:
            raise ValueError(
                "projection-mixed EXL3 requires exactly two consecutive tiers, "
                f"got {tier_bits}"
            )

        w13_param = layer.w13_trellis
        w2_param = layer.w2_trellis
        gate_suh, up_suh = self._rank_sliced_backing(layer, "w13_suh")
        gate_svh, up_svh = self._rank_sliced_backing(layer, "w13_svh")
        down_suh = self._rank_sliced_backing(layer, "w2_suh")
        down_svh = self._rank_sliced_backing(layer, "w2_svh")
        intermediate_rotations = torch.cat(
            (gate_svh, up_svh, down_suh), dim=1
        ).contiguous()
        device = gate_suh.device
        tile_config = (64, 256, 64, 256)

        native_tiers = []
        tier_counts = []
        for bits in tier_bits:
            gate_ids = tuple(i for i, entry in enumerate(rates) if entry[0] == bits)
            up_ids = tuple(i for i, entry in enumerate(rates) if entry[1] == bits)
            down_ids = tuple(i for i, entry in enumerate(rates) if entry[2] == bits)
            last = 16 * bits

            gate = tuple(w13_param.exl3_tensors[(i, "w1")] for i in gate_ids)
            up = tuple(w13_param.exl3_tensors[(i, "w3")] for i in up_ids)
            if gate or up:
                w13 = torch.stack(gate + up).contiguous()
            else:
                w13 = torch.zeros(
                    (1, hidden_size // 16, intermediate_size // 16, last),
                    dtype=torch.int16,
                    device=device,
                )
            down = tuple(w2_param.exl3_tensors[(i, "w2")] for i in down_ids)
            if down:
                w2 = torch.stack(down).contiguous()
            else:
                w2 = torch.zeros(
                    (1, intermediate_size // 16, hidden_size // 16, last),
                    dtype=torch.int16,
                    device=device,
                )
            expected_w13 = (
                max(len(gate_ids) + len(up_ids), 1),
                hidden_size // 16,
                intermediate_size // 16,
                last,
            )
            expected_w2 = (
                max(len(down_ids), 1),
                intermediate_size // 16,
                hidden_size // 16,
                last,
            )
            if tuple(w13.shape) != expected_w13 or tuple(w2.shape) != expected_w2:
                raise ValueError(
                    f"projection-mixed EXL3 K{bits} slab mismatch: "
                    f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}, "
                    f"expected={expected_w13}/{expected_w2}"
                )
            native_tiers.append(
                ProjectionTrellisTierWeights(
                    bits=bits,
                    w13=w13,
                    w2=w2,
                    gate_experts=gate_ids,
                    up_experts=up_ids,
                    down_experts=down_ids,
                )
            )
            tier_counts.append((bits, len(gate_ids), len(up_ids), len(down_ids)))

        weight_plan = api.plan_weights(
            quant_modes="w4a16",
            source_format="exl3_trellis_mcg",
            activation=layer.activation.value,
            params_dtype=layer.exl3_params_dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout="w13",
            trellis_bits=tier_bits[0],
            trellis_tile_config=tile_config,
            trellis_codebook="mcg",
            trellis_rate_granularity="per_expert_projection",
        )
        layer.exl3_trellis_weights = api.prepare_weights(
            plan=weight_plan,
            params_dtype=layer.exl3_params_dtype,
            projection_tiers=tuple(native_tiers),
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
        )
        layer.exl3_trellis_tile_config = tile_config

        # The prepared owner holds compact projection-tier allocations. Drop
        # every per-expert source view so the unified-memory peak cannot retain
        # both checkpoint and execution layouts for later layers.
        for prefix in ("w13", "w2"):
            for suffix in ("suh", "svh", "trellis", "mcg", "mul1"):
                param = getattr(layer, f"{prefix}_{suffix}")
                param.exl3_tensors.clear()
                param.exl3_backing = None
        logger.info("EXL3 projection-mixed Trellis %s: %s", layer.layer_name, tier_counts)

'''


def patch_exl3(root: Path) -> None:
    path = root / "model_executor/layers/quantization/exl3.py"
    replace_once(path, "from types import SimpleNamespace\n", "", "remove private API shim")
    replace_once(
        path,
        "_SPARKINFER_MIXED_TRELLIS_API: Any | None = None\n",
        "",
        "remove private mixed API cache",
    )
    replace_once(
        path,
        "_MIXED_TRELLIS_RUNTIMES: dict[tuple[Any, ...], dict[str, Any]] = {}\n",
        "",
        "remove private mixed runtime cache",
    )
    replace_once(
        path,
        "_MIXED_TRELLIS_ROUTE_BLOCK_SIZE = 8\n",
        "",
        "remove private mixed route constant",
    )
    replace_span(
        path,
        "def _load_b12x_mixed_trellis() -> Any:\n",
        "def _unique_tensor_storage_bytes(*buffers: Any) -> int:\n",
        "",
        "remove private mixed kernel loader",
    )
    replace_once(
        path,
        "        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}\n"
        "        self.standard_fused_moe = False",
        "        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}\n"
        "        self.standard_projection_bits_by_layer: dict[\n"
        "            int, tuple[tuple[int, int, int], ...]\n"
        "        ] = {}\n"
        "        self.standard_fused_moe = False",
        "projection-specific standard bitrate map",
    )
    replace_span(
        path,
        "    def _configure_standard_deepseek(\n",
        "    def normalize_standard_weight_name(self, name: str) -> str:\n",
        STANDARD_CONFIG_METHODS,
        "projection-specific standard metadata",
    )
    replace_once(
        path,
        "    del weight_name\n"
        "    param.load_exl3_weight(\n"
        "        loaded_weight,\n"
        "        expert_id=expert_id,\n"
        "        shard_id=shard_id,\n"
        "    )\n"
        "    return True if return_success else None\n",
        "    try:\n"
        "        param.load_exl3_weight(\n"
        "            loaded_weight, expert_id=expert_id, shard_id=shard_id\n"
        "        )\n"
        "    except (RuntimeError, ValueError) as exc:\n"
        "        raise type(exc)(\n"
        "            f\"{exc}; weight={weight_name!r}, expert={expert_id}, \"\n"
        "            f\"shard={shard_id!r}, shape={tuple(loaded_weight.shape)}\"\n"
        "        ) from exc\n"
        "    return True if return_success else None\n",
        "expert loader diagnostics",
    )
    replace_once(
        path,
        "            layer.exl3_layer_bitrates = (\n"
        "                self.quant_config.rank_sliced_layer_bitrates(str(layer.layer_name))\n"
        "                if rank_sliced\n"
        "                else (int(self.quant_config.bits),) * num_experts\n"
        "            )\n"
        "            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1",
        "            if rank_sliced:\n"
        "                layer.exl3_layer_bitrates = (\n"
        "                    self.quant_config.rank_sliced_layer_bitrates(\n"
        "                        str(layer.layer_name)\n"
        "                    )\n"
        "                )\n"
        "                layer.exl3_projection_bitrates = tuple(\n"
        "                    (bits, bits, bits) for bits in layer.exl3_layer_bitrates\n"
        "                )\n"
        "            else:\n"
        "                layer.exl3_projection_bitrates = (\n"
        "                    self.quant_config.standard_layer_projection_bitrates(\n"
        "                        str(layer.layer_name), num_experts\n"
        "                    )\n"
        "                )\n"
        "                layer.exl3_layer_bitrates = tuple(\n"
        "                    rates[0] for rates in layer.exl3_projection_bitrates\n"
        "                )\n"
        "            all_rates = {\n"
        "                bits\n"
        "                for rates in layer.exl3_projection_bitrates\n"
        "                for bits in rates\n"
        "            }\n"
        "            layer.exl3_mixed_bitrate = len(all_rates) > 1",
        "attach projection-specific layer rates",
    )
    replace_span(
        path,
        "    def _prepare_mixed_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n",
        "    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n",
        PROJECTION_PREPARE_METHOD,
        "public projection-mixed preparation",
    )
    replace_once(
        path,
        "            trellis_bits=bits,\n"
        "            trellis_tile_config=tile_config,\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(",
        "            trellis_bits=bits,\n"
        "            trellis_tile_config=tile_config,\n"
        "            trellis_codebook=\"mcg\",\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(",
        "uniform Trellis codebook contract",
    )
    replace_once(
        path,
        "        marker = layer.w13_mcg.exl3_tensors[(0, \"w1\")]\n"
        "        weight_plan = api.plan_weights(\n"
        "            quant_modes=\"w4a16\",\n"
        "            source_format=\"exl3_trellis_mcg\",\n"
        "            activation=layer.activation.value,\n"
        "            params_dtype=layer.exl3_params_dtype,",
        "        marker = layer.w13_mcg.exl3_tensors[(0, \"w1\")]\n"
        "        weight_plan = api.plan_weights(\n"
        "            quant_modes=\"w4a16\",\n"
        "            source_format=\"exl3_trellis_mcg\",\n"
        "            activation=layer.activation.value,\n"
        "            params_dtype=torch.float16,",
        "uniform full-rotation plan dtype",
    )
    replace_once(
        path,
        "            trellis_codebook=\"mcg\",\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(\n"
        "            plan=weight_plan,\n"
        "            params_dtype=layer.exl3_params_dtype,",
        "            trellis_codebook=\"mcg\",\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(\n"
        "            plan=weight_plan,\n"
        "            params_dtype=torch.float16,",
        "uniform full-rotation prepared dtype",
    )
    replace_span(
        path,
        "    def _mixed_rank_sliced_runtime(\n",
        "    def _rank_sliced_runtime(\n",
        "",
        "remove private mixed execution path",
    )
    replace_once(
        path,
        "            x.dtype,\n"
        "            int(layer.exl3_hidden_size),",
        "            x.dtype,\n"
        "            topk_ids.dtype,\n"
        "            int(layer.exl3_hidden_size),",
        "route dtype in runtime cache key",
    )
    replace_once(
        path,
        '                quant_mode="w4a16",\n'
        "                w4a16_block_size_m=plan_block_m,\n"
        "            )",
        '                quant_mode="w4a16",\n'
        "                w4a16_block_size_m=plan_block_m,\n"
        "                swiglu_limit=getattr(layer, \"swiglu_limit\", None),\n"
        "                full_rotation_output_dtype=(\n"
        "                    torch.bfloat16\n"
        "                    if x.dtype == torch.bfloat16\n"
        "                    else torch.float32\n"
        "                ),\n"
        "                mixed_trellis_route_id_dtypes=(topk_ids.dtype,),\n"
        "                mixed_trellis_broadcast_suh=(False,),\n"
        "                mixed_trellis_broadcast_svh=(False,),\n"
        "            )",
        "bounded public mixed launch variants",
    )
    replace_once(
        path,
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            return self._apply_mixed_rank_sliced(\n"
        "                layer,\n"
        "                x,\n"
        "                topk_weights,\n"
        "                topk_ids,\n"
        "            )\n"
        "        runtime = self._rank_sliced_runtime(layer, x, topk_ids)",
        "        runtime = self._rank_sliced_runtime(layer, x, topk_ids)",
        "route mixed execution through public planner",
    )
    source = path.read_text()
    forbidden = (
        "_load_b12x_mixed_trellis",
        "run_mixed_trellis",
        "compile_mixed_trellis",
        "_apply_mixed_rank_sliced",
    )
    leftovers = [name for name in forbidden if name in source]
    if leftovers:
        raise RuntimeError(f"private mixed EXL3 entry points remain: {leftovers}")


def main(root: Path) -> None:
    patch_registry(root)
    patch_deepseek_model(root)
    patch_dspark(root)
    patch_mtp(root)
    patch_exl3(root)
    paths = (
        root / "model_executor/layers/quantization/__init__.py",
        root / "model_executor/layers/quantization/exl3.py",
        root / "models/deepseek_v4/nvidia/model.py",
        root / "models/deepseek_v4/nvidia/dspark.py",
        root / "models/deepseek_v4/nvidia/mtp.py",
    )
    for path in paths:
        compile(path.read_text(), str(path), "exec")
    print("vLLM 0.28 EXL3 projection-mixed integration installed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_root", type=Path)
    args = parser.parse_args()
    main(args.vllm_root)
