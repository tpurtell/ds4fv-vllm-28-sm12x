#!/usr/bin/env python3
"""No-GPU validation for the qualified projection-mixed EXL3 profiles."""

from __future__ import annotations

import inspect
import os

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.transformers_utils.config import get_config
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
from flashinfer.jit.mla import gen_sparse_mla_sm120_module


model_path = os.environ.get("EXL3_MODEL_PATH")
if not model_path:
    raise SystemExit("EXL3_MODEL_PATH must point to a pinned EXL3 snapshot")

assert get_quantization_config("exl3") is Exl3Config
hf_config = get_config(model_path, trust_remote_code=False)
quant_config = Exl3Config.from_config(dict(hf_config.quantization_config))
quant_config.maybe_update_config(model_path, hf_config=hf_config)

assert quant_config.standard_fused_moe
assert sorted(quant_config.standard_projection_bits_by_layer) == list(range(46))
for layer_index, rates in quant_config.standard_projection_bits_by_layer.items():
    assert len(rates) == 256, (layer_index, len(rates))

projection_k3_totals = tuple(
    sum(
        rates[projection_index] == 3
        for layer in quant_config.standard_projection_bits_by_layer.values()
        for rates in layer
    )
    for projection_index in range(3)
)
profile_by_totals = {
    (705, 1176, 1882): "text-k2.1-d2.2-v3",
    (1238, 2064, 3303): "vision-k2.2-d2-v1",
}
profile = profile_by_totals.get(projection_k3_totals)
assert profile is not None, projection_k3_totals

declared_layer_types = getattr(hf_config, "mlp_layer_types", None)
if profile == "vision-k2.2-d2-v1":
    assert declared_layer_types[:3] == ["hash_moe"] * 3
    assert declared_layer_types[3:] == ["moe"] * 40

for layer_index, rates in quant_config.standard_projection_bits_by_layer.items():
    expected_bits = (
        {2}
        if profile == "vision-k2.2-d2-v1" and layer_index >= 43
        else {2, 3}
    )
    assert {bit for triple in rates for bit in triple} == expected_bits, layer_index

source = inspect.getsource(Exl3MoEMethod)
assert "api.prepare_weights" in source
assert "projection_tiers=tuple(native_tiers)" in source
assert source.count('trellis_codebook="mcg"') >= 2
assert source.count("params_dtype=torch.float16") >= 2
assert "tile_config = (64, 256, 64, 256)" in source
assert "run_mixed_trellis" not in source
assert "compile_mixed_trellis" not in source
assert has_flashinfer_sparse_mla_sm120_config(64, 192)
jit_source = inspect.getsource(gen_sparse_mla_sm120_module)
assert '"sparse_mla_sm120_ds4fv_k192_v1"' in jit_source

print(
    f"EXL3 metadata: {profile}; 46 layers x 256 experts; public B12x path only"
)
