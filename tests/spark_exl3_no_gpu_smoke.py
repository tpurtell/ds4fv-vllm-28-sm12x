#!/usr/bin/env python3
"""No-GPU validation for the v3 projection-mixed EXL3 integration."""

from __future__ import annotations

import inspect
import os

from transformers import AutoConfig

from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
from flashinfer.jit.mla import gen_sparse_mla_sm120_module


model_path = os.environ.get("EXL3_MODEL_PATH")
if not model_path:
    raise SystemExit("EXL3_MODEL_PATH must point to the pinned v3 snapshot")

assert get_quantization_config("exl3") is Exl3Config
hf_config = AutoConfig.from_pretrained(model_path)
quant_config = Exl3Config.from_config(dict(hf_config.quantization_config))
quant_config.maybe_update_config(model_path, hf_config=hf_config)

assert quant_config.standard_fused_moe
assert sorted(quant_config.standard_projection_bits_by_layer) == list(range(46))
for layer_index, rates in quant_config.standard_projection_bits_by_layer.items():
    assert len(rates) == 256, (layer_index, len(rates))
    assert {bit for triple in rates for bit in triple} == {2, 3}

# Backbone plus the three D2.2 draft layers.
assert sum(
    rates[0] == 3
    for layer in quant_config.standard_projection_bits_by_layer.values()
    for rates in layer
) == 705
assert sum(
    rates[1] == 3
    for layer in quant_config.standard_projection_bits_by_layer.values()
    for rates in layer
) == 1176
assert sum(
    rates[2] == 3
    for layer in quant_config.standard_projection_bits_by_layer.values()
    for rates in layer
) == 1882

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
    "EXL3 metadata: 46 layers x 256 projection-mixed experts; "
    "public B12x path only"
)
