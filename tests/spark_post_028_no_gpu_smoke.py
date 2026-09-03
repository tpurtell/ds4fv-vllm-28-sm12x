#!/usr/bin/env python3
"""Source-only contracts for the post-v0.28 DeepSeek backports."""

from __future__ import annotations

import os
from pathlib import Path


root = Path(
    os.environ.get(
        "VLLM_PACKAGE_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"
    )
)


def read(relative: str) -> str:
    path = root / relative
    assert path.is_file(), path
    return path.read_text()


rope = read("models/deepseek_v4/common/rope.py")
assert 'key = "compress" if compress_ratio > 1 else "main"' in rope
assert "rope_parameters = dict(rope_parameters[key])" in rope
assert 'rope_parameters["factor"] = 1.0' in rope

gate = read("model_executor/layers/fused_moe/router/gate_linear.py")
assert "self._router_gemm_cublas_capable" in gate
assert "current_platform.is_cuda() or current_platform.is_rocm()" in gate

dsv4_root = root / "models/deepseek_v4"
assert not (dsv4_root / "eager_scratch.py").exists()
for source_path in dsv4_root.rglob("*.py"):
    source = source_path.read_text()
    assert "eager_scratch" not in source, source_path
    assert "_global_topk_output_buffers" not in source, source_path

attention = read("models/deepseek_v4/attention.py")
assert "q_out = torch.empty(" in attention
assert "qnorm_rope_kv_insert_nvfp4_ds_mla" in attention
assert "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(" in attention
assert "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_out(" not in attention

cache_utils = read("models/deepseek_v4/common/ops/cache_utils.py")
assert "output_buffers:" not in cache_utils
assert "global_topk_indices = torch.empty_like(topk_indices)" in cache_utils

parser = read("parser/deepseek_v4.py")
assert "DSML_PARAM_START" in parser
assert '"PARAM_START": DSML_PARAM_START' in parser
assert "(?=<{_ESCAPED_DSML}parameter\\s+name=)" in parser

chat_utils = read("entrypoints/chat_utils.py")
assert "arguments that are not valid" in chat_utils
assert '"JSON object; coercing to an empty object."' in chat_utils

encoding = read("tokenizers/deepseek_v4_encoding.py")
assert 'role == "system"' in encoding
assert "index == len(messages) - 1" in encoding

block_table = read("v1/worker/gpu/block_table.py")
assert "self.kernel_block_sizes_tensor" in block_table
assert "kv_block_size = tl.load(block_sizes + group_id)" in block_table
assert "kernel_block_size = tl.load(kernel_block_sizes + group_id)" in block_table
assert "virtual_block_size = kv_block_size * group_cp_size" in block_table
assert "block_indices = local_positions // kernel_block_size" in block_table

kv_interface = read("v1/kv_cache_interface.py")
assert "non_causal_mtd_set" in kv_interface
assert "len(non_causal_mtd_set) == 1" in kv_interface

print("Post-v0.28 source contracts passed without importing vLLM or CUDA")
