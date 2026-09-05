#!/usr/bin/env python3
"""CUDA-hidden contracts for the isolated DeepSeek-V4 AG/RS combine fast path."""

from pathlib import Path

import vllm


root = Path(vllm.__file__).resolve().parent
source = (root / "models/deepseek_v4/nvidia/flashinfer_sparse.py").read_text()

for fragment in (
    "def _dcp_ag_rs_combine_into(",
    "_dcp_correct_and_pack_rs_kernel[(num_tokens, global_heads)](",
    "pynccl_comm.reduce_scatter(destination, packed_output)",
    "assert self.dcp_manager.query_gather is not None",
    "gathered_query = self.dcp_manager.query_gather(",
    "local_query.contiguous()",
    "del gathered_query",
    "return q.contiguous()",
):
    assert fragment in source, fragment

assert "def _dcp_head_major_query_gather(" not in source
assert "retain strided head-major DCP query" not in source

print("Spark CUDA-hidden DCP AG/RS combine-only contracts passed")
