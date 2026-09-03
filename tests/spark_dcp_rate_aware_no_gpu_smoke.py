#!/usr/bin/env python3
"""CUDA-hidden contracts for DeepSeek-V4 rate-aware DCP cache ownership."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import torch
import vllm

from vllm.v1.core.kv_cache_utils import (
    _get_kv_cache_groups_uniform_groups,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
    get_kv_cache_dcp_world_size,
)


def spec(*, ratio: int, model_version: str | None = "deepseek_v4"):
    return MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=584,
        alignment=576,
        compress_ratio=ratio,
        model_version=model_version,
    )


c4 = spec(ratio=4)
c4_indexer = spec(ratio=4, model_version=None)
c128 = spec(ratio=128)
swa = SlidingWindowMLASpec(
    block_size=64,
    num_kv_heads=1,
    head_size=512,
    dtype=torch.uint8,
    state_content_bytes=584,
    alignment=576,
    compress_ratio=1,
    model_version="deepseek_v4",
    sliding_window=512,
)
compressor_state = SlidingWindowMLASpec(
    block_size=4,
    num_kv_heads=1,
    head_size=2048,
    dtype=torch.float32,
    model_version="deepseek_v4",
    sliding_window=8,
)

assert get_kv_cache_dcp_world_size(c4, 2) == 2
assert get_kv_cache_dcp_world_size(c4_indexer, 2) == 2
assert get_kv_cache_dcp_world_size(c128, 2) == 1
assert get_kv_cache_dcp_world_size(swa, 2) == 1
assert get_kv_cache_dcp_world_size(compressor_state, 2) == 1
assert get_kv_cache_dcp_world_size(c128, 1) == 1

sharded = UniformTypeKVCacheSpecs.from_specs({"c4": c4, "indexer": c4_indexer})
replicated = UniformTypeKVCacheSpecs.from_specs({"c128": c128})
assert sharded is not None and replicated is not None
assert get_kv_cache_dcp_world_size(sharded, 2) == 2
assert get_kv_cache_dcp_world_size(replicated, 2) == 1

mixed = UniformTypeKVCacheSpecs.from_specs({"c4": c4, "c128": c128})
assert mixed is not None
try:
    get_kv_cache_dcp_world_size(mixed, 2)
except ValueError as error:
    assert "cannot mix replicated and DCP-sharded" in str(error)
else:
    raise AssertionError("mixed-ownership packed group was not rejected")

config = SimpleNamespace(
    model_config=SimpleNamespace(max_model_len=500_000),
    parallel_config=SimpleNamespace(
        decode_context_parallel_size=2,
        tensor_parallel_size=2,
    ),
    cache_config=SimpleNamespace(
        block_size=256,
        enable_prefix_caching=True,
        prefix_match_unit=None,
    ),
    kv_transfer_config=None,
    max_in_flight_tokens=8192,
)
assert c4.max_num_blocks_per_req(config, 500_000) == math.ceil(500_000 / 512)
assert c128.max_num_blocks_per_req(config, 500_000) == math.ceil(500_000 / 256)
assert c4.max_memory_usage_bytes(config) // c4.page_size_bytes == math.ceil(
    250_000 / 256
)
assert c128.max_memory_usage_bytes(config) // c128.page_size_bytes == math.ceil(
    500_000 / 256
)
assert swa.max_memory_usage_bytes(config) // swa.page_size_bytes == (
    math.ceil((512 - 1 + 8192) / 64) + 1
)

# The real packed-cache optimizer receives C4/indexer/C128 together. DCP2 must
# split that input before tuple packing so every resulting block table has one
# unambiguous ownership width. The resulting 512-token scheduling alignment
# still permits 64-token prefix hashes for replicated SWA blocks.
swa_group = UniformTypeKVCacheSpecs.from_specs({"swa": swa})
assert swa_group is not None
groups = _get_kv_cache_groups_uniform_groups(config, [mixed, swa_group])
assert len(groups) <= 6
for group in groups:
    get_kv_cache_dcp_world_size(group.kv_cache_spec, 2)
assert any("c128" in group.layer_names for group in groups)
assert all(
    not ({"c128"} & set(group.layer_names) and {"c4"} & set(group.layer_names))
    for group in groups
)
scheduler_config = KVCacheConfig(
    num_blocks=100,
    kv_cache_tensors=[],
    kv_cache_groups=groups,
)
assert resolve_kv_cache_block_sizes(scheduler_config, config) == (512, 64)

# Mirror the worker kernel's two address modes. Replicated groups produce the
# same dense slot on both ranks; C4 has exactly one owner and dense local slots.
def slot(position: int, block_id: int, block_size: int, cp_size: int, rank: int):
    offset = position % (block_size * cp_size)
    if cp_size == 1:
        return block_id * block_size + offset
    owner = offset % cp_size
    local = offset // cp_size
    return block_id * block_size + local if owner == rank else None


for position in range(32):
    assert slot(position, 7, 8, 1, 0) == slot(position, 7, 8, 1, 1)
for position in range(16):
    values = [slot(position, 9, 8, 2, rank) for rank in range(2)]
    assert sum(value is not None for value in values) == 1

# C4 record n is emitted at original-token position 4*n+3 but owned by
# compressed-record parity n%2. With striped state ownership, half the
# boundaries have no rank that owns both the boundary state and destination;
# even the other half cannot read the complete eight-state compressor window.
boundary_owner_mismatches = 0
for record in range(16):
    boundary = 4 * record + 3
    compressed_owner = record % 2
    state_owner = boundary % 2
    boundary_owner_mismatches += compressed_owner != state_owner
    assert slot(boundary, 3, 4, 1, compressed_owner) is not None
assert boundary_owner_mismatches == 8

root = Path(vllm.__file__).resolve().parent
block_table = (root / "v1/worker/gpu/block_table.py").read_text()
model_runner = (root / "v1/worker/gpu/model_runner.py").read_text()
kv_utils = (root / "v1/core/kv_cache_utils.py").read_text()
coordinator = (root / "v1/core/kv_cache_coordinator.py").read_text()
dflash = (root / "v1/worker/gpu/spec_decode/dflash/speculator.py").read_text()
sparse_mla = (root / "models/deepseek_v4/sparse_mla.py").read_text()
sparse_swa = (root / "v1/attention/backends/mla/sparse_swa.py").read_text()
compressor = (root / "models/deepseek_v4/compressor.py").read_text()
sm120 = (root / "models/deepseek_v4/nvidia/flashinfer_sparse.py").read_text()

for fragment in (
    "self.kv_cache_cp_sizes = (",
    "self.kv_cache_cp_sizes_tensor",
    "group_cp_size = tl.load(kv_cache_cp_sizes + group_id)",
    "is_sharded = group_cp_size == CP_SIZE",
):
    assert fragment in block_table, fragment
for fragment in (
    "group_cp_size = get_kv_cache_dcp_world_size(spec, self.dcp_size)",
    "kv_cache_cp_sizes=kv_cache_cp_sizes",
):
    assert fragment in model_runner, fragment
for fragment in (
    "split the original mixed full-MLA family",
    "get_kv_cache_dcp_world_size(g.kv_cache_spec, dcp)",
):
    assert fragment in kv_utils, fragment
assert "kv_cache_group.kv_cache_spec, dcp_world_size" in coordinator
for fragment in (
    "self.block_tables.kv_cache_cp_sizes[gid]",
    "if self.block_tables.kv_cache_cp_sizes[gid] > 1",
    "else 0",
):
    assert fragment in dflash, fragment
assert "configured_dcp if self.compress_ratio == 4 else 1" in sparse_mla
assert "self.dcp_world_size = 1" in sparse_swa
assert 'model_version="deepseek_v4"' in compressor
for fragment in (
    "self.dcp_world_size > 1 and self.compress_ratio == 4",
    "def _swa_lengths_for_shard(",
    "attention._dcp_empty_swa_lens",
    "swa_lens = _swa_lengths_for_shard(self, swa_lens)",
    "swa_lens_chunk = _swa_lengths_for_shard(self, swa_lens_chunk)",
):
    assert fragment in sm120, fragment

print("Spark CUDA-hidden DeepSeek-V4 rate-aware DCP contracts passed")
