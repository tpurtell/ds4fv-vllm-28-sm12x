#!/usr/bin/env python3
"""CUDA-hidden contracts for DCP2 sliding-window cache management."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import (
    SlidingWindowManager,
    get_manager_for_kv_cache_spec,
)
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec


def make_spec(model_version: str | None) -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.float8_e4m3fn,
        sliding_window=128,
        extra_retained_tokens=3,
        model_version=model_version,
    )


spec = make_spec(None)
dsv4_spec = make_spec("deepseek_v4")
max_in_flight_tokens = 8192
max_model_len = 500_000
expected_dcp2_blocks = 18
expected_replicated_blocks = 34
assert spec.max_admission_blocks_per_request(
    max_in_flight_tokens=max_in_flight_tokens,
    max_model_len=max_model_len,
    dcp_world_size=2,
) == expected_dcp2_blocks
assert dsv4_spec.max_admission_blocks_per_request(
    max_in_flight_tokens=max_in_flight_tokens,
    max_model_len=max_model_len,
    dcp_world_size=1,
) == expected_replicated_blocks
config = SimpleNamespace(
    max_in_flight_tokens=max_in_flight_tokens,
    model_config=SimpleNamespace(max_model_len=max_model_len),
    parallel_config=SimpleNamespace(decode_context_parallel_size=2),
)
assert spec.max_memory_usage_bytes(config) == expected_dcp2_blocks * spec.page_size_bytes
assert dsv4_spec.max_memory_usage_bytes(config) == (
    expected_replicated_blocks * dsv4_spec.page_size_bytes
)

finder = inspect.getsource(SlidingWindowManager.find_longest_cache_hit)
mask = inspect.getsource(SlidingWindowManager.reachable_block_mask)
factory = inspect.getsource(get_manager_for_kv_cache_spec)
coordinator = inspect.getsource(HybridKVCacheCoordinator)
assert "kv_cache_spec.block_size * dcp_world_size" in finder
assert "effective_block_size or kv_cache_spec.block_size" in mask
assert 'admission_kwargs["dcp_world_size"]' in factory
assert "(FullAttentionSpec, SlidingWindowSpec, MambaSpec)" in coordinator
assert "DCP not support sliding window" not in finder

print("Spark CUDA-hidden DCP2 sliding-window cache contracts passed")
