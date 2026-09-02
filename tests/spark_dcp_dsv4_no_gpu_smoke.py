#!/usr/bin/env python3
"""CUDA-hidden contracts for DeepSeek-V4 compressed/SWA DCP2 attention."""

from __future__ import annotations

import math
from pathlib import Path

import vllm


def local_prefix_len(end: int, world: int, rank: int, interleave: int = 1) -> int:
    """Number of global records in ``[0, end)`` owned by ``rank``."""
    cycle = world * interleave
    full_cycles, remainder = divmod(end, cycle)
    rank_remainder = min(max(remainder - rank * interleave, 0), interleave)
    return full_cycles * interleave + rank_remainder


def owner_and_local(record: int, world: int, interleave: int = 1) -> tuple[int, int]:
    group, lane = divmod(record, interleave)
    return group % world, (group // world) * interleave + lane


def dcp_slot(
    position: int,
    block_id: int,
    block_size: int,
    world: int,
    rank: int,
    interleave: int = 1,
) -> int | None:
    """Mirror BlockTables' global-position to rank-local slot transform."""
    virtual_offset = position % (block_size * world)
    owner, local_offset = owner_and_local(virtual_offset, world, interleave)
    return block_id * block_size + local_offset if owner == rank else None


def softmax_value(logits: list[float], values: list[float]) -> tuple[float, float]:
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    denominator = sum(weights)
    return (
        sum(weight * value for weight, value in zip(weights, values, strict=True))
        / denominator,
        math.log(denominator) + maximum,
    )


# DCP ownership is a lossless, non-overlapping partition for SWA, C4, and C128.
for world in (2, 4):
    for interleave in (1, 2, 4):
        for end in range(0, 521):
            counts = [local_prefix_len(end, world, rank, interleave) for rank in range(world)]
            assert sum(counts) == end
            owned = [[] for _ in range(world)]
            for record in range(end):
                rank, local = owner_and_local(record, world, interleave)
                owned[rank].append(local)
            for rank in range(world):
                assert owned[rank] == list(range(counts[rank]))

# DSpark's custom context/query slot builder must produce the same disjoint
# per-rank mapping as the normal BlockTables path, including virtual-block
# rollover. Every global position has exactly one writer and each rank's local
# slots are dense within a physical cache block.
for world in (2, 4):
    for interleave in (1, 2, 4):
        block_size = 8
        for virtual_block in range(3):
            block_id = 10 + virtual_block
            writers = [[] for _ in range(world)]
            start = virtual_block * block_size * world
            for position in range(start, start + block_size * world):
                slots = [
                    dcp_slot(position, block_id, block_size, world, rank, interleave)
                    for rank in range(world)
                ]
                assert sum(slot is not None for slot in slots) == 1
                for rank, slot in enumerate(slots):
                    if slot is not None:
                        writers[rank].append(slot)
            for rank in range(world):
                assert writers[rank] == list(
                    range(block_id * block_size, (block_id + 1) * block_size)
                )

# Compression happens before DCP sharding. Check both cache families at the
# boundary where a completed compressed record changes owner.
for ratio in (4, 128):
    completed = []
    for position in range(ratio * 9):
        if (position + 1) % ratio == 0:
            record = position // ratio
            completed.append((position, record, *owner_and_local(record, 2)))
    assert [entry[2] for entry in completed] == [0, 1, 0, 1, 0, 1, 0, 1, 0]
    assert [entry[3] for entry in completed if entry[2] == 0] == [0, 1, 2, 3, 4]
    assert [entry[3] for entry in completed if entry[2] == 1] == [0, 1, 2, 3]

# C4 exact-global top-K is localized and compacted independently on each rank.
global_topk = [11, 0, 9, 4, 7, 2, 8, -1]
rank_local: list[list[int]] = []
for rank in range(2):
    local = [
        owner_and_local(record, 2)[1]
        for record in global_topk
        if record >= 0 and owner_and_local(record, 2)[0] == rank
    ]
    rank_local.append(local)
assert rank_local == [[0, 2, 1, 4], [5, 4, 3]]

# A sink is a single global softmax term. Including it once in one rank's LSE
# and then doing the normal LSE-weighted merge is exactly equivalent to the
# unsharded attention result.
rank0_logits, rank0_values = [0.7, -0.4, 0.2], [2.0, -1.0, 0.0]  # sink is last
rank1_logits, rank1_values = [0.1, 1.2], [3.0, 4.0]
expected, expected_lse = softmax_value(
    rank0_logits + rank1_logits, rank0_values + rank1_values
)
partial0, lse0 = softmax_value(rank0_logits, rank0_values)
partial1, lse1 = softmax_value(rank1_logits, rank1_values)
merged, merged_lse = softmax_value([lse0, lse1], [partial0, partial1])
assert math.isclose(merged, expected, rel_tol=1e-12, abs_tol=1e-12)
assert math.isclose(merged_lse, expected_lse, rel_tol=1e-12, abs_tol=1e-12)

# Rank-major packed reduce-scatter makes each destination's token-major output
# one contiguous NCCL chunk. Query gather deliberately retains vLLM's
# token-major contiguous layout because B12x is faster with that input.
tokens, local_heads, world = 3, 4, 2
packed_destinations = [
    [
        [(destination, token, head) for head in range(local_heads)]
        for token in range(tokens)
    ]
    for destination in range(world)
]
assert len(packed_destinations) == world
assert all(len(chunk) == tokens for chunk in packed_destinations)

root = Path(vllm.__file__).resolve().parent
sm120 = (root / "models/deepseek_v4/nvidia/flashinfer_sparse.py").read_text()
sparse_mla = (root / "models/deepseek_v4/sparse_mla.py").read_text()
swa = (root / "v1/attention/backends/mla/sparse_swa.py").read_text()
compressor = (root / "v1/attention/backends/mla/compressor_utils.py").read_text()
cache_utils = (root / "models/deepseek_v4/common/ops/cache_utils.py").read_text()
indexer = (root / "v1/attention/backends/mla/indexer.py").read_text()
attn_utils = (root / "v1/attention/backends/utils.py").read_text()
speculator = (root / "v1/worker/gpu/spec_decode/speculator.py").read_text()
dflash_speculator = (
    root / "v1/worker/gpu/spec_decode/dflash/speculator.py"
).read_text()
model_runner = (root / "v1/worker/gpu/model_runner.py").read_text()
draft_cudagraph = (
    root / "v1/worker/gpu/spec_decode/dflash/cudagraph.py"
).read_text()

for fragment in (
    "self.dcp_manager = MLADCPManager(",
    "gathered_query = self.dcp_manager.query_gather(",
    "local_query.contiguous()",
    "def _dcp_ag_rs_combine_into(",
    "_dcp_correct_and_pack_rs_kernel[(num_tokens, global_heads)](",
    "pynccl_comm.reduce_scatter(destination, packed_output)",
    "if self.dcp_manager.use_a2a:",
    "combined = self.dcp_manager.combine(",
    'return_lse=True, lse_scale="base2"',
    'is_lse_base_on_e=False',
    'self._dcp_attn_sink.fill_(-float("inf"))',
    "if self.dcp_rank == 0:",
    "swa_metadata.dcp_local_seq_lens[:num_reqs]",
    "self.dcp_world_size if self.compress_ratio == 4 else 1",
):
    assert fragment in sm120, fragment

assert "def _dcp_head_major_query_gather(" not in sm120
assert "rank_offsets = dcp_rank" in attn_utils
assert "rank_offsets = torch.tensor(dcp_rank" not in attn_utils

for fragment in (
    "supports_dcp_with_varlen=True",
    "global_num_compressed = (position + 1) // compress_ratio",
    "num_compressed = full_cycles * CP_INTERLEAVE + rank_remainder",
    "if DCP_WORLD_SIZE > 1:",
    "is_valid_token = True",
):
    assert fragment in sparse_mla, fragment

for fragment in (
    "A real query row exists on every DCP rank",
    "local_start = _dcp_local_prefix_len(",
    "local_end = _dcp_local_prefix_len(",
    "dcp_local_seq_lens=dcp_local_seq_lens",
):
    assert fragment in swa, fragment

assert "compressed_pos = pos // COMPRESS_RATIO" in compressor
assert "virtual_block_size = block_size * dcp_world_size" in compressor
assert "tl.cumsum(is_valid.to(tl.int32), axis=0)" in cache_utils
assert "owner == DCP_RANK" in cache_utils
assert "DCP is not supported with sparse indexer KV compression" not in indexer
for fragment in (
    "prepare_dcp_local_seq_lens(",
    "if self.block_tables.cp_size > 1:",
    "dcp_local_seq_lens=dcp_local_seq_lens",
):
    assert fragment in speculator, fragment
for fragment in (
    "self.block_tables.kv_cache_cp_sizes[gid]",
    "if self.block_tables.kv_cache_cp_sizes[gid] > 1",
    "else 0",
    "ctx_virtual_block_size = block_size * CP_SIZE",
    "ctx_resident = is_valid_ctx & ctx_is_local & (ctx_block_id != 0)",
    "q_virtual_block_size = block_size * CP_SIZE",
    "q_resident = is_query & q_is_local & (q_block_id != 0)",
    "CP_SIZE=cp_size",
    "CP_INTERLEAVE=cp_interleave",
):
    assert fragment in dflash_speculator, fragment
for fragment in (
    "Eager/profile dummy runs bypass prepare_inputs_to_capture()",
    "if self.use_dcp:",
    "input_batch.dcp_local_seq_lens = (",
):
    assert fragment in model_runner, fragment
for fragment in (
    "if block_tables.cp_size > 1:",
    "prepare_dcp_local_seq_lens(",
    "dcp_local_seq_lens=dcp_local_seq_lens",
):
    assert fragment in draft_cudagraph, fragment

print("Spark CUDA-hidden DeepSeek-V4 DCP2 attention contracts passed")
