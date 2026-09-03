#!/usr/bin/env python3
"""Backport post-v0.28 DCP correctness contracts onto the rate-aware path.

This adapts vllm-project/vllm#51031 to the recipe's per-cache-group DCP
geometry and takes the remaining cache-group invariant from #54277.  Logical
KV blocks define DCP interleaving; backend-selected kernel blocks define the
final page-table lookup and slot offset.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def patch_slot_mapping(root: Path) -> None:
    path = root / "v1/worker/gpu/block_table.py"
    replace_once(
        path,
        '''        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; block_sizes_tensor needs re-populating
        # because its storage lives under the kv_cache pool tag and comes back
        # with undefined contents.
''',
        '''        # cache raw data_ptr() values that go stale once the underlying tensors
        # are reallocated on wake; the size tensors need re-populating because
        # their storage lives under the kv_cache pool tag and come back with
        # undefined contents.
''',
        "#51031 size tensor wake-up contract",
    )
    replace_once(
        path,
        '''        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.kv_cache_cp_sizes_tensor = torch.tensor(
''',
        '''        self.block_sizes_tensor = torch.tensor(
            self.block_sizes, dtype=torch.int32, device=self.device
        )
        self.kernel_block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.kv_cache_cp_sizes_tensor = torch.tensor(
''',
        "#51031 logical and kernel block tensors",
    )
    replace_once(
        path,
        '''            self.block_table_strides,
            self.block_sizes_tensor,
            self.kv_cache_cp_sizes_tensor,
''',
        '''            self.block_table_strides,
            self.block_sizes_tensor,
            self.kernel_block_sizes_tensor,
            self.kv_cache_cp_sizes_tensor,
''',
        "#51031 slot mapping call",
    )
    replace_once(
        path,
        '''    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    kv_cache_cp_sizes,  # [num_kv_cache_groups]
''',
        '''    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    kernel_block_sizes,  # [num_kv_cache_groups]
    kv_cache_cp_sizes,  # [num_kv_cache_groups]
''',
        "#51031 slot mapping kernel arguments",
    )
    replace_once(
        path,
        '''    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)
    group_cp_size = tl.load(kv_cache_cp_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        if CP_SIZE == 1:
            block_indices = positions // block_size
            block_offsets = positions % block_size
            block_numbers = tl.load(
                block_table_ptr + req_state_idx * block_table_stride + block_indices
            )
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # A DCP model may replicate selected cheap cache groups. Select
            # ordinary or virtual-block addressing from this group's width.
            virtual_block_size = block_size * group_cp_size
            block_indices = positions // virtual_block_size
            block_offsets = positions % virtual_block_size
            block_numbers = tl.load(
                block_table_ptr + req_state_idx * block_table_stride + block_indices
            )
            is_sharded = group_cp_size == CP_SIZE
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            sharded_slot = block_numbers * block_size + local_offsets
            replicated_slot = block_numbers * block_size + block_offsets
            slot_ids = tl.where(
                is_sharded,
                tl.where(is_local, sharded_slot, PAD_ID),
                replicated_slot,
            )

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
''',
        '''    block_table_ptr = _load_ptr(block_table_ptrs + group_id, tl.int32)
    block_table_stride = tl.load(block_table_strides + group_id)
    kv_block_size = tl.load(block_sizes + group_id)
    kernel_block_size = tl.load(kernel_block_sizes + group_id)
    group_cp_size = tl.load(kv_cache_cp_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        if CP_SIZE == 1:
            local_positions = positions
            is_local = True
        else:
            # DCP interleave is expressed in logical KV-block coordinates,
            # independently for sharded and replicated cache groups.
            virtual_block_size = kv_block_size * group_cp_size
            virtual_block_indices = positions // virtual_block_size
            virtual_block_offsets = positions % virtual_block_size
            is_sharded = group_cp_size == CP_SIZE
            owned_by_rank = (
                virtual_block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            )
            rounds = virtual_block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = virtual_block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            sharded_positions = virtual_block_indices * kv_block_size + local_offsets
            replicated_positions = (
                virtual_block_indices * kv_block_size + virtual_block_offsets
            )
            local_positions = tl.where(
                is_sharded, sharded_positions, replicated_positions
            )
            is_local = tl.where(is_sharded, owned_by_rank, True)

        block_indices = local_positions // kernel_block_size
        block_offsets = local_positions % kernel_block_size
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices,
            mask=is_local,
            other=0,
        )
        slot_ids = block_numbers * kernel_block_size + block_offsets
        if CP_SIZE != 1:
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)
''',
        "#51031 rate-aware logical/kernel slot geometry",
    )


def patch_mla_group_contract(root: Path) -> None:
    path = root / "v1/kv_cache_interface.py"
    replace_once(
        path,
        '''        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
''',
        '''        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        non_causal_mtd_set = {
            spec.non_causal_multi_token_decode for spec in specs
        }
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
            and len(non_causal_mtd_set) == 1
''',
        "#54277 cache group capability agreement",
    )
    replace_once(
        path,
        '''            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, and KV block "
            "stride indexing."
''',
        '''            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, KV block "
            "stride indexing, and non-causal multi-token decode capability."
''',
        "#54277 cache group assertion message",
    )
    replace_once(
        path,
        '''            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
''',
        '''            non_causal_multi_token_decode=non_causal_mtd_set.pop(),
''',
        "#54277 cache group capability merge",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1])
    patch_slot_mapping(root)
    patch_mla_group_contract(root)
    print("Applied vLLM post-0.28 DCP correctness backports")


if __name__ == "__main__":
    main()
