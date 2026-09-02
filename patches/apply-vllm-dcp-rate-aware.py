#!/usr/bin/env python3
"""Replicate cheap DeepSeek-V4 cache families while DCP-sharding C4.

DeepSeek-V4's fixed SWA cache and 128:1 compressed cache are inexpensive to
replicate. Keeping those families local removes the DCP query/output exchange
from C1 and C128 layers; the dominant C4 cache and its indexer remain sharded.
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


def patch_cache_specs(root: Path) -> None:
    path = root / "v1/kv_cache_interface.py"
    replace_once(
        path,
        '''    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)
''',
        '''    def max_num_blocks_per_req(self, vllm_config: VllmConfig, max_len: int) -> int:
        kv_shard_count = get_kv_cache_dcp_world_size(
            self,
            vllm_config.parallel_config.decode_context_parallel_size,
        )
        return cdiv(max_len, self.block_size * kv_shard_count)
''',
        "per-spec block-table width",
    )
    replace_once(
        path,
        '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
''',
        '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = get_kv_cache_dcp_world_size(
            self,
            vllm_config.parallel_config.decode_context_parallel_size,
        )
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes
''',
        "per-spec full-attention sizing",
    )
    replace_once(
        path,
        '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
            dcp_world_size=(
                vllm_config.parallel_config.decode_context_parallel_size
            ),
        )
        return max_blocks * self.page_size_bytes
''',
        '''    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
            dcp_world_size=get_kv_cache_dcp_world_size(
                self,
                vllm_config.parallel_config.decode_context_parallel_size,
            ),
        )
        return max_blocks * self.page_size_bytes
''',
        "per-spec sliding-window sizing",
    )
    replace_once(
        path,
        '''    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
''',
        '''    def max_memory_usage_pages(self, vllm_config: VllmConfig) -> int:
        return max(
            cdiv(spec.max_memory_usage_bytes(vllm_config), spec.page_size_bytes)
            for spec in self.kv_cache_specs.values()
        )


def get_kv_cache_dcp_world_size(
    kv_cache_spec: KVCacheSpec,
    configured_dcp_world_size: int,
) -> int:
    """Return the physical KV ownership width for one cache family.

    Attention can still run in a DCP-configured model while a cheap family is
    replicated.  DeepSeek-V4 C128 records cost only 1/128 of a dense cache and
    SWA is bounded by its fixed window, so both stay local.  C4 main/indexer
    caches retain the configured DCP width.
    """
    if configured_dcp_world_size <= 1:
        return 1
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        widths = {
            get_kv_cache_dcp_world_size(spec, configured_dcp_world_size)
            for spec in kv_cache_spec.kv_cache_specs.values()
        }
        if len(widths) != 1:
            raise ValueError(
                "A packed KV cache group cannot mix replicated and DCP-sharded "
                f"layers; effective widths are {sorted(widths)}."
            )
        return next(iter(widths))
    if isinstance(kv_cache_spec, MambaSpec):
        return 1
    if (
        isinstance(kv_cache_spec, SlidingWindowMLASpec)
        and kv_cache_spec.model_version == "deepseek_v4"
    ):
        return 1
    if (
        isinstance(kv_cache_spec, MLAAttentionSpec)
        and kv_cache_spec.model_version == "deepseek_v4"
        and kv_cache_spec.compress_ratio == 128
    ):
        return 1
    if isinstance(kv_cache_spec, AttentionSpec):
        return configured_dcp_world_size
    return 1


def get_kv_cache_spec_kind(kv_cache_spec: KVCacheSpec) -> KVCacheSpecKind:
''',
        "effective cache DCP helper",
    )


def patch_kv_grouping(root: Path) -> None:
    path = root / "v1/core/kv_cache_utils.py"
    replace_once(
        path,
        '''    UniformTypeKVCacheSpecs,
    replace_as,
)''',
        '''    UniformTypeKVCacheSpecs,
    get_kv_cache_dcp_world_size,
    replace_as,
)''',
        "cache DCP helper import",
    )
    replace_once(
        path,
        '''    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in grouped_specs[1:]
        for spec in group.kv_cache_specs.values()
    )
''',
        '''    # A packed group has one block table and therefore one ownership
    # topology. Under DCP2, split the original mixed full-MLA family into
    # sharded C4/indexer and replicated C128 families before optimizing tuple
    # widths. DCP1 retains the established one-Spark grouping unchanged.
    configured_dcp = vllm_config.parallel_config.decode_context_parallel_size
    if configured_dcp > 1:
        ownership_groups: list[UniformTypeKVCacheSpecs] = []
        for grouped_spec in grouped_specs:
            buckets: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
            for layer_name, layer_spec in grouped_spec.kv_cache_specs.items():
                width = get_kv_cache_dcp_world_size(layer_spec, configured_dcp)
                buckets[width][layer_name] = layer_spec
            for width in sorted(buckets, reverse=True):
                split = UniformTypeKVCacheSpecs.from_specs(buckets[width])
                assert split is not None
                ownership_groups.append(split)
        grouped_specs = ownership_groups

    assert all(
        isinstance(spec, (MLAAttentionSpec, SlidingWindowMLASpec))
        for group in grouped_specs
        for spec in group.kv_cache_specs.values()
    )
''',
        "split packed groups by DCP ownership",
    )
    replace_once(
        path,
        '''    if len(groups) <= 1:
        bs = cache_config.block_size * dcp
        return bs, bs

    group_block_sizes = [
        g.kv_cache_spec.block_size * dcp
        if isinstance(g.kv_cache_spec, AttentionSpec)
        else g.kv_cache_spec.block_size
        for g in groups
    ]
''',
        '''    if len(groups) <= 1:
        spec = groups[0].kv_cache_spec
        bs = spec.block_size * get_kv_cache_dcp_world_size(spec, dcp)
        return bs, bs

    group_block_sizes = [
        g.kv_cache_spec.block_size
        * get_kv_cache_dcp_world_size(g.kv_cache_spec, dcp)
        for g in groups
    ]
''',
        "rate-aware scheduler/hash block sizes",
    )


def patch_coordinator(root: Path) -> None:
    path = root / "v1/core/kv_cache_coordinator.py"
    replace_once(
        path,
        '''    SlidingWindowSpec,
)''',
        '''    SlidingWindowSpec,
    get_kv_cache_dcp_world_size,
)''',
        "coordinator cache DCP helper import",
    )
    replace_once(
        path,
        '''                dcp_world_size=dcp_world_size,
                pcp_world_size=pcp_world_size,
''',
        '''                dcp_world_size=get_kv_cache_dcp_world_size(
                    kv_cache_group.kv_cache_spec, dcp_world_size
                ),
                pcp_world_size=pcp_world_size,
''',
        "per-manager DCP ownership",
    )
    replace_once(
        path,
        '''        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size > 1:
            self.block_size *= dcp_world_size
''',
        '''        self.dcp_world_size = get_kv_cache_dcp_world_size(
            self.kv_cache_spec, dcp_world_size
        )
        self.pcp_world_size = pcp_world_size
        if self.dcp_world_size > 1:
            self.block_size *= self.dcp_world_size
''',
        "unitary effective DCP ownership",
    )
    replace_once(
        path,
        '''                    dcp_world_size=(
                        self.dcp_world_size
                        if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec))
                        else 1
                    ),
''',
        '''                    dcp_world_size=get_kv_cache_dcp_world_size(
                        spec, self.dcp_world_size
                    ),
''',
        "rate-aware cache-hit ownership",
    )


def patch_block_tables(root: Path) -> None:
    path = root / "v1/worker/gpu/block_table.py"
    replace_once(
        path,
        '''        kernel_block_sizes: list[int],
        cp_size: int = 1,
''',
        '''        kernel_block_sizes: list[int],
        kv_cache_cp_sizes: list[int] | None = None,
        cp_size: int = 1,
''',
        "per-group CP constructor API",
    )
    replace_once(
        path,
        '''        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
''',
        '''        self.cp_size = cp_size
        self.cp_rank = cp_rank
        self.cp_interleave = cp_interleave

        self.num_kv_cache_groups = len(self.block_sizes)
        self.kv_cache_cp_sizes = (
            list(kv_cache_cp_sizes)
            if kv_cache_cp_sizes is not None
            else [cp_size] * self.num_kv_cache_groups
        )
        assert len(max_num_blocks_per_group) == self.num_kv_cache_groups
        assert len(self.kv_cache_cp_sizes) == self.num_kv_cache_groups
        assert all(size in (1, cp_size) for size in self.kv_cache_cp_sizes)
''',
        "store per-group CP widths",
    )
    replace_once(
        path,
        '''        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)
''',
        '''        self.block_sizes_tensor = torch.tensor(
            self.kernel_block_sizes, dtype=torch.int32, device=self.device
        )
        self.kv_cache_cp_sizes_tensor = torch.tensor(
            self.kv_cache_cp_sizes, dtype=torch.int32, device=self.device
        )
        self.input_block_table_ptrs = self._make_ptr_tensor(self.input_block_tables)
''',
        "per-group CP width tensor",
    )
    replace_once(
        path,
        '''            self.block_table_strides,
            self.block_sizes_tensor,
            slot_mappings,
''',
        '''            self.block_table_strides,
            self.block_sizes_tensor,
            self.kv_cache_cp_sizes_tensor,
            slot_mappings,
''',
        "slot kernel CP width input",
    )
    replace_once(
        path,
        '''    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
''',
        '''    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    kv_cache_cp_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
''',
        "slot kernel CP width parameter",
    )
    replace_once(
        path,
        '''    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
''',
        '''    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)
    group_cp_size = tl.load(kv_cache_cp_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
''',
        "load group CP width",
    )
    replace_once(
        path,
        '''        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)
''',
        '''        if CP_SIZE == 1:
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
''',
        "rate-aware slot address transform",
    )


def patch_model_runner(root: Path) -> None:
    path = root / "v1/worker/gpu/model_runner.py"
    replace_once(
        path,
        '''from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
''',
        '''from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MambaSpec,
    get_kv_cache_dcp_world_size,
)
''',
        "model-runner cache DCP helper import",
    )
    replace_once(
        path,
        '''        block_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            # When using DCP, each request's KV cache is sharded among different ranks.
            # As a result, one block on the current rank covers `block_size * cp_size`
            # tokens in the full, global (unsharded) sequence.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * self.dcp_size
            )
''',
        '''        block_sizes = []
        kv_cache_cp_sizes = []
        max_num_blocks_per_group = []
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            spec = kv_cache_group.kv_cache_spec
            block_sizes.append(spec.block_size)
            group_cp_size = get_kv_cache_dcp_world_size(spec, self.dcp_size)
            kv_cache_cp_sizes.append(group_cp_size)
            # One local block covers either an ordinary replicated page or a
            # virtual DCP page, depending on this cache family's ownership.
            max_num_blocks = cdiv(
                block_table_max_model_len, spec.block_size * group_cp_size
            )
''',
        "per-group model-runner cache widths",
    )
    replace_once(
        path,
        '''            kernel_block_sizes=self.kernel_block_sizes,
            cp_size=self.dcp_size,
''',
        '''            kernel_block_sizes=self.kernel_block_sizes,
            kv_cache_cp_sizes=kv_cache_cp_sizes,
            cp_size=self.dcp_size,
''',
        "pass per-group cache widths",
    )


def patch_dflash(root: Path) -> None:
    path = root / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    replace_once(
        path,
        '''                self.block_tables.cp_size,
                self.block_tables.cp_rank,
''',
        '''                self.block_tables.kv_cache_cp_sizes[gid],
                (
                    self.block_tables.cp_rank
                    if self.block_tables.kv_cache_cp_sizes[gid] > 1
                    else 0
                ),
''',
        "DSpark per-group cache width and logical rank",
    )


def patch_metadata(root: Path) -> None:
    path = root / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        path,
        '''        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
''',
        '''        parallel_config = vllm_config.parallel_config
        configured_dcp = parallel_config.decode_context_parallel_size
        self.dcp_world_size = configured_dcp if self.compress_ratio == 4 else 1
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
''',
        "C128 replicated metadata",
    )

    path = root / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        path,
        '''        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
''',
        '''        parallel_config = self.vllm_config.parallel_config
        # The bounded DeepSeek-V4 SWA cache is replicated on every DCP rank.
        # C4 attention includes it only on rank 0 before the exact LSE merge.
        self.dcp_world_size = 1
        self.dcp_rank = 0
''',
        "replicated SWA metadata",
    )


def patch_attention(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        '''def _attention_sink_for_shard(attention) -> torch.Tensor:
    if getattr(attention, "dcp_manager", None) is not None:
        return attention._dcp_attn_sink
    return attention.attn_sink


def _use_b12x_compressed_mla_decode() -> bool:
''',
        '''def _attention_sink_for_shard(attention) -> torch.Tensor:
    if getattr(attention, "dcp_manager", None) is not None:
        return attention._dcp_attn_sink
    return attention.attn_sink


def _swa_lengths_for_shard(
    attention, lengths: torch.Tensor
) -> torch.Tensor:
    # Replicated SWA is one logical softmax input. Include it on rank 0 only;
    # C4 records remain disjoint across ranks and are merged with the normal LSE.
    if attention.dcp_manager is not None and attention.dcp_rank != 0:
        return attention._dcp_empty_swa_lens[: lengths.shape[0]]
    return lengths


def _use_b12x_compressed_mla_decode() -> bool:
''',
        "single-owner replicated SWA helper",
    )
    replace_once(
        path,
        '''        if self.dcp_world_size > 1:
            if self.cp_kv_cache_interleave_size != 1:
''',
        '''        if self.dcp_world_size > 1 and self.compress_ratio == 4:
            if self.cp_kv_cache_interleave_size != 1:
''',
        "C4-only DCP attention exchange",
    )
    replace_once(
        path,
        '''            self.register_buffer(
                "_dcp_attn_sink",
                torch.full(
                    (self._dcp_attention_heads,),
                    -float("inf"),
                    dtype=torch.float32,
                    device=self.attn_sink.device,
                ),
                persistent=False,
            )
''',
        '''            self.register_buffer(
                "_dcp_attn_sink",
                torch.full(
                    (self._dcp_attention_heads,),
                    -float("inf"),
                    dtype=torch.float32,
                    device=self.attn_sink.device,
                ),
                persistent=False,
            )
            self.register_buffer(
                "_dcp_empty_swa_lens",
                torch.zeros(
                    vllm_config.scheduler_config.max_num_batched_tokens,
                    dtype=torch.int32,
                    device=self.attn_sink.device,
                ),
                persistent=False,
            )
''',
        "empty replicated SWA lengths",
    )
    replace_once(
        path,
        '''        assert swa_indices is not None
        assert swa_lens is not None
        q = self._prepare_query(q, output)
''',
        '''        assert swa_indices is not None
        assert swa_lens is not None
        swa_lens = _swa_lengths_for_shard(self, swa_lens)
        q = self._prepare_query(q, output)
''',
        "decode single-owner SWA",
    )
    replace_once(
        path,
        '''            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
''',
        '''            swa_indices_chunk = swa_metadata.prefill_swa_indices[query_start:query_end]
            swa_lens_chunk = swa_metadata.prefill_swa_lens[query_start:query_end]
            swa_lens_chunk = _swa_lengths_for_shard(self, swa_lens_chunk)
            if extra_kv_paged is not None and extra_sparse_indices_chunk is None:
''',
        "prefill single-owner SWA",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dcp-rate-aware.py VLLM_ROOT")
    root = Path(sys.argv[1]).resolve()
    if not (root / "models/deepseek_v4/attention.py").is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")
    patch_cache_specs(root)
    patch_kv_grouping(root)
    patch_coordinator(root)
    patch_block_tables(root)
    patch_model_runner(root)
    patch_dflash(root)
    patch_metadata(root)
    patch_attention(root)


if __name__ == "__main__":
    main()
