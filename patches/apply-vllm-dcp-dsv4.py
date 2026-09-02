#!/usr/bin/env python3
"""Make DeepSeek-V4's compressed sparse-attention metadata DCP-aware."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def patch_dcp_rank_scalar_sync(root: Path) -> None:
    """Avoid a pageable host-to-device scalar copy on every DCP step."""
    path = root / "v1/attention/backends/utils.py"
    replace_once(
        path,
        '''    else:
        rank_offsets = torch.tensor(dcp_rank, dtype=torch.int32, device=seq_lens.device)
        seq_lens_tiled = seq_lens_i32
''',
        '''    else:
        # Keep the known rank as a scalar. Constructing a fresh CUDA tensor
        # from this Python integer performs a pageable host-to-device copy;
        # that copy synchronizes the current stream once per decode step.
        rank_offsets = dcp_rank
        seq_lens_tiled = seq_lens_i32
''',
        "DCP rank scalar stream-sync removal",
    )


def patch_draft_dcp_metadata(root: Path) -> None:
    """Populate the draft model's persistent per-rank sequence lengths.

    The target runner prepares this tensor before its attention metadata is
    built, but DFlash/DSpark rebuild their own metadata from separate input
    buffers.  Without doing the same preparation here the draft's DSV4 SWA
    attention receives ``None`` even though its cache is DCP-sharded.
    """
    path = root / "v1/worker/gpu/spec_decode/speculator.py"
    replace_once(
        path,
        '''from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
''',
        '''from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
''',
        "draft DCP sequence helper import",
    )
    replace_once(
        path,
        '''        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)
        attn_metadata = build_attn_metadata(
''',
        '''        draft_seq_lens_cpu_upper_bound[:num_reqs].clamp_(max=self.max_model_len)

        # The draft query-preparation kernel writes its own absolute seq_lens,
        # independently of the target runner. Derive the corresponding local
        # lengths into this speculator's persistent buffer before building
        # metadata. This stays CUDA-graph safe and also clears padded rows.
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                self.input_buffers.dcp_local_seq_lens,
                self.input_buffers.seq_lens,
                num_reqs,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
            )
            dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[
                :num_reqs_padded
            ]

        attn_metadata = build_attn_metadata(
''',
        "prepare draft DCP local lengths",
    )
    replace_once(
        path,
        '''            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
        )
''',
        '''            causal=causal,
            seq_lens_cpu_upper_bound=draft_seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=dcp_local_seq_lens,
        )
''',
        "pass draft DCP local lengths to metadata builders",
    )

    # DFlash/DSpark prepare both context and parallel-query KV slots in their
    # own Triton kernel rather than calling BlockTables.compute_slot_mappings().
    # Apply the same virtual-block ownership transform here; otherwise every
    # DCP rank writes every global position into a different, incorrect local
    # slot and the draft cache is corrupted after the first proposal step.
    path = root / "v1/worker/gpu/spec_decode/dflash/speculator.py"
    replace_once(
        path,
        '''                self.max_num_tokens,
                self.max_model_len,
                self.sample_from_anchor,
''',
        '''                self.max_num_tokens,
                self.max_model_len,
                self.block_tables.cp_size,
                self.block_tables.cp_rank,
                self.block_tables.cp_interleave,
                self.sample_from_anchor,
''',
        "pass draft DCP slot topology",
    )
    replace_once(
        path,
        '''    max_num_tokens,
    max_model_len,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
''',
        '''    max_num_tokens,
    max_model_len,
    cp_rank,
    SAMPLE_FROM_ANCHOR: tl.constexpr,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_SLOT_ID: tl.constexpr,
''',
        "draft DCP slot kernel topology",
    )
    replace_once(
        path,
        '''    ctx_block_num = ctx_pos // block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction; rejected suffix rows are invalid context as well.
    # Neither kind of row may write draft KV into physical block 0.
    ctx_resident = is_valid_ctx & (ctx_block_id != 0)
    ctx_slot = tl.where(
        ctx_resident,
        ctx_block_id * block_size + (ctx_pos % block_size),
        PAD_SLOT_ID,
    )
''',
        '''    ctx_virtual_block_size = block_size * CP_SIZE
    ctx_block_num = ctx_pos // ctx_virtual_block_size
    ctx_block_num = tl.minimum(ctx_block_num, block_table_stride - 1)
    ctx_virtual_offset = ctx_pos % ctx_virtual_block_size
    ctx_is_local = (
        (ctx_virtual_offset // CP_INTERLEAVE) % CP_SIZE == cp_rank
    )
    ctx_round = ctx_virtual_offset // (CP_INTERLEAVE * CP_SIZE)
    ctx_remainder = ctx_virtual_offset % CP_INTERLEAVE
    ctx_local_offset = ctx_round * CP_INTERLEAVE + ctx_remainder
    ctx_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + ctx_block_num,
        mask=is_valid_ctx,
        other=0,
    ).to(tl.int64)
    # Block 0 is the null block. Old sliding-window context positions can map
    # to it after eviction; rejected suffix rows are invalid context as well.
    # A peer-owned global position must also remain a query-only row here.
    ctx_resident = is_valid_ctx & ctx_is_local & (ctx_block_id != 0)
    ctx_slot = tl.where(
        ctx_resident,
        ctx_block_id * block_size + ctx_local_offset,
        PAD_SLOT_ID,
    )
''',
        "map draft context slots to their DCP owner",
    )
    replace_once(
        path,
        '''    q_block_num = query_pos // block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & (q_block_id != 0)
    q_slot = tl.where(
        q_resident,
        q_block_id * block_size + (query_pos % block_size),
        PAD_SLOT_ID,
    )
''',
        '''    q_virtual_block_size = block_size * CP_SIZE
    q_block_num = query_pos // q_virtual_block_size
    q_block_num = tl.minimum(q_block_num, block_table_stride - 1)
    q_virtual_offset = query_pos % q_virtual_block_size
    q_is_local = (q_virtual_offset // CP_INTERLEAVE) % CP_SIZE == cp_rank
    q_round = q_virtual_offset // (CP_INTERLEAVE * CP_SIZE)
    q_remainder = q_virtual_offset % CP_INTERLEAVE
    q_local_offset = q_round * CP_INTERLEAVE + q_remainder
    q_block_id = tl.load(
        block_table_ptr + req_idx * block_table_stride + q_block_num,
        mask=is_query,
        other=0,
    ).to(tl.int64)
    # A null block is never a writable cache slot. This can occur when a
    # sliding-window block table contains evicted/global padding entries.
    q_resident = is_query & q_is_local & (q_block_id != 0)
    q_slot = tl.where(
        q_resident,
        q_block_id * block_size + q_local_offset,
        PAD_SLOT_ID,
    )
''',
        "map draft query slots to their DCP owner",
    )
    replace_once(
        path,
        '''    max_num_tokens: int,
    max_model_len: int,
    sample_from_anchor: bool = False,
''',
        '''    max_num_tokens: int,
    max_model_len: int,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
    sample_from_anchor: bool = False,
''',
        "draft DCP slot public API",
    )
    replace_once(
        path,
        '''        max_num_tokens,
        max_model_len,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
''',
        '''        max_num_tokens,
        max_model_len,
        cp_rank,
        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        CP_SIZE=cp_size,
        CP_INTERLEAVE=cp_interleave,
        PAD_SLOT_ID=PAD_SLOT_ID,
''',
        "launch draft DCP slot kernel",
    )

    # Regular eager/profile dummy runs do not use the CUDA-graph helper that
    # already populates this field. B12x deliberately executes one such real
    # attention pass to JIT live cache strides, so prepare it in the common
    # model-runner dummy path after any synthetic context has changed seq_lens.
    path = root / "v1/worker/gpu/model_runner.py"
    replace_once(
        path,
        '''            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

        attn_metadata = None
''',
        '''            else:
                assert batch_desc.cg_mode != CUDAGraphMode.FULL, (
                    "Attention metadata must be prepared for dummy runs when using "
                    "FULL cudagraph mode."
                )
                block_tables = None
                slot_mappings = None

            # Eager/profile dummy runs bypass prepare_inputs_to_capture(),
            # which normally fills this buffer for DCP. Derive it after
            # set_dummy_context() so attention sees the final synthetic lengths.
            if self.use_dcp:
                prepare_dcp_local_seq_lens(
                    self.input_buffers.dcp_local_seq_lens,
                    input_batch.seq_lens,
                    input_batch.num_reqs,
                    self.dcp_size,
                    self.dcp_rank,
                    self.cp_interleave,
                )
                input_batch.dcp_local_seq_lens = (
                    self.input_buffers.dcp_local_seq_lens[
                        : input_batch.num_reqs_after_padding
                    ]
                )

        attn_metadata = None
''',
        "prepare eager/profile dummy DCP local lengths",
    )

    # DFlash/DSpark use a separate graph manager and therefore do not reach
    # vLLM's general prepare_inputs_to_capture() DCP handling either.
    path = root / "v1/worker/gpu/spec_decode/dflash/cudagraph.py"
    replace_once(
        path,
        '''from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cudagraph_utils import (
''',
        '''from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
''',
        "draft capture DCP sequence helper import",
    )
    replace_once(
        path,
        '''    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        attn_metadata = build_attn_metadata(
''',
        '''    attn_metadata = None
    if not skip_attn:
        dcp_local_seq_lens = None
        if block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_batch.seq_lens,
                num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens[:num_reqs]
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        attn_metadata = build_attn_metadata(
''',
        "prepare draft capture DCP local lengths",
    )
    replace_once(
        path,
        '''            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
''',
        '''            kv_cache_config=kv_cache_config,
            dcp_local_seq_lens=dcp_local_seq_lens,
            for_cudagraph_capture=True,
            causal=causal,
''',
        "pass draft capture DCP local lengths",
    )


def patch_compressed_slot_mapping(root: Path) -> None:
    path = root / "v1/attention/backends/mla/compressor_utils.py"
    replace_once(
        path,
        '''    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
''',
        '''    block_table_stride,
    block_size,
    dcp_world_size,
    dcp_rank,
    cp_kv_cache_interleave_size,
    COMPRESS_RATIO: tl.constexpr,
''',
        "compressed slot DCP kernel parameters",
    )
    replace_once(
        path,
        '''        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        pos_after_compress = pos // COMPRESS_RATIO

        block_ids = pos_after_compress // block_size
        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + block_ids,
            mask=mask & is_valid,
        )
        slot_ids = block_numbers * block_size + pos_after_compress % block_size

        # NOTE
        slot_ids = tl.where(is_valid, slot_ids, PAD_ID)
''',
        '''        is_valid = (pos + 1) % COMPRESS_RATIO == 0
        compressed_pos = pos // COMPRESS_RATIO

        # Compression is semantic: first select the completed C4/C128 record,
        # then shard that compressed record across DCP ranks. One physical page
        # per rank therefore spans ``block_size * dcp_world_size`` compressed
        # positions in the scheduler's global coordinate space.
        virtual_block_size = block_size * dcp_world_size
        virtual_block = compressed_pos // virtual_block_size
        virtual_offset = compressed_pos % virtual_block_size
        is_local = (
            virtual_offset // cp_kv_cache_interleave_size
        ) % dcp_world_size == dcp_rank
        local_offset = (
            virtual_offset
            // (dcp_world_size * cp_kv_cache_interleave_size)
        ) * cp_kv_cache_interleave_size + (
            virtual_offset % cp_kv_cache_interleave_size
        )

        block_numbers = tl.load(
            block_table_ptr + batch_idx * block_table_stride + virtual_block,
            mask=mask & is_valid & is_local,
        )
        slot_ids = block_numbers * block_size + local_offset
        slot_ids = tl.where(is_valid & is_local, slot_ids, PAD_ID)
''',
        "compress before DCP slot sharding",
    )
    replace_once(
        path,
        '''    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
''',
        '''    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
''',
        "compressed slot DCP API",
    )
    replace_once(
        path,
        '''        block_table.stride(0),
        block_size,
        compress_ratio,
        PAD_ID=-1,
''',
        '''        block_table.stride(0),
        block_size,
        dcp_world_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
        compress_ratio,
        PAD_ID=-1,
''',
        "compressed slot DCP launch",
    )


def patch_indexer_metadata(root: Path) -> None:
    path = root / "v1/attention/backends/mla/indexer.py"
    replace_once(
        path,
        '''        if self.dcp_world_size > 1 and self.compress_ratio > 1:
            raise NotImplementedError(
                "DCP is not supported with sparse indexer KV compression "
                f"(compress_ratio={self.compress_ratio})."
            )

''',
        '''        # Compressed DeepSeek-V4 cache records are sharded after
        # compression. The slot mapper and per-rank sequence bounds below use
        # that same compressed coordinate space under DCP.

''',
        "remove compressed-indexer DCP guard",
    )
    replace_once(
        path,
        '''                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
''',
        '''                self.kv_cache_spec.storage_block_size,
                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
''',
        "indexer compressed slot DCP arguments",
    )
    replace_once(
        path,
        '''            # DCP: localize the now-expanded per-token global bounds to this
            # rank's owned KV. Done here (after expansion) so each token's global
            # causal length is localized individually; see the comment above.
            if dcp_local_seq_lens is not None:
                seq_lens = self._dcp_localize_decode_seq_lens(
                    seq_lens, num_decodes, seq_lens_is_buffer_view
                )

            # For DeepseekV4 (compress_ratio > 1), the indexer KV cache stores
            # compressed tokens. Convert uncompressed seq_lens to compressed.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]
''',
        '''            # DeepSeek-V4 shards completed compressed records, not the
            # original token stream. Convert each expanded global causal bound
            # to C4/C128 space before applying the DCP ownership transform.
            if self.compress_ratio > 1:
                if seq_lens_is_buffer_view:
                    seq_lens //= self.compress_ratio
                else:
                    # Copy to avoid mutating shared state; keeps CG address stable.
                    self.expanded_seq_lens_buffer[:num_decodes] = (
                        seq_lens // self.compress_ratio
                    )
                    self.expanded_seq_lens_buffer[num_decodes:num_decode_tokens] = 0
                    seq_lens = self.expanded_seq_lens_buffer[:num_decode_tokens]

            # Localize the now-expanded compressed bounds independently for
            # every target/draft row. Doing this before compression would give
            # the wrong owner at compression boundaries.
            if dcp_local_seq_lens is not None:
                seq_lens = self._dcp_localize_decode_seq_lens(
                    seq_lens, num_decodes, seq_lens_is_buffer_view
                )
''',
        "compress decode bounds before DCP localization",
    )


def patch_c128_metadata(root: Path) -> None:
    path = root / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        path,
        '''from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
''',
        '''from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.platforms.interface import DeviceCapability
''',
        "C128 DCP group import",
    )
    replace_once(
        path,
        '''        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
''',
        '''        # The B12x DCP path supports the same variable-length DSpark
        # decode rows as the non-DCP sparse path. Keep C128's split aligned with
        # SWA (1 + speculative tokens); otherwise DCP silently resets this to 1
        # and the two cache groups classify the same draft rows differently.
        self._init_reorder_batch_threshold(
            1,
            supports_spec_as_decode=True,
            supports_dcp_with_varlen=True,
        )
''',
        "retain speculative decode threshold under DCP",
    )
    replace_once(
        path,
        '''        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
''',
        '''        assert hasattr(self.kv_cache_spec, "compress_ratio")
        self.compress_ratio = self.kv_cache_spec.compress_ratio
        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4 compressed DCP currently requires "
                "cp_kv_cache_interleave_size=1."
            )

        # Pre-allocate compressed slot mapping buffer for CUDA graph address
''',
        "C128 builder DCP constants",
    )
    replace_once(
        path,
        '''            c128a_max_compressed = cdiv(
                self.model_config.max_model_len, self.compress_ratio
            )
''',
        '''            c128a_max_compressed = cdiv(
                cdiv(self.model_config.max_model_len, self.compress_ratio),
                self.dcp_world_size,
            )
''',
        "C128 per-rank buffer width",
    )
    replace_once(
        path,
        '''                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
            )
''',
        '''                self.compress_ratio,
                out=self.compressed_slot_mapping_buffer,
                dcp_world_size=self.dcp_world_size,
                dcp_rank=self.dcp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
''',
        "C128 compressed slot DCP arguments",
    )
    replace_once(
        path,
        '''            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
        )
''',
        '''            self.c128a_prefill_buffer,
            max_compressed_tokens=self.c128a_max_compressed,
            dcp_world_size=self.dcp_world_size,
            dcp_rank=self.dcp_rank,
            cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
        )
''',
        "C128 metadata DCP arguments",
    )
    replace_once(
        path,
        '''    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
''',
        '''    prefill_buffer: torch.Tensor,
    max_compressed_tokens: int = 8192,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
''',
        "C128 metadata public DCP API",
    )
    replace_once(
        path,
        '''        block_size,
        slot_mapping,
        BLOCK_SIZE=1024,
''',
        '''        block_size,
        slot_mapping,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_INTERLEAVE=cp_kv_cache_interleave_size,
        BLOCK_SIZE=1024,
''',
        "C128 metadata DCP launch",
    )
    replace_once(
        path,
        '''    slot_mapping_ptr,
    BLOCK_SIZE: tl.constexpr,
):
''',
        '''    slot_mapping_ptr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
''',
        "C128 kernel DCP constants",
    )
    replace_once(
        path,
        '''    num_compressed = (position + 1) // compress_ratio
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
''',
        '''    global_num_compressed = (position + 1) // compress_ratio
    dcp_cycle = DCP_WORLD_SIZE * CP_INTERLEAVE
    full_cycles = global_num_compressed // dcp_cycle
    remainder = global_num_compressed % dcp_cycle
    rank_start = DCP_RANK * CP_INTERLEAVE
    rank_remainder = tl.minimum(
        tl.maximum(remainder - rank_start, 0), CP_INTERLEAVE
    )
    num_compressed = full_cycles * CP_INTERLEAVE + rank_remainder
    num_compressed = tl.minimum(num_compressed, max_compressed_tokens)
''',
        "C128 local completed-record count",
    )
    replace_once(
        path,
        '''        is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
''',
        '''        is_valid_token = tl.load(slot_mapping_ptr + token_idx) >= 0
        if DCP_WORLD_SIZE > 1:
            # Every launched row is a real query on every DCP rank. The common
            # slot mapping is owner-specific and therefore cannot be used as a
            # query-validity predicate after KV sharding.
            is_valid_token = True
''',
        "C128 DCP query validity",
    )


def patch_swa_metadata(root: Path) -> None:
    path = root / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        path,
        '''from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
''',
        '''from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
''',
        "SWA DCP group import",
    )
    replace_once(
        path,
        '''    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs]
''',
        '''    block_size: int
    seq_lens: torch.Tensor | None = None  # [num_seqs], global
    dcp_local_seq_lens: torch.Tensor | None = None  # [num_seqs], this rank
''',
        "SWA local sequence lengths",
    )
    replace_once(
        path,
        '''        self.block_size = mla_spec.block_size
        self.max_model_len = self.vllm_config.model_config.max_model_len
''',
        '''        self.block_size = mla_spec.block_size
        parallel_config = self.vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.dcp_rank = (
            get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0
        )
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        if self.dcp_world_size > 1 and self.cp_kv_cache_interleave_size != 1:
            raise NotImplementedError(
                "DeepSeek-V4 SWA DCP currently requires "
                "cp_kv_cache_interleave_size=1."
            )
        self.max_model_len = self.vllm_config.model_config.max_model_len
''',
        "SWA builder DCP constants",
    )
    replace_once(
        path,
        '''        slot_mapping = common_attn_metadata.slot_mapping

        # Split into decode and prefill portions using configurable threshold
''',
        '''        slot_mapping = common_attn_metadata.slot_mapping
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        # Split into decode and prefill portions using configurable threshold
''',
        "SWA local lengths input",
    )
    replace_once(
        path,
        '''        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        is_valid_token.copy_(slot_mapping >= 0)
''',
        '''        is_valid_token = self.is_valid_token[: slot_mapping.shape[0]]
        if self.dcp_world_size > 1:
            # A real query row exists on every DCP rank even when its new KV
            # token belongs to a peer. Ownership is encoded only by slot_mapping.
            is_valid_token.fill_(False)
            is_valid_token[: common_attn_metadata.num_actual_tokens] = True
        else:
            is_valid_token.copy_(slot_mapping >= 0)
''',
        "separate DCP ownership from query validity",
    )
    replace_once(
        path,
        '''                    self.block_size,
                    token_offset=0,
                    TRITON_BLOCK_SIZE=1024,
''',
        '''                    self.block_size,
                    token_offset=0,
                    DCP_WORLD_SIZE=self.dcp_world_size,
                    DCP_RANK=self.dcp_rank,
                    CP_INTERLEAVE=self.cp_kv_cache_interleave_size,
                    TRITON_BLOCK_SIZE=1024,
''',
        "noncausal SWA DCP launch",
    )
    # Causal decode, prefill, and draft-update calls share these tail shapes.
    source = path.read_text()
    old = '''                    HAS_MM_PREFIX=False,
                    TRITON_BLOCK_SIZE=1024,
'''
    new = '''                    HAS_MM_PREFIX=False,
                    DCP_WORLD_SIZE=self.dcp_world_size,
                    DCP_RANK=self.dcp_rank,
                    CP_INTERLEAVE=self.cp_kv_cache_interleave_size,
                    TRITON_BLOCK_SIZE=1024,
'''
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: causal decode DCP launch anchor mismatch")
    path.write_text(source.replace(old, new, 1))
    replace_once(
        path,
        '''                HAS_MM_PREFIX=mm_query_ranges is not None,
                TRITON_BLOCK_SIZE=1024,
''',
        '''                HAS_MM_PREFIX=mm_query_ranges is not None,
                DCP_WORLD_SIZE=self.dcp_world_size,
                DCP_RANK=self.dcp_rank,
                CP_INTERLEAVE=self.cp_kv_cache_interleave_size,
                TRITON_BLOCK_SIZE=1024,
''',
        "prefill SWA DCP launch",
    )
    replace_once(
        path,
        '''            HAS_MM_PREFIX=False,
            TRITON_BLOCK_SIZE=1024,
''',
        '''            HAS_MM_PREFIX=False,
            DCP_WORLD_SIZE=self.dcp_world_size,
            DCP_RANK=self.dcp_rank,
            CP_INTERLEAVE=self.cp_kv_cache_interleave_size,
            TRITON_BLOCK_SIZE=1024,
''',
        "draft-update SWA DCP launch",
    )
    replace_once(
        path,
        '''            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
''',
        '''            seq_lens=seq_lens,
            dcp_local_seq_lens=dcp_local_seq_lens,
            query_start_loc=query_start_loc,
''',
        "return SWA local sequence lengths",
    )
    replace_once(
        path,
        '''@triton.jit(do_not_specialize=["token_offset"])
def _compute_swa_indices_and_lens_kernel(
''',
        '''@triton.jit
def _dcp_local_prefix_len(
    global_end,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
):
    cycle = DCP_WORLD_SIZE * CP_INTERLEAVE
    full_cycles = global_end // cycle
    remainder = global_end % cycle
    rank_start = DCP_RANK * CP_INTERLEAVE
    rank_remainder = tl.minimum(
        tl.maximum(remainder - rank_start, 0), CP_INTERLEAVE
    )
    return full_cycles * CP_INTERLEAVE + rank_remainder


@triton.jit(do_not_specialize=["token_offset"])
def _compute_swa_indices_and_lens_kernel(
''',
        "SWA DCP prefix helper",
    )
    replace_once(
        path,
        '''    token_offset,
    HAS_MM_PREFIX: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
''',
        '''    token_offset,
    HAS_MM_PREFIX: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
''',
        "causal SWA DCP constants",
    )
    replace_once(
        path,
        '''    swa_len = end_pos - start_pos
    tl.device_assert(swa_len <= index_width, "SWA index width is too small")
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
''',
        '''    local_start = _dcp_local_prefix_len(
        start_pos, DCP_WORLD_SIZE, DCP_RANK, CP_INTERLEAVE
    )
    local_end = _dcp_local_prefix_len(
        end_pos, DCP_WORLD_SIZE, DCP_RANK, CP_INTERLEAVE
    )
    swa_len = local_end - local_start
    tl.device_assert(swa_len <= index_width, "SWA index width is too small")
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        local_pos = local_start + offset
        block_indices = local_pos // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=offset < swa_len,
        )
        block_offsets = local_pos % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
''',
        "causal SWA local slot mapping",
    )
    replace_once(
        path,
        '''    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
''',
        '''    block_size,
    token_offset,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
''',
        "noncausal SWA DCP constants",
    )
    replace_once(
        path,
        '''    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
''',
        '''    local_start = _dcp_local_prefix_len(
        start_pos, DCP_WORLD_SIZE, DCP_RANK, CP_INTERLEAVE
    )
    local_end = _dcp_local_prefix_len(
        end_pos, DCP_WORLD_SIZE, DCP_RANK, CP_INTERLEAVE
    )
    swa_len = local_end - local_start
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        local_pos = local_start + offset
        block_indices = local_pos // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=offset < swa_len,
        )
        block_offsets = local_pos % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
''',
        "noncausal SWA local slot mapping",
    )


def patch_c4_topk_mapping(root: Path) -> None:
    path = root / "models/deepseek_v4/common/ops/cache_utils.py"
    replace_once(
        path,
        '''    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
        '''    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
''',
        "C4 top-k DCP API",
    )
    replace_once(
        path,
        '''        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
''',
        '''        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
    if dcp_world_size > 1:
        # The exact global top-K contains records owned by every rank. Clear the
        # row before the kernel compacts only this rank's owned records.
        global_topk_indices.fill_(-1)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
''',
        "clear compacted C4 output",
    )
    replace_once(
        path,
        '''        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
''',
        '''        block_size,
        is_valid_token,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_INTERLEAVE=cp_kv_cache_interleave_size,
        TRITON_BLOCK_SIZE=triton.next_power_of_2(topk_indices.shape[-1]),
''',
        "C4 top-k DCP launch",
    )
    start = path.read_text()
    old = '''@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    # Graph replay may include padded token rows whose request mapping
    # and top-k buffer still contain stale values. Clamp the request row
    # before pointer arithmetic; validity then gates every table lookup.
    safe_req_idx = tl.where(is_valid_token, req_idx, 0)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        is_valid = (local_idx >= 0) & is_valid_token

        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr
            + safe_req_idx * block_table_stride
            + block_indices,
            mask=mask & is_valid,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    # Zero out length for padding tokens.
    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))
'''
    new = '''@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    token_valid = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    safe_req_idx = tl.where(token_valid, req_idx, 0)

    offset = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = offset < topk
    compressed_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    is_valid = mask & (compressed_idx >= 0) & token_valid

    if DCP_WORLD_SIZE > 1:
        # Indexer DCP merge returns exact global compressed-record ids. Retain
        # this rank's records, convert them to local compressed coordinates,
        # and compact them so attention's topk_length remains a prefix count.
        group = compressed_idx // CP_INTERLEAVE
        owner = group % DCP_WORLD_SIZE
        is_valid &= owner == DCP_RANK
        local_idx = (
            (group // DCP_WORLD_SIZE) * CP_INTERLEAVE
            + compressed_idx % CP_INTERLEAVE
        )
    else:
        local_idx = compressed_idx

    safe_local_idx = tl.maximum(local_idx, 0)
    block_indices = safe_local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + safe_req_idx * block_table_stride + block_indices,
        mask=is_valid,
        other=0,
    )
    block_offsets = safe_local_idx % block_size
    slot_ids = block_numbers * block_size + block_offsets
    slot_ids = tl.where(is_valid, slot_ids, -1)

    if DCP_WORLD_SIZE > 1:
        compact_offset = tl.cumsum(is_valid.to(tl.int32), axis=0) - 1
        tl.store(
            global_topk_indices_ptr
            + token_idx * global_topk_indices_stride
            + compact_offset,
            slot_ids,
            mask=is_valid,
        )
    else:
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )

    count = tl.sum(is_valid.to(tl.int32), axis=0)
    tl.store(topk_lens_ptr + token_idx, tl.where(token_valid, count, 0))
'''
    if start.count(old) != 1:
        raise RuntimeError(f"{path}: C4 top-k kernel anchor mismatch")
    path.write_text(start.replace(old, new, 1))


def patch_sm120_call_sites(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    source = path.read_text()
    old = '''                        output_buffers=self._global_topk_output_buffers(
                            self.topk_indices_buffer[:num_decode_tokens]
                        ),
                    )
'''
    new = '''                        output_buffers=self._global_topk_output_buffers(
                            self.topk_indices_buffer[:num_decode_tokens]
                        ),
                        dcp_world_size=self.dcp_world_size,
                        dcp_rank=self.dcp_rank,
                        cp_kv_cache_interleave_size=(
                            self.cp_kv_cache_interleave_size
                        ),
                    )
'''
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: decode C4 DCP call anchor mismatch")
    source = source.replace(old, new, 1)
    old = '''                    output_buffers=self._global_topk_output_buffers(local_topk_indices),
                )
'''
    new = '''                    output_buffers=self._global_topk_output_buffers(local_topk_indices),
                    # C4 indexer output contains exact global record ids after
                    # its DCP merge. C128 prefill metadata is already rank-local.
                    dcp_world_size=(
                        self.dcp_world_size if self.compress_ratio == 4 else 1
                    ),
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=(
                        self.cp_kv_cache_interleave_size
                    ),
                )
'''
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: prefill C4 DCP call anchor mismatch")
    path.write_text(source.replace(old, new, 1))


def patch_sm120_dcp_attention(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        '''from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
''',
        '''from vllm.v1.attention.backends.mla.compressor_utils import (
    get_dspark_swa_index_width,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import mask_dcp_empty_shards_
from vllm.v1.attention.ops.dcp_utils import MLADCPManager
''',
        "SM120 MLA DCP manager import",
    )
    replace_once(
        path,
        '''def _use_b12x_compressed_mla_decode() -> bool:
''',
        '''@triton.jit
def _dcp_correct_and_pack_rs_kernel(
    partial_output_ptr,
    gathered_lse_ptr,
    packed_output_ptr,
    output_stride_token,
    output_stride_head,
    output_stride_dim,
    lse_stride_rank,
    lse_stride_token,
    lse_stride_head,
    rank,
    num_tokens,
    HEAD_DIM: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    """LSE-correct local partials directly into NCCL destination chunks."""
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim = tl.arange(0, HEAD_DIM)
    peer = tl.arange(0, WORLD_SIZE)

    lse_offsets = (
        peer * lse_stride_rank
        + token * lse_stride_token
        + head * lse_stride_head
    )
    lses = tl.load(gathered_lse_ptr + lse_offsets).to(tl.float32)
    lses = tl.where(
        (lses != lses) | (lses == float("inf")), -float("inf"), lses
    )
    lse_max = tl.max(lses, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    global_lse = tl.log2(tl.sum(tl.exp2(lses - lse_max), axis=0)) + lse_max

    local_lse = tl.load(
        gathered_lse_ptr
        + rank * lse_stride_rank
        + token * lse_stride_token
        + head * lse_stride_head
    ).to(tl.float32)
    correction = local_lse - global_lse
    correction = tl.where(
        (correction != correction) | (correction == float("inf")),
        -float("inf"),
        correction,
    )
    factor = tl.exp2(correction)

    source_offsets = (
        token * output_stride_token
        + head * output_stride_head
        + dim * output_stride_dim
    )
    values = tl.load(partial_output_ptr + source_offsets) * factor
    values = tl.where(factor == 0.0, 0.0, values)

    destination_rank = head // LOCAL_HEADS
    local_head = head - destination_rank * LOCAL_HEADS
    packed_offsets = (
        destination_rank * num_tokens * LOCAL_HEADS * HEAD_DIM
        + token * LOCAL_HEADS * HEAD_DIM
        + local_head * HEAD_DIM
        + dim
    )
    tl.store(packed_output_ptr + packed_offsets, values)


def _dcp_ag_rs_combine_into(
    dcp_manager: MLADCPManager,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    destination: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Fuse LSE correction with rank-major packing and scatter in place."""
    group = dcp_manager.group
    world_size = group.world_size
    if dcp_manager.use_a2a or world_size <= 1:
        raise RuntimeError("packed AG/RS combine requires multi-rank ag_rs")
    num_tokens, global_heads, head_dim = partial_output.shape
    if global_heads % world_size != 0:
        raise RuntimeError(
            f"DCP global head count {global_heads} is not divisible by {world_size}."
        )
    local_heads = global_heads // world_size
    destination = destination[:num_tokens, :local_heads]
    if not destination.is_contiguous():
        raise RuntimeError("DeepSeek-V4 DCP output destination must be contiguous.")

    partial_lse = partial_lse.contiguous()
    mask_dcp_empty_shards_(partial_lse, seq_lens, query_start_loc)
    gathered_lse = group.all_gather(partial_lse, dim=0).reshape(
        world_size, num_tokens, global_heads
    )
    packed_output = partial_output.new_empty(
        (world_size * num_tokens, local_heads, head_dim)
    )
    _dcp_correct_and_pack_rs_kernel[(num_tokens, global_heads)](
        partial_output,
        gathered_lse,
        packed_output,
        *partial_output.stride(),
        *gathered_lse.stride(),
        group.rank_in_group,
        num_tokens,
        HEAD_DIM=head_dim,
        LOCAL_HEADS=local_heads,
        WORLD_SIZE=world_size,
    )

    communicator = group.device_communicator
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is None or pynccl_comm.disabled:
        raise RuntimeError(
            "DeepSeek-V4 packed DCP AG/RS requires the active PyNccl communicator."
        )
    pynccl_comm.reduce_scatter(destination, packed_output)


def _attention_sink_for_shard(attention) -> torch.Tensor:
    if getattr(attention, "dcp_manager", None) is not None:
        return attention._dcp_attn_sink
    return attention.attn_sink


def _use_b12x_compressed_mla_decode() -> bool:
''',
        "DCP sink selector",
    )
    replace_once(
        path,
        '''    def process_b12x_o_proj_weights_after_loading(self) -> None:
        if not self._b12x_o_proj_enabled:
            return
''',
        '''    def process_b12x_o_proj_weights_after_loading(self) -> None:
        if self.dcp_manager is not None:
            # Q heads are gathered before each rank attends its local KV shard,
            # so gather the TP-sharded sinks into the same global-head order.
            # Only one rank contributes the null-attention sink to the final
            # distributed log-sum-exp; counting it on every rank changes the
            # trained softmax denominator.
            gathered_sink = self.dcp_manager.group.all_gather(
                self.attn_sink.data[: self.n_local_heads].contiguous(), dim=0
            )
            self._dcp_attn_sink.fill_(-float("inf"))
            if self.dcp_rank == 0:
                self._dcp_attn_sink[: gathered_sink.shape[0]].copy_(gathered_sink)
        if not self._b12x_o_proj_enabled:
            return
''',
        "gather DCP attention sinks after loading",
    )
    replace_once(
        path,
        '''    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._b12x_o_proj_enabled = (
            vllm_config.kernel_config.linear_backend == "b12x"
        )
        self._b12x_o_proj_weights = None
''',
        '''    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self._b12x_o_proj_enabled = (
            vllm_config.kernel_config.linear_backend == "b12x"
        )
        self._b12x_o_proj_weights = None

        parallel_config = vllm_config.parallel_config
        self.dcp_world_size = parallel_config.decode_context_parallel_size
        self.cp_kv_cache_interleave_size = (
            parallel_config.cp_kv_cache_interleave_size
        )
        self.dcp_rank = 0
        self.dcp_manager: MLADCPManager | None = None
        self._dcp_attention_heads = self.padded_heads
        if self.dcp_world_size > 1:
            if self.cp_kv_cache_interleave_size != 1:
                raise NotImplementedError(
                    "DeepSeek-V4 SM12x DCP currently requires "
                    "cp_kv_cache_interleave_size=1."
                )
            if not self._b12x_o_proj_enabled:
                raise RuntimeError(
                    "DeepSeek-V4 SM12x DCP requires the B12x backend so "
                    "partial attention LSE can be merged exactly."
                )
            gathered_heads = self.n_local_heads * self.dcp_world_size
            self._dcp_attention_heads = _pad_to_supported_q_heads(gathered_heads)
            if self._dcp_attention_heads != gathered_heads:
                raise RuntimeError(
                    "DeepSeek-V4 DCP requires an unpadded gathered head count; "
                    f"got {gathered_heads} heads padded to "
                    f"{self._dcp_attention_heads}."
                )
            self.dcp_manager = MLADCPManager(
                vllm_config=vllm_config,
                device=self.attn_sink.device,
                num_heads=self.n_local_heads,
                query_head_dim=self.head_dim,
                output_head_dim=self.head_dim,
                query_dtype=torch.bfloat16,
                output_dtype=torch.bfloat16,
                padded_num_heads=self._dcp_attention_heads,
                is_lse_base_on_e=False,
                use_pcp=False,
            )
            self.dcp_rank = self.dcp_manager.group.rank_in_group
            self.register_buffer(
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
        "initialize SM120 DCP attention exchange",
    )
    replace_once(
        path,
        '''                self.kv_cache_dtype == "nvfp4_ds_mla"
                or _use_b12x_compressed_mla_decode()
''',
        '''                self.kv_cache_dtype == "nvfp4_ds_mla"
                or _use_b12x_compressed_mla_decode()
                or self.dcp_world_size > 1
''',
        "force LSE-capable B12x decode for DCP",
    )
    replace_once(
        path,
        '''                padded_heads=self.padded_heads,
''',
        '''                padded_heads=self._dcp_attention_heads,
''',
        "size B12x workspace for gathered DCP heads",
    )
    replace_once(
        path,
        '''        required_topks = {
            _required_sm120_sparse_topk(vllm_config, self.window_size),
            self.swa_cache_window_size,
        }
''',
        '''        required_topks = (
            {
                _required_sm120_sparse_topk(vllm_config, self.window_size),
                self.swa_cache_window_size,
            }
            if self.dcp_world_size == 1
            else set()
        )
''',
        "skip unused FlashInfer dispatch checks under DCP",
    )
    replace_once(
        path,
        '''    def _reserve_empty_forward_workspace(self) -> None:
        device = torch.device(
            "cuda", torch.accelerator.current_device_index()
        )
        self._get_workspace(device)
        if self._b12x_compressed_mla_enabled:
            self._get_b12x_compressed_mla_workspace(device)
''',
        '''    def _reserve_empty_forward_workspace(
        self, q: torch.Tensor, output: torch.Tensor
    ) -> None:
        device = torch.device(
            "cuda", torch.accelerator.current_device_index()
        )
        self._get_workspace(device)
        if self._b12x_compressed_mla_enabled:
            self._get_b12x_compressed_mla_workspace(device)
        if self.dcp_manager is not None:
            # Mirror the two DCP peak-live stages. Gathered Q is freed after
            # attention so its block can be reused by the packed AG/RS input.
            num_tokens = int(q.shape[0])
            _profile_gathered_q = torch.empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            _profile_partial_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            _profile_partial_lse = torch.empty(
                (num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=device,
            )
            del _profile_gathered_q
            _profile_gathered_lse = torch.empty(
                (self.dcp_world_size, num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=device,
            )
            _profile_packed_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            _ = (
                _profile_partial_output,
                _profile_partial_lse,
                _profile_gathered_lse,
                _profile_packed_output,
            )
''',
        "profile DCP attention exchange memory",
    )
    replace_once(
        path,
        '''        swa_kv_cache: torch.Tensor,
        swa_only: bool,
    ) -> None:
''',
        '''        swa_kv_cache: torch.Tensor,
        swa_only: bool,
        lse_output: torch.Tensor | None = None,
    ) -> None:
''',
        "DCP partial LSE output plumbing",
    )
    replace_once(
        path,
        '''                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
''',
        '''                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                lse_output=(
                    lse_output[num_decode_tokens:] if lse_output is not None else None
                ),
            )
''',
        "prefill partial LSE destination",
    )
    replace_once(
        path,
        '''                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )
''',
        '''                swa_only=swa_only,
                output=output[:num_decode_tokens],
                lse_output=(
                    lse_output[:num_decode_tokens] if lse_output is not None else None
                ),
            )
''',
        "decode partial LSE destination",
    )
    replace_once(
        path,
        '''        if attn_metadata is None:
            self._reserve_empty_forward_workspace()
            output.zero_()
            return
''',
        '''        if attn_metadata is None:
            self._reserve_empty_forward_workspace(q, output)
            output.zero_()
            return
''',
        "profile DCP attention buffers",
    )
    replace_once(
        path,
        '''        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
''',
        '''        if self.dcp_manager is not None:
            num_tokens = (
                swa_metadata.num_decode_tokens + swa_metadata.num_prefill_tokens
            )
            if num_tokens == 0:
                return
            local_query = q[:num_tokens, : self.n_local_heads]
            if local_query.dtype == torch.float8_e4m3fn:
                local_query = local_query.to(torch.bfloat16)
            elif local_query.dtype != torch.bfloat16:
                raise TypeError(
                    "DeepSeek-V4 DCP query must be BF16 or FP8, got "
                    f"{local_query.dtype}."
                )
            assert self.dcp_manager.query_gather is not None
            gathered_query = self.dcp_manager.query_gather(
                local_query.contiguous()
            )
            partial_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            partial_lse = torch.empty(
                (num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=output.device,
            )
            self._forward_sparse_impl(
                q=gathered_query,
                output=partial_output,
                flashmla_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
                self_kv_cache=self_kv_cache,
                swa_kv_cache=swa_kv_cache,
                swa_only=swa_only,
                lse_output=partial_lse,
            )
            # The packed combine input has the same size as gathered Q. Drop
            # the last reference now so the CUDA allocator can reuse that block.
            del gathered_query
            if swa_metadata.dcp_local_seq_lens is None:
                raise RuntimeError("DeepSeek-V4 DCP local sequence lengths missing.")
            if swa_metadata.query_start_loc is None:
                raise RuntimeError("DeepSeek-V4 DCP query offsets missing.")
            num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
            seq_lens = swa_metadata.dcp_local_seq_lens[:num_reqs]
            query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
            if self.dcp_manager.use_a2a:
                combined = self.dcp_manager.combine(
                    partial_output,
                    partial_lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
                output[:num_tokens].copy_(combined)
            else:
                _dcp_ag_rs_combine_into(
                    self.dcp_manager,
                    partial_output,
                    partial_lse,
                    output[:num_tokens],
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
            return

        self._forward_sparse_impl(
            q=q,
            output=output,
            flashmla_metadata=flashmla_metadata,
            swa_metadata=swa_metadata,
            self_kv_cache=self_kv_cache,
            swa_kv_cache=swa_kv_cache,
            swa_only=swa_only,
        )

    def _prepare_query(self, q: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
''',
        "gather Q and LSE-merge DCP partial attention",
    )
    replace_once(
        path,
        '''        indexed_page_size: int | None,
        output: torch.Tensor,
    ) -> None:
''',
        '''        indexed_page_size: int | None,
        output: torch.Tensor,
        return_lse: bool = False,
    ) -> torch.Tensor | None:
''',
        "B12x decode LSE API",
    )
    replace_once(
        path,
        '''        compressed_sparse_mla.run(
            binding=binding,
            swa_k_cache=swa_cache,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_cache,
            indexed_page_size=indexed_page_size,
            attn_sink=self.attn_sink,
            sm_scale=self.scale,
            expected_num_q_heads=self.padded_heads,
            out=output,
        )
        logger.info_once(
            "Using shared B12x compressed sparse MLA decode on SM12x."
        )
''',
        '''        run_kwargs = dict(
            binding=binding,
            swa_k_cache=swa_cache,
            swa_page_size=swa_page_size,
            indexed_k_cache=indexed_cache,
            indexed_page_size=indexed_page_size,
            attn_sink=_attention_sink_for_shard(self),
            sm_scale=self.scale,
            expected_num_q_heads=self._dcp_attention_heads,
            out=output,
        )
        lse = None
        if return_lse:
            _, lse = compressed_sparse_mla.run(
                **run_kwargs, return_lse=True, lse_scale="base2"
            )
        else:
            compressed_sparse_mla.run(**run_kwargs)
        logger.info_once(
            "Using shared B12x compressed sparse MLA decode on SM12x."
        )
        return lse
''',
        "return B12x decode LSE",
    )
    replace_once(
        path,
        '''    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
''',
        '''    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
        lse_output: torch.Tensor | None = None,
    ) -> None:
''',
        "decode LSE destination API",
    )
    replace_once(
        path,
        '''                ),
                output=output,
            )
        else:
''',
        '''                ),
                output=output,
                return_lse=lse_output is not None,
            )
            if lse_output is not None:
                if lse is None:
                    raise RuntimeError("B12x DCP decode did not return LSE.")
                lse_output.copy_(lse)
        else:
            if lse_output is not None:
                raise RuntimeError(
                    "DeepSeek-V4 DCP requires the LSE-capable B12x decode path."
                )
''',
        "capture decode partial LSE",
    )
    # Name the result introduced by the previous replacement.
    replace_once(
        path,
        '''        if self._b12x_compressed_mla_enabled:
            self._b12x_compressed_mla_decode(
''',
        '''        if self._b12x_compressed_mla_enabled:
            lse = self._b12x_compressed_mla_decode(
''',
        "bind B12x decode LSE result",
    )
    replace_once(
        path,
        '''                sinks=self.attn_sink,
                kv_layout="NHD",
''',
        '''                sinks=_attention_sink_for_shard(self),
                kv_layout="NHD",
''',
        "single-owner sink in fallback decode",
    )
    replace_once(
        path,
        '''        extra_lengths: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
''',
        '''        extra_lengths: torch.Tensor | None,
        output: torch.Tensor,
    ) -> torch.Tensor:
''',
        "B12x prefill LSE return API",
    )
    replace_once(
        path,
        '''        run_unified_prefill(
''',
        '''        prefill_result = run_unified_prefill(
''',
        "bind B12x prefill LSE",
    )
    replace_once(
        path,
        '''            attn_sink=self.attn_sink,
            output=output,
''',
        '''            attn_sink=_attention_sink_for_shard(self),
            output=output,
''',
        "single-owner sink in B12x prefill",
    )
    replace_once(
        path,
        '''            ),
        )

    def _forward_prefill(
''',
        '''            ),
        )
        # Some no-GPU contract fixtures replace the launcher with a sink that
        # returns None; the real B12x launcher returns ``(output, lse_out)``.
        if prefill_result is None:
            return lse_out
        return prefill_result[1]

    def _forward_prefill(
''',
        "return B12x prefill LSE",
    )
    replace_once(
        path,
        '''        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
''',
        '''        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
        lse_output: torch.Tensor | None = None,
    ) -> None:
''',
        "prefill LSE destination API",
    )
    replace_once(
        path,
        '''            if use_b12x_prefill:
                self._b12x_prefill(
''',
        '''            if self.dcp_manager is not None and not use_b12x_prefill:
                raise RuntimeError(
                    "DeepSeek-V4 DCP requires a B12x-supported prefill width; "
                    f"got {b12x_prefill_width}."
                )
            if use_b12x_prefill:
                chunk_lse = self._b12x_prefill(
''',
        "force LSE-capable B12x prefill under DCP",
    )
    replace_once(
        path,
        '''                    output=output_chunk,
                )
            else:
''',
        '''                    output=output_chunk,
                )
                if lse_output is not None:
                    lse_output[query_start:query_end].copy_(chunk_lse)
            else:
                if lse_output is not None:
                    raise RuntimeError(
                        "DeepSeek-V4 DCP requires the LSE-capable B12x prefill path."
                    )
''',
        "capture prefill partial LSE",
    )
    # The only remaining SM120 fallback sink is the prefill launcher.
    source = path.read_text()
    marker = '''                    sinks=self.attn_sink,
                    kv_layout="NHD",
'''
    if source.count(marker) != 1:
        raise RuntimeError(f"{path}: fallback prefill sink anchor mismatch")
    path.write_text(
        source.replace(
            marker,
            '''                    sinks=_attention_sink_for_shard(self),
                    kv_layout="NHD",
''',
            1,
        )
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dcp-dsv4.py VLLM_ROOT")
    root = Path(sys.argv[1]).resolve()
    if not (root / "models/deepseek_v4/attention.py").is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")
    patch_dcp_rank_scalar_sync(root)
    patch_draft_dcp_metadata(root)
    patch_compressed_slot_mapping(root)
    patch_indexer_metadata(root)
    patch_c128_metadata(root)
    patch_swa_metadata(root)
    patch_c4_topk_mapping(root)
    patch_sm120_call_sites(root)
    patch_sm120_dcp_attention(root)


if __name__ == "__main__":
    main()
