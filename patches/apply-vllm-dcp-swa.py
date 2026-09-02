#!/usr/bin/env python3
"""Make vLLM's sliding-window KV manager DCP-aware."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def replace_count(
    path: Path, old: str, new: str, expected: int, label: str
) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != expected:
        raise RuntimeError(
            f"{path}: {label} expected {expected} anchors, found {count}"
        )
    path.write_text(source.replace(old, new))


def patch_spec(root: Path) -> None:
    path = root / "v1/kv_cache_interface.py"
    replace_once(
        path,
        '''    extra_retained_tokens: int = 0

    def max_admission_blocks_per_request(
        self, max_in_flight_tokens: int, max_model_len: int
    ) -> int:
''',
        '''    extra_retained_tokens: int = 0

    def max_admission_blocks_per_request(
        self,
        max_in_flight_tokens: int,
        max_model_len: int,
        dcp_world_size: int = 1,
    ) -> int:
''',
        "DCP-aware sliding-window admission signature",
    )
    replace_once(
        path,
        '''        # +1 because the sliding window may not start from the beginning of
        # the block. E.g. block size 4 and num_token 4 needs two blocks
        # [XXCD][EF] to store the 6-token window [CDEF].
        return cdiv(num_tokens, self.block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        assert vllm_config.parallel_config.decode_context_parallel_size == 1, (
            "DCP not support sliding window."
        )
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
        )
''',
        '''        # +1 because the sliding window may not start from the beginning of
        # the effective DCP block. E.g. block size 4 and num_token 4 needs two
        # blocks [XXCD][EF] to store the 6-token window [CDEF]. A physical page
        # on each rank spans ``block_size * dcp_world_size`` global tokens.
        effective_block_size = self.block_size * dcp_world_size
        return cdiv(num_tokens, effective_block_size) + 1

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_blocks = self.max_admission_blocks_per_request(
            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            max_model_len=vllm_config.model_config.max_model_len,
            dcp_world_size=(
                vllm_config.parallel_config.decode_context_parallel_size
            ),
        )
''',
        "DCP-aware sliding-window admission accounting",
    )


def patch_manager(root: Path) -> None:
    path = root / "v1/core/single_type_kv_cache_manager.py"
    replace_count(
        path,
        '''        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
    ) -> list[bool] | None:
''',
        '''        retention_interval: int | None = None,
        reachable_boundaries: Sequence[int] = (),
        effective_block_size: int | None = None,
    ) -> list[bool] | None:
''',
        3,
        "effective block size mask API",
    )
    replace_once(
        path,
        '''            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
        )
''',
        '''            retention_interval=retention_interval,
            reachable_boundaries=reachable_boundaries,
            effective_block_size=self.block_size,
        )
''',
        "effective block size mask call",
    )
    replace_once(
        path,
        '''        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."
        # Fine-grained partial hits are not supported for sliding window now
        assert alignment_tokens % kv_cache_spec.block_size == 0, (
            "SlidingWindowManager does not support fine-grained (partial) cache hits"
        )
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            kv_cache_spec.block_size,
''',
        '''        assert dcp_world_size >= 1
        assert pcp_world_size == 1, "PCP not support sliding window attn now."
        # One physical KV page per rank represents this many global tokens.
        block_size = kv_cache_spec.block_size * dcp_world_size
        # Fine-grained partial hits are not supported for sliding window now.
        assert alignment_tokens % block_size == 0, (
            "SlidingWindowManager does not support fine-grained (partial) cache hits"
        )
        block_hashes = resolve_block_hashes(
            block_hashes,
            block_pool.hash_block_size,
            block_size,
''',
        "DCP-aware sliding-window hit preamble",
    )
    replace_once(
        path,
        '''        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, kv_cache_spec.block_size, drop_eagle_block
        )
''',
        '''        sliding_window_contiguous_blocks = cls._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window, block_size, drop_eagle_block
        )
''',
        "DCP-aware sliding-window hit width",
    )
    replace_once(
        path,
        '''        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
''',
        '''        max_num_blocks = max_length // block_size
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
''',
        "DCP-aware sliding-window block count",
    )
    replace_once(
        path,
        '''        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        if alignment_tokens is None:
            # Fast path: when the coordinator imposes no alignment constraint.
            return None
        assert alignment_tokens % kv_cache_spec.block_size == 0

        block_size = kv_cache_spec.block_size
''',
        '''        assert isinstance(kv_cache_spec, SlidingWindowSpec)
        if alignment_tokens is None:
            # Fast path: when the coordinator imposes no alignment constraint.
            return None
        block_size = effective_block_size or kv_cache_spec.block_size
        assert alignment_tokens % block_size == 0
''',
        "DCP-aware sliding-window retention mask",
    )
    replace_once(
        path,
        '''        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(
                max_in_flight_tokens=max_in_flight_tokens,
                max_model_len=max_model_len,
            )
        )
''',
        '''        admission_kwargs = {
            "max_in_flight_tokens": max_in_flight_tokens,
            "max_model_len": max_model_len,
        }
        if isinstance(kv_cache_spec, SlidingWindowSpec):
            admission_kwargs["dcp_world_size"] = int(
                kwargs.get("dcp_world_size", 1)
            )
        kwargs["max_admission_blocks_per_request"] = (
            kv_cache_spec.max_admission_blocks_per_request(**admission_kwargs)
        )
''',
        "DCP-aware runtime admission cap",
    )


def patch_coordinator(root: Path) -> None:
    path = root / "v1/core/kv_cache_coordinator.py"
    replace_once(
        path,
        '''        if dcp_world_size > 1:
            # DCP shards full-attention KV across ranks and replicates Mamba
            # state; other spec types (e.g. sliding window) have no DCP-aware
            # handling yet, so reject them explicitly.
            for g in kv_cache_config.kv_cache_groups:
                assert isinstance(g.kv_cache_spec, (FullAttentionSpec, MambaSpec)), (
                    "DCP with hybrid KV cache layouts only supports "
                    "full-attention and Mamba groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
''',
        '''        if dcp_world_size > 1:
            # DCP shards both full-attention and sliding-window KV pages across
            # ranks. Mamba state remains replicated.
            for g in kv_cache_config.kv_cache_groups:
                assert isinstance(
                    g.kv_cache_spec,
                    (FullAttentionSpec, SlidingWindowSpec, MambaSpec),
                ), (
                    "DCP with hybrid KV cache layouts only supports full-attention, "
                    "sliding-window, and Mamba groups, got: "
                    f"{type(g.kv_cache_spec).__name__}."
                )
''',
        "allow DCP sliding-window hybrid groups",
    )
    replace_once(
        path,
        '''                    dcp_world_size=(
                        self.dcp_world_size
                        if isinstance(spec, FullAttentionSpec)
                        else 1
                    ),
''',
        '''                    dcp_world_size=(
                        self.dcp_world_size
                        if isinstance(spec, (FullAttentionSpec, SlidingWindowSpec))
                        else 1
                    ),
''',
        "pass DCP width to sliding-window hit lookup",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dcp-swa.py VLLM_ROOT")
    root = Path(sys.argv[1])
    if not (root / "v1/kv_cache_interface.py").is_file():
        raise RuntimeError(f"not a vLLM source root: {root}")
    patch_spec(root)
    patch_manager(root)
    patch_coordinator(root)


if __name__ == "__main__":
    main()
