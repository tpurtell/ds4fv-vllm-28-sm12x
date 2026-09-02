#!/usr/bin/env python3
"""Balance DeepSeek-V4 hybrid KV tuple groups for dense slab packing."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_block(
    path: Path,
    start: str,
    end: str,
    replacement: str,
    label: str,
) -> None:
    source = path.read_text()
    if source.count(start) != 1 or source.count(end) != 1:
        raise RuntimeError(f"{path}: {label} source anchors are not unique")
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    path.write_text(source[:start_index] + replacement + source[end_index:])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dsv4-kv-groups.py VLLM_ROOT")

    path = Path(sys.argv[1]) / "v1/core/kv_cache_utils.py"
    replace_block(
        path,
        "def _get_kv_cache_groups_uniform_groups(\n",
        "def _annotate_eagle_groups_deepseek_v4(\n",
        '''def _get_kv_cache_groups_uniform_groups(
    grouped_specs: list[UniformTypeKVCacheSpecs],
) -> list[KVCacheGroupSpec]:
    """Generate balanced packed-cache groups for DeepSeek-V4."""
    assert len(grouped_specs) > 0 and all(
        isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs
    )
    full_mla_spec = grouped_specs[0]
    assert all(
        isinstance(spec, MLAAttentionSpec)
        for spec in full_mla_spec.kv_cache_specs.values()
    )
    assert all(
        isinstance(spec, SlidingWindowMLASpec)
        for group in grouped_specs[1:]
        for spec in group.kv_cache_specs.values()
    )

    # Every block ID is owned by one cache group, while all groups draw from
    # one packed physical slab. Upstream keeps the complete full-MLA stack in
    # one group, which makes that 21-tuple stack the byte stride even when a
    # much smaller C4, C128, or SWA group owns the block. Chunk every cache
    # family into the same small tuple width so the shared slab retains its
    # flexibility without paying a whole-model stride for a partial group.
    requested_tuple_width = int(
        os.getenv("VLLM_DSV4_KV_TUPLES_PER_GROUP", "3")
    )
    if requested_tuple_width < 1:
        raise ValueError(
            "VLLM_DSV4_KV_TUPLES_PER_GROUP must be at least 1, got "
            f"{requested_tuple_width}"
        )
    max_tuple_count = max(
        spec.get_num_layer_tuples() for spec in grouped_specs
    )
    tuple_width = min(requested_tuple_width, max_tuple_count)

    def split_spec(
        grouped_spec: UniformTypeKVCacheSpecs,
    ) -> list[KVCacheGroupSpec]:
        layers_per_size: dict[int, list[str]] = defaultdict(list)
        for layer_name, layer_spec in grouped_spec.kv_cache_specs.items():
            layers_per_size[layer_spec.page_size_bytes].append(layer_name)

        # Full MLA can have one more C4 tuple than C128. Preserve that partial
        # tail instead of truncating it as zip() would; SWA families generally
        # contain one page size and follow the same code path.
        page_lanes = list(layers_per_size.values())
        tuple_count = max(len(lane) for lane in page_lanes)
        layer_tuples = [
            tuple(lane[index] for lane in page_lanes if index < len(lane))
            for index in range(tuple_count)
        ]

        groups: list[KVCacheGroupSpec] = []
        for tuple_start in range(0, tuple_count, tuple_width):
            group_layer_names = [
                layer_name
                for layer_tuple in layer_tuples[
                    tuple_start : tuple_start + tuple_width
                ]
                for layer_name in layer_tuple
            ]
            group_layer_specs = {
                name: grouped_spec.kv_cache_specs[name]
                for name in group_layer_names
            }
            sub_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
            assert sub_spec is not None
            groups.append(
                KVCacheGroupSpec(
                    layer_names=group_layer_names,
                    kv_cache_spec=sub_spec,
                )
            )
        return groups

    kv_cache_groups = [
        group
        for grouped_spec in grouped_specs
        for group in split_spec(grouped_spec)
    ]
    packed_stride = max(
        cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).page_size_bytes
        for group in kv_cache_groups
    )
    logger.info(
        "DeepSeek-V4 packed KV groups: tuple_counts=%s, tuple_width=%d, "
        "groups=%d, block_stride=%d bytes",
        [spec.get_num_layer_tuples() for spec in grouped_specs],
        tuple_width,
        len(kv_cache_groups),
        packed_stride,
    )
    return kv_cache_groups


''',
        "balanced DeepSeek-V4 KV tuple groups",
    )


if __name__ == "__main__":
    main()
