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


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dsv4-kv-groups.py VLLM_ROOT")

    path = Path(sys.argv[1]) / "v1/core/kv_cache_utils.py"
    replace_block(
        path,
        "def _get_kv_cache_groups_uniform_groups(\n",
        "def _annotate_eagle_groups_deepseek_v4(\n",
        '''def _get_kv_cache_groups_uniform_groups(
    vllm_config: VllmConfig,
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
    # one group, which makes that stack the byte stride even when a smaller
    # C4, C128, or SWA group owns the block. Choose a separate tuple width for
    # each cache family by minimizing the exact one-request admission bytes:
    #
    #   shared stride * sum(groups_for_family * max_pages_for_family)
    #
    # A group-count ceiling bounds scheduler metadata and Python BlockPool
    # overhead. Candidate strides come from real tuple/page boundaries, so the
    # result is deterministic and contains no heuristic byte increments.
    # More groups reduce slab padding but add one block-table/metadata path per
    # group on every decode/prefill step. Preserve the established five-group
    # layout for every release profile: a 14-group Vision diagnostic regressed
    # matched C4 decode by 9.8%, while a six-group 131K/8K text diagnostic
    # regressed matched 8K prefill by 7.5%.
    max_groups = 5
    tuple_counts = [
        spec.get_num_layer_tuples() for spec in grouped_specs
    ]
    tuple_bytes = [sum(spec.get_page_sizes()) for spec in grouped_specs]
    max_pages = [
        spec.max_memory_usage_pages(vllm_config) for spec in grouped_specs
    ]
    candidate_strides = sorted(
        {
            page_bytes * width
            for page_bytes, tuple_count in zip(tuple_bytes, tuple_counts)
            for width in range(1, tuple_count + 1)
        }
    )
    best: tuple[int, int, int, list[int], list[int]] | None = None
    for candidate_stride in candidate_strides:
        max_tuple_widths = [
            min(tuple_count, candidate_stride // page_bytes)
            for page_bytes, tuple_count in zip(tuple_bytes, tuple_counts)
        ]
        if any(width < 1 for width in max_tuple_widths):
            continue
        group_counts = [
            cdiv(tuple_count, width)
            for tuple_count, width in zip(tuple_counts, max_tuple_widths)
        ]
        num_groups = sum(group_counts)
        if num_groups > max_groups:
            continue
        # Once the group count is known, spread tuples evenly across its
        # groups. This retains upstream's balanced/interleaved physical layout
        # (for example, C4 46 -> 23/23 rather than 26/20) without changing the
        # packed stride or one-request admission bytes.
        tuple_widths = [
            cdiv(tuple_count, group_count)
            for tuple_count, group_count in zip(tuple_counts, group_counts)
        ]
        block_stride = max(
            width * page_bytes
            for width, page_bytes in zip(tuple_widths, tuple_bytes)
        )
        required_blocks = sum(
            group_count * pages
            for group_count, pages in zip(group_counts, max_pages)
        )
        candidate = (
            block_stride * required_blocks,
            num_groups,
            block_stride,
            tuple_widths,
            group_counts,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate
    assert best is not None
    required_bytes, _, _, tuple_widths, group_counts = best

    def split_spec(
        grouped_spec: UniformTypeKVCacheSpecs,
        tuple_width: int,
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
        num_tuple_groups = cdiv(tuple_count, tuple_width)
        for group_index in range(num_tuple_groups):
            group_layer_names = [
                layer_name
                for layer_tuple in layer_tuples[group_index::num_tuple_groups]
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
        for grouped_spec, tuple_width in zip(grouped_specs, tuple_widths)
        for group in split_spec(grouped_spec, tuple_width)
    ]
    packed_stride = max(
        cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).page_size_bytes
        for group in kv_cache_groups
    )
    # The optimizer scores a family by group_count * max_pages. Recompute the
    # final number from the actual split groups so partial tail groups and the
    # startup log use the exact same accounting as admission control below.
    required_bytes = packed_stride * sum(
        cast(
            UniformTypeKVCacheSpecs, group.kv_cache_spec
        ).max_memory_usage_pages(vllm_config)
        for group in kv_cache_groups
    )
    logger.info(
        "DeepSeek-V4 packed KV groups: tuple_counts=%s, tuple_bytes=%s, "
        "max_pages=%s, tuple_widths=%s, group_counts=%s, groups=%d, "
        "block_stride=%d bytes, one_request=%d bytes",
        tuple_counts,
        tuple_bytes,
        max_pages,
        tuple_widths,
        group_counts,
        len(kv_cache_groups),
        packed_stride,
        required_bytes,
    )
    return kv_cache_groups


''',
        "balanced DeepSeek-V4 KV tuple groups",
    )
    replace_once(
        path,
        "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(grouped_specs)\n",
        "        kv_cache_groups = _get_kv_cache_groups_uniform_groups(\n"
        "            vllm_config, grouped_specs\n"
        "        )\n",
        "pass config to DeepSeek-V4 KV group optimizer",
    )
    replace_once(
        path,
        "        # Special case (only DeepseekV4 for now): all groups are\n"
        "        # UniformTypeKVCacheSpecs.\n"
        "        # They must already be page_size aligned and share a common padded\n"
        "        # layer-tuple layout. Even groups with fewer actual tuples still reserve\n"
        "        # the global number of tuple slots in the shared tensor layout.\n"
        "        full_mla_spec = cast(UniformTypeKVCacheSpecs, kv_cache_groups[0].kv_cache_spec)\n"
        "        layer_tuple_bytes = sum(full_mla_spec.get_page_sizes())\n"
        "        num_layer_tuples = max(\n"
        "            cast(UniformTypeKVCacheSpecs, group.kv_cache_spec).get_num_layer_tuples()\n"
        "            for group in kv_cache_groups\n"
        "        )\n"
        "\n"
        "        total_max_mem_usage_bytes = 0\n"
        "        for group in kv_cache_groups:\n"
        "            group_spec = cast(UniformTypeKVCacheSpecs, group.kv_cache_spec)\n"
        "            g_max_mem_usage_pages = group_spec.max_memory_usage_pages(vllm_config)\n"
        "            g_max_mem_usage_page_bytes = (\n"
        "                num_layer_tuples * g_max_mem_usage_pages * layer_tuple_bytes\n"
        "            )\n"
        "            total_max_mem_usage_bytes += g_max_mem_usage_page_bytes\n"
        "        return total_max_mem_usage_bytes\n",
        "        # DeepSeek-V4 groups draw block IDs from one packed global slab.\n"
        "        # Group widths may differ, so account with the actual maximum\n"
        "        # physical stride and each group's own admission-page count.\n"
        "        block_stride, _ = _get_packed_kv_cache_layout(kv_cache_groups)\n"
        "        return block_stride * sum(\n"
        "            cast(\n"
        "                UniformTypeKVCacheSpecs, group.kv_cache_spec\n"
        "            ).max_memory_usage_pages(vllm_config)\n"
        "            for group in kv_cache_groups\n"
        "        )\n",
        "variable-width packed KV admission accounting",
    )


if __name__ == "__main__":
    main()
