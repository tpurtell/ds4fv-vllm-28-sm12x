#!/usr/bin/env python3
"""Enable DeepSeek-V4 indexer metadata for DSpark adaptive verification.

vLLM 0.28 already implements confidence-based DSpark verification, but its
SM12x DeepSeek-V4 indexer rejects device/CPU query-length mismatches.  The
flattened SM12x path is already device-driven except for a uniform fast-path
decision.  Adaptive verification preserves the total verification budget and
the decode/prefill boundary, so force the existing device-length expansion and
advertise graph support for that path.

Every edit is anchored to the pinned vLLM source and must match exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label} expected exactly one source anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


def patch_indexer(root: Path) -> None:
    path = root / "v1/attention/backends/mla/indexer.py"
    replace_once(
        path,
        "class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):\n"
        "    @staticmethod\n"
        "    def get_name() -> str:\n",
        "class DeepseekV4IndexerBackend(DeepseekV32IndexerBackend):\n"
        "    @classmethod\n"
        "    def supports_device_cpu_query_lens_mismatch(cls) -> bool:\n"
        "        # DSpark adaptive verification preserves the global draft budget\n"
        "        # but assigns different per-request lengths on device. The SM12x\n"
        "        # flattened path below expands from those device lengths.\n"
        "        return True\n\n"
        "    @staticmethod\n"
        "    def get_name() -> str:\n",
        "DeepSeek-V4 adaptive-verification capability",
    )
    replace_once(
        path,
        "        if _supports_varlen_paged_mqa_logits():\n"
        "            return AttentionCGSupport.ALWAYS\n"
        "        return AttentionCGSupport.UNIFORM_BATCH\n",
        "        speculative_config = vllm_config.speculative_config\n"
        "        if _supports_varlen_paged_mqa_logits() or (\n"
        "            speculative_config is not None\n"
        "            and speculative_config.method == \"dspark\"\n"
        "            and speculative_config.enable_adaptive_verification\n"
        "        ):\n"
        "            # The adaptive SM12x path is token-count graph stable: the\n"
        "            # total draft budget is unchanged while device query lengths\n"
        "            # select the per-request flattened rows.\n"
        "            return AttentionCGSupport.ALWAYS\n"
        "        return AttentionCGSupport.UNIFORM_BATCH\n",
        "adaptive-verification CUDA graph support",
    )
    replace_once(
        path,
        "        self.num_speculative_tokens = (\n"
        "            self.vllm_config.speculative_config.num_speculative_tokens\n"
        "            if self.vllm_config.speculative_config\n"
        "            else 0\n"
        "        )\n"
        "        self.use_fp4_indexer_cache = dsa_indexer_uses_fp4(self.vllm_config)\n",
        "        speculative_config = self.vllm_config.speculative_config\n"
        "        self.num_speculative_tokens = (\n"
        "            speculative_config.num_speculative_tokens\n"
        "            if speculative_config\n"
        "            else 0\n"
        "        )\n"
        "        self.enable_adaptive_verification = bool(\n"
        "            speculative_config\n"
        "            and speculative_config.method == \"dspark\"\n"
        "            and speculative_config.enable_adaptive_verification\n"
        "        )\n"
        "        self.use_fp4_indexer_cache = dsa_indexer_uses_fp4(self.vllm_config)\n",
        "adaptive-verification builder state",
    )
    replace_once(
        path,
        "            if (\n"
        "                not self.supports_varlen\n"
        "                and min_decode_len == max_decode_len\n",
        "            if (\n"
        "                not self.enable_adaptive_verification\n"
        "                and not self.supports_varlen\n"
        "                and min_decode_len == max_decode_len\n",
        "device-length flattened decode selection",
    )


def patch_sparse_mla(root: Path) -> None:
    path = root / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        path,
        "    _cudagraph_support: ClassVar[AttentionCGSupport] = "
        "AttentionCGSupport.UNIFORM_BATCH\n\n"
        "    def __init__(\n",
        "    _cudagraph_support: ClassVar[AttentionCGSupport] = "
        "AttentionCGSupport.UNIFORM_BATCH\n\n"
        "    @classmethod\n"
        "    def get_cudagraph_support(\n"
        "        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec\n"
        "    ) -> AttentionCGSupport:\n"
        "        speculative_config = vllm_config.speculative_config\n"
        "        if (\n"
        "            speculative_config is not None\n"
        "            and speculative_config.method == \"dspark\"\n"
        "            and speculative_config.enable_adaptive_verification\n"
        "        ):\n"
        "            # Sparse MLA metadata consumes device query_start_loc and\n"
        "            # device token-to-request rows; adaptive verification keeps\n"
        "            # the total token count and decode/prefill boundary stable.\n"
        "            return AttentionCGSupport.ALWAYS\n"
        "        return cls._cudagraph_support\n\n"
        "    def __init__(\n",
        "sparse MLA adaptive-verification CUDA graph support",
    )


def patch_sparse_swa(root: Path) -> None:
    path = root / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        path,
        "    _cudagraph_support: ClassVar[AttentionCGSupport] = "
        "AttentionCGSupport.UNIFORM_BATCH\n"
        "    supports_draft_decode_metadata_update = True\n\n"
        "    def __init__(self, *args, **kwargs):\n",
        "    _cudagraph_support: ClassVar[AttentionCGSupport] = "
        "AttentionCGSupport.UNIFORM_BATCH\n"
        "    supports_draft_decode_metadata_update = True\n\n"
        "    @classmethod\n"
        "    def get_cudagraph_support(\n"
        "        cls, vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec\n"
        "    ) -> AttentionCGSupport:\n"
        "        speculative_config = vllm_config.speculative_config\n"
        "        if (\n"
        "            speculative_config is not None\n"
        "            and speculative_config.method == \"dspark\"\n"
        "            and speculative_config.enable_adaptive_verification\n"
        "        ):\n"
        "            # SWA indices, lengths, and token-to-request mapping are all\n"
        "            # built from device metadata. CPU lengths only describe the\n"
        "            # unchanged prefill portion and scheduling upper bounds.\n"
        "            return AttentionCGSupport.ALWAYS\n"
        "        return cls._cudagraph_support\n\n"
        "    def __init__(self, *args, **kwargs):\n",
        "sparse SWA adaptive-verification CUDA graph support",
    )


def patch_global_topk_padding(root: Path) -> None:
    path = root / "models/deepseek_v4/common/ops/cache_utils.py"
    replace_once(
        path,
        "    is_valid_token = tl.load(is_valid_token_ptr + token_idx)\n"
        "    req_idx = tl.load(token_to_req_indices_ptr + token_idx)\n\n"
        "    count = tl.zeros((), dtype=tl.int32)\n",
        "    is_valid_token = tl.load(is_valid_token_ptr + token_idx)\n"
        "    req_idx = tl.load(token_to_req_indices_ptr + token_idx)\n"
        "    # Graph replay may include padded token rows whose request mapping\n"
        "    # and top-k buffer still contain stale values. Clamp the request row\n"
        "    # before pointer arithmetic; validity then gates every table lookup.\n"
        "    safe_req_idx = tl.where(is_valid_token, req_idx, 0)\n\n"
        "    count = tl.zeros((), dtype=tl.int32)\n",
        "global top-k padded request clamp",
    )
    replace_once(
        path,
        "        is_valid = local_idx >= 0\n\n"
        "        block_indices = local_idx // block_size\n"
        "        block_numbers = tl.load(\n"
        "            block_table_ptr + req_idx * block_table_stride + block_indices,\n",
        "        is_valid = (local_idx >= 0) & is_valid_token\n\n"
        "        block_indices = local_idx // block_size\n"
        "        block_numbers = tl.load(\n"
        "            block_table_ptr\n"
        "            + safe_req_idx * block_table_stride\n"
        "            + block_indices,\n",
        "global top-k padded lookup mask",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1]).resolve()
    patch_indexer(root)
    patch_sparse_mla(root)
    patch_sparse_swa(root)
    patch_global_topk_padding(root)


if __name__ == "__main__":
    main()
