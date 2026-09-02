#!/usr/bin/env python3
"""Size the sparse-indexer gather workspace from serving concurrency."""

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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-indexer-workspace.py VLLM_ROOT")
    path = Path(sys.argv[1]) / "v1/attention/backends/mla/indexer.py"
    replace_once(
        path,
        "def get_max_prefill_buffer_size(vllm_config: VllmConfig):\n"
        "    max_model_len = vllm_config.model_config.max_model_len\n"
        "    # NOTE(Chen): 40 is a magic number for controlling the prefill buffer size.\n"
        "    # Each entry is 128 fp8 bytes and 4 scale bytes for a total of 132 bytes.\n"
        "    # The flashmla_sparse backend uses a workspace size of 5 * max_model_len.\n"
        "    # The memory usage of the workspace there is 576 * 2 bytes; so we size this as\n"
        "    # (576 * 2 // 132) * 5 = 40 to maximize this workspace size while still fitting\n"
        "    # within the flashmla_sparse workspace.\n"
        "    # For DeepSeek-V3.2, the max_model_len is 163840.\n"
        "    #   40 * 163840 * 132 = 865075200 bytes = 825 MB\n"
        "    return max_model_len * 40\n",
        "def get_max_prefill_buffer_size(vllm_config: VllmConfig):\n"
        "    max_model_len = vllm_config.model_config.max_model_len\n"
        "    max_num_seqs = vllm_config.scheduler_config.max_num_seqs\n"
        "    # The gather workspace contains each request's K sequence once; query rows\n"
        "    # are independently sliced by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB. The\n"
        "    # metadata builder also splits request groups when their summed K lengths\n"
        "    # exceed this bound. Consequently max_num_seqs full-length contexts are\n"
        "    # the largest legal live set, while one max_model_len is the correctness\n"
        "    # floor for a single uncompressed indexer. Preserve upstream's multiplier\n"
        "    # as a ceiling for unusually high-concurrency profiles.\n"
        "    return max_model_len * max(1, min(40, max_num_seqs))\n",
        "concurrency-bounded sparse-indexer gather workspace",
    )
    replace_once(
        path,
        "        self.expanded_block_table_buffer = torch.zeros(\n"
        "            (scheduler_config.max_num_batched_tokens, block_table_width),\n"
        "            dtype=torch.int32,\n"
        "            device=self.device,\n"
        "        )\n",
        "        # This table is consumed only by flattened decode rows, never by\n"
        "        # prefill. Its legal live rows are bounded by request concurrency\n"
        "        # times the target+draft width, plus CUDA-graph padding. Sizing the\n"
        "        # row count from max_num_batched_tokens multiplies the long-context\n"
        "        # block-table width by the 8K prefill budget and wastes hundreds of\n"
        "        # MiB per target/draft builder.\n"
        "        self.max_decode_tokens = max(\n"
        "            scheduler_config.max_num_seqs * next_n,\n"
        "            self.vllm_config.compilation_config.max_cudagraph_capture_size\n"
        "            or 0,\n"
        "        )\n"
        "        self.expanded_block_table_buffer = torch.zeros(\n"
        "            (self.max_decode_tokens, block_table_width),\n"
        "            dtype=torch.int32,\n"
        "            device=self.device,\n"
        "        )\n",
        "decode-only expanded block-table rows",
    )
    replace_once(
        path,
        "        \"\"\"Prepare native or per-token flattened decode tensors.\"\"\"\n"
        "        min_decode_len = int(decode_lens_cpu.min().item())\n",
        "        \"\"\"Prepare native or per-token flattened decode tensors.\"\"\"\n"
        "        assert num_decode_tokens <= self.max_decode_tokens, (\n"
        "            num_decode_tokens,\n"
        "            self.max_decode_tokens,\n"
        "        )\n"
        "        min_decode_len = int(decode_lens_cpu.min().item())\n",
        "flattened decode row bound",
    )


if __name__ == "__main__":
    main()
