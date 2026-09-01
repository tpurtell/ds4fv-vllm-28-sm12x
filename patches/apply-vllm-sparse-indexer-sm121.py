#!/usr/bin/env python3
"""Avoid the persistent sparse-indexer TopK kernel on SM120-family GPUs."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-sparse-indexer-sm121.py VLLM_ROOT")
    path = (
        Path(sys.argv[1])
        / "model_executor/layers/sparse_attn_indexer.py"
    )
    source = path.read_text()
    old = (
        "        use_persistent_topk = current_platform.is_cuda() and topk_tokens in (\n"
        "            512,\n"
        "            1024,\n"
        "            2048,\n"
        "        )\n"
    )
    new = (
        "        # The persistent radix TopK path can deadlock after repeated\n"
        "        # DSv4 decode launches on GB10/SM121.  The symptom is a later\n"
        "        # multimodal prefill pinned at full GPU utilization while the\n"
        "        # engine reports no running request.  Keep the ordinary\n"
        "        # per-row decoder on the SM120 capability family.\n"
        "        use_persistent_topk = (\n"
        "            current_platform.is_cuda()\n"
        "            and not current_platform.is_device_capability_family(120)\n"
        "            and topk_tokens in (512, 1024, 2048)\n"
        "        )\n"
    )
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: persistent TopK dispatch anchor expected once, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


if __name__ == "__main__":
    main()
