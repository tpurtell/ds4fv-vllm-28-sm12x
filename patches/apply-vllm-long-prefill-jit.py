#!/usr/bin/env python3
"""Warm the generic non-zero DeepSeek sparse-indexer query-slice class."""

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
        raise SystemExit("usage: apply-vllm-long-prefill-jit.py VLLM_ROOT")
    path = (
        Path(sys.argv[1])
        / "v1/attention/backends/mla/indexer.py"
    )
    replace_once(
        path,
        "            query_slice_start=WarmupIntRange(0, 2),\n"
        "            query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),\n",
        "            # Triton specializes scalar integers into zero/divisible,\n"
        "            # one, and ordinary non-zero classes. Query slicing above\n"
        "            # the logits budget is the first runtime path to use the\n"
        "            # ordinary class, so include representative value 2.\n"
        "            query_slice_start=WarmupIntRange(0, 3),\n"
        "            query_slice_stop=(1, 2 * max_tokens - 1, 2 * max_tokens),\n",
        "generic non-zero prefill query-slice warmup",
    )


if __name__ == "__main__":
    main()
