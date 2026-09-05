#!/usr/bin/env python3
"""Remove the per-step CUDA scalar construction from vLLM DCP length math."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dcp-rank-scalar.py VLLM_ROOT")
    path = (
        Path(sys.argv[1]).resolve()
        / "v1/attention/backends/utils.py"
    )
    source = path.read_text()
    old = '''    else:
        rank_offsets = torch.tensor(dcp_rank, dtype=torch.int32, device=seq_lens.device)
        seq_lens_tiled = seq_lens_i32
'''
    new = '''    else:
        # Keep the known rank as a scalar. Constructing a fresh CUDA tensor
        # from this Python integer performs a pageable host-to-device copy;
        # that copy synchronizes the current stream once per decode step.
        rank_offsets = dcp_rank
        seq_lens_tiled = seq_lens_i32
'''
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one rank-scalar anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


if __name__ == "__main__":
    main()
