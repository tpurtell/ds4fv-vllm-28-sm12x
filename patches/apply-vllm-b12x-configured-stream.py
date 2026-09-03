#!/usr/bin/env python3
"""Make B12x shared-stream detection valid before lazy MoE initialization."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    path = (
        Path(sys.argv[1])
        / "model_executor/layers/fused_moe/runner/moe_runner.py"
    )
    source = path.read_text()
    old = '''    @property
    def _uses_b12x_moe_kernel(self) -> bool:
        moe_kernel = getattr(self._quant_method, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        if fused_experts is None:
            return False

        from vllm.model_executor.layers.fused_moe.b12x_moe import (
            B12xExperts,
        )

        return isinstance(fused_experts, B12xExperts)
'''
    new = '''    @property
    def _uses_b12x_moe_kernel(self) -> bool:
        # MoE kernels may finish lazy initialization after MoERunner constructs
        # SharedExperts. The configured backend is already authoritative then.
        kernel_config = getattr(get_current_vllm_config(), "kernel_config", None)
        if getattr(kernel_config, "moe_backend", None) == "b12x":
            return True

        moe_kernel = getattr(self._quant_method, "moe_kernel", None)
        fused_experts = getattr(moe_kernel, "fused_experts", None)
        if fused_experts is None:
            return False

        from vllm.model_executor.layers.fused_moe.b12x_moe import (
            B12xExperts,
        )

        return isinstance(fused_experts, B12xExperts)
'''
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one B12x property, found {count}")
    path.write_text(source.replace(old, new, 1))
    print("Applied configured-backend B12x stream detection patch")


if __name__ == "__main__":
    main()
