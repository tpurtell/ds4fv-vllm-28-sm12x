#!/usr/bin/env python3
"""Prevent unsafe auxiliary-stream overlap with B12x MoE kernels.

The vLLM B12x adapter conservatively declares that its execution plans do not
support auxiliary-stream overlap: resident-grid plans can use device-wide
barriers.  The generic MoE runner did not consume that restriction and could
run the dense shared expert concurrently with B12x, eventually corrupting the
CUDA context.  Keep the shared expert on the caller stream only for B12x; all
other backends retain vLLM's normal overlap policy.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1])

    shared = root / "model_executor/layers/fused_moe/runner/shared_experts.py"
    replace_once(
        shared,
        '''        mk_can_overlap_shared_experts: Callable[[], bool],
    ):
''',
        '''        mk_can_overlap_shared_experts: Callable[[], bool],
        disable_aux_stream_overlap: bool = False,
    ):
''',
        "backend overlap capability argument",
    )
    replace_once(
        shared,
        '''        if envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM:
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
''',
        '''        if (
            envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
            or disable_aux_stream_overlap
        ):
            logger.debug_once("Disabling MoE shared_experts cuda stream")
            self._stream = None
''',
        "backend-aware shared-expert stream policy",
    )

    runner = root / "model_executor/layers/fused_moe/runner/moe_runner.py"
    replace_once(
        runner,
        '''                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
            )
''',
        '''                enable_dbo=enable_dbo,
                mk_can_overlap_shared_experts=can_overlap,
                # B12x resident-grid plans can use device-wide barriers and
                # therefore must not overlap the dense shared expert on a
                # second CUDA stream.
                disable_aux_stream_overlap=self._uses_b12x_moe_kernel,
            )
''',
        "B12x shared-expert overlap policy",
    )

    print("Applied B12x shared-expert stream-safety patch")


if __name__ == "__main__":
    main()
