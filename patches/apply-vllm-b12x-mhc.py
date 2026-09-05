#!/usr/bin/env python3
"""Add an opt-in B12x decode mHC path to DeepSeek V4.

The B12x functional operation owns its result allocation and has no mutated
arguments.  That is the capture/torch.compile-safe lifecycle intended for an
integration that does not reserve graph-shape-specific serving buffers.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-b12x-mhc.py VLLM_ROOT")
    root = Path(sys.argv[1]).resolve()
    model = root / "models/deepseek_v4/nvidia/model.py"
    source = model.read_text()

    source = replace_once(
        source,
        "import typing\n",
        "import os\nimport typing\n",
        "model imports",
    )

    anchor = "\n\nclass DeepseekV4MLP(nn.Module):\n"
    helper = r'''

_DS4FV_USE_B12X_MHC_DECODE = os.environ.get(
    "DS4FV_USE_B12X_MHC_DECODE", "0"
).strip().lower() in {"1", "true", "yes", "on"}
_DS4FV_B12X_MHC_MAX_TOKENS = int(
    os.environ.get("DS4FV_B12X_MHC_MAX_TOKENS", "16")
)
if _DS4FV_B12X_MHC_MAX_TOKENS < 1:
    raise ValueError("DS4FV_B12X_MHC_MAX_TOKENS must be positive")


def _deepseek_v4_mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    *,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = int(residual.shape[0])
    use_b12x = (
        _DS4FV_USE_B12X_MHC_DECODE
        and tokens <= _DS4FV_B12X_MHC_MAX_TOKENS
    )
    if not use_b12x:
        return mhc_fused_post_pre_tilelang(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits=n_splits,
            tile_n=tile_n,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
        )

    if hc_pre_eps != hc_sinkhorn_eps:
        raise ValueError("B12x mHC requires matching pre and Sinkhorn epsilons")
    if hc_post_mult_value != 2.0:
        raise ValueError("B12x mHC requires hc_post_mult_value=2.0")
    from b12x.norm import mhc

    residual_out, post_out, comb_out, y_out = mhc.run_post_pre(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        rms_eps=rms_eps,
        hc_eps=hc_pre_eps,
        sinkhorn_iters=sinkhorn_repeat,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    # Preserve vLLM's established [tokens, hc_mult, 1] state ABI.
    return residual_out, post_out.unsqueeze(-1), comb_out, y_out
'''
    source = replace_once(source, anchor, helper + anchor, "mHC helper anchor")
    call = "residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang("
    count = source.count(call)
    if count != 2:
        raise RuntimeError(f"mHC call sites: expected two matches, found {count}")
    source = source.replace(
        call,
        "residual, post_mix, res_mix, x = _deepseek_v4_mhc_fused_post_pre(",
    )
    model.write_text(source)


if __name__ == "__main__":
    main()
