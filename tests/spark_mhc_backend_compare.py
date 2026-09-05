#!/usr/bin/env python3
"""Compare vLLM TileLang and B12x mHC against the same FP32 oracle.

This is an SM12x diagnostic.  It must be run in the recipe container on a
DGX Spark; importing the two kernel backends intentionally initializes CUDA.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
import torch.nn.functional as F

from b12x.norm import mhc
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_tilelang,
)


def _mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    *,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = residual.flatten(1).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + hc_eps
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4)
    comb = torch.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    y = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    return y, post, comb


def _mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    return (
        post.unsqueeze(-1) * x.unsqueeze(1).float()
        + (comb.unsqueeze(-1) * residual.unsqueeze(2).float()).sum(dim=1)
    ).to(x.dtype)


def _make_inputs(
    tokens: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    hidden = 4096
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    def normal(shape: tuple[int, ...], divisor: float) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .div_(divisor)
            .to(device)
        )

    residual = normal((tokens, 4, hidden), 3).to(torch.bfloat16).contiguous()
    x = normal((tokens, hidden), 4).to(torch.bfloat16).contiguous()
    fn = normal((24, 4 * hidden), 64).contiguous()
    scale = normal((3,), 3).contiguous()
    bias = normal((24,), 5).contiguous()
    norm_weight = (1 + normal((hidden,), 8)).to(torch.bfloat16).contiguous()
    return residual, x, fn, scale, bias, norm_weight


def _make_binding(
    tokens: int, device: torch.device
) -> tuple[mhc.Plan, mhc.Binding, tuple[torch.Tensor, ...]]:
    hidden = 4096
    plan = mhc.plan(
        mhc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=hidden,
            dtype=torch.bfloat16,
            split_k=mhc.DEFAULT_SPLIT_K,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = mhc.bind(
        plan,
        scratch=scratch,
        tokens=tokens,
        expected_m=tokens,
        y=torch.empty((tokens, hidden), dtype=torch.bfloat16, device=device),
        post=torch.empty((tokens, 4), dtype=torch.float32, device=device),
        comb=torch.empty((tokens, 4, 4), dtype=torch.float32, device=device),
        out=torch.empty(
            (tokens, 4, hidden), dtype=torch.bfloat16, device=device
        ),
    )
    return plan, binding, scratch


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual_f = actual.float().flatten()
    expected_f = expected.float().flatten()
    delta = actual_f - expected_f
    actual_norm = torch.linalg.vector_norm(actual_f)
    expected_norm = torch.linalg.vector_norm(expected_f)
    denom = actual_norm * expected_norm
    cosine = (
        torch.dot(actual_f, expected_f) / denom
        if float(denom) != 0.0
        else torch.tensor(float("nan"), device=actual.device)
    )
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(delta.square().mean())),
        "cosine": float(cosine),
        "exact_fraction": float((actual_f == expected_f).float().mean()),
    }


def _clone_outputs(outputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(output.clone() for output in outputs)


def _repeat_metrics(
    run: Any,
    baseline: tuple[torch.Tensor, ...],
    repeats: int,
) -> list[dict[str, Any]]:
    worst = [
        {"finite": True, "max_abs": 0.0, "exact": True}
        for _ in baseline
    ]
    for _ in range(repeats):
        outputs = run()
        torch.cuda.synchronize()
        for index, (actual, expected) in enumerate(zip(outputs, baseline)):
            delta = (actual.float() - expected.float()).abs()
            worst[index]["finite"] &= bool(torch.isfinite(actual).all())
            worst[index]["max_abs"] = max(
                float(worst[index]["max_abs"]), float(delta.max())
            )
            worst[index]["exact"] &= bool(torch.equal(actual, expected))
    return worst


def _run_shape(tokens: int, seed: int, repeats: int) -> dict[str, Any]:
    device = torch.device("cuda", torch.cuda.current_device())
    residual, x, fn, scale, bias, norm_weight = _make_inputs(
        tokens, seed, device
    )
    rms_eps = 1.0e-6
    hc_eps = 1.0e-6
    norm_eps = 1.0e-6
    sinkhorn_iters = 20

    _, prev_post, prev_comb = _mhc_pre_reference(
        residual,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    prev_post = prev_post.contiguous()
    prev_comb = prev_comb.contiguous()
    residual_ref = _mhc_post_reference(x, residual, prev_post, prev_comb)
    y_raw_ref, post_ref, comb_ref = _mhc_pre_reference(
        residual_ref,
        fn,
        scale,
        bias,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
    )
    rms_scale = torch.rsqrt(
        y_raw_ref.square().mean(dim=-1, keepdim=True) + norm_eps
    )
    y_ref = (
        y_raw_ref.to(torch.bfloat16).float()
        * rms_scale
        * norm_weight.float()
    ).to(torch.bfloat16)
    reference = (residual_ref, post_ref, comb_ref, y_ref)

    def run_tilelang() -> tuple[torch.Tensor, ...]:
        residual_out, post_out, comb_out, y_out = (
            mhc_fused_post_pre_tilelang(
                x,
                residual,
                prev_post,
                prev_comb,
                fn,
                scale,
                bias,
                rms_eps,
                hc_eps,
                hc_eps,
                2.0,
                sinkhorn_iters,
                n_splits=1,
                tile_n=1,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )
        )
        return residual_out, post_out.squeeze(-1), comb_out, y_out

    _, binding, scratch = _make_binding(tokens, device)

    def run_b12x() -> tuple[torch.Tensor, ...]:
        return mhc.run_post_pre(
            x,
            residual,
            prev_post,
            prev_comb,
            fn,
            scale,
            bias,
            rms_eps=rms_eps,
            hc_eps=hc_eps,
            sinkhorn_iters=sinkhorn_iters,
            norm_weight=norm_weight,
            norm_eps=norm_eps,
            binding=binding,
        )

    tilelang = run_tilelang()
    b12x = run_b12x()
    torch.cuda.synchronize()
    tilelang_baseline = _clone_outputs(tilelang)
    b12x_baseline = _clone_outputs(b12x)
    names = ("residual", "post", "comb", "normalized_y")
    result: dict[str, Any] = {
        "tokens": tokens,
        "seed": seed,
        "b12x_scratch_bytes": sum(t.numel() * t.element_size() for t in scratch),
        "tilelang_vs_reference": {
            name: _metrics(actual, expected)
            for name, actual, expected in zip(names, tilelang_baseline, reference)
        },
        "b12x_vs_reference": {
            name: _metrics(actual, expected)
            for name, actual, expected in zip(names, b12x_baseline, reference)
        },
        "tilelang_vs_b12x": {
            name: _metrics(actual, expected)
            for name, actual, expected in zip(
                names, tilelang_baseline, b12x_baseline
            )
        },
        "tilelang_repeat": dict(
            zip(names, _repeat_metrics(run_tilelang, tilelang_baseline, repeats))
        ),
        "b12x_repeat": dict(
            zip(names, _repeat_metrics(run_b12x, b12x_baseline, repeats))
        ),
    }
    del binding, scratch
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        default="1,3,8,16,17,33,96,384",
        help="comma-separated token-row shapes",
    )
    parser.add_argument("--seed", type=int, default=91450)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    tokens = [int(value) for value in args.tokens.split(",") if value]
    if any(value <= 0 for value in tokens):
        raise SystemExit("all token counts must be positive")
    if args.repeats < 0:
        raise SystemExit("--repeats must be non-negative")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run this diagnostic on a DGX Spark")
    capability = torch.cuda.get_device_capability()
    if capability not in {(12, 0), (12, 1)}:
        raise SystemExit(f"SM120/SM121 required, got compute capability {capability}")

    results = {
        "device": torch.cuda.get_device_name(),
        "capability": list(capability),
        "torch": torch.__version__,
        "shapes": [
            _run_shape(value, args.seed + value, args.repeats) for value in tokens
        ],
    }
    print(json.dumps(results, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
