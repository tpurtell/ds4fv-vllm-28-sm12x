#!/usr/bin/env python3
"""Compare B12x and vLLM TileLang mHC at DeepSeek V4 shapes.

This is a Spark-only GPU microbenchmark.  Run it inside the recipe image;
never run it on the build workstation.
"""

from __future__ import annotations

import argparse
import json

import torch
from b12x.norm import mhc as b12x_mhc
from vllm.model_executor.kernels.mhc.tilelang import (
    mhc_fused_post_pre_tilelang,
)


def make_inputs(tokens: int, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = torch.device("cuda:0")
    x = torch.randn((tokens, 4096), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    residual = torch.randn((tokens, 4, 4096), generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    post = torch.randn((tokens, 4, 1), generator=generator).to(
        device=device, dtype=torch.float32
    )
    comb = torch.softmax(
        torch.randn((tokens, 4, 4), generator=generator).to(device), dim=-1
    )
    fn = (
        torch.randn((24, 4 * 4096), generator=generator).to(device) / 64
    ).contiguous()
    scale = torch.randn((3,), generator=generator).to(device).contiguous()
    bias = torch.randn((24,), generator=generator).to(device).contiguous()
    norm_weight = torch.ones((4096,), dtype=torch.bfloat16, device=device)
    return x, residual, post, comb, fn, scale, bias, norm_weight


def run_b12x(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    x, residual, post, comb, fn, scale, bias, norm_weight = inputs
    residual_out, post_out, comb_out, y = b12x_mhc.run_post_pre(
        x,
        residual,
        post.squeeze(-1),
        comb,
        fn,
        scale,
        bias,
        rms_eps=1.0e-5,
        hc_eps=1.0e-6,
        sinkhorn_iters=20,
        norm_weight=norm_weight,
        norm_eps=1.0e-5,
    )
    return residual_out, post_out.unsqueeze(-1), comb_out, y


def run_tilelang(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    x, residual, post, comb, fn, scale, bias, norm_weight = inputs
    return mhc_fused_post_pre_tilelang(
        x,
        residual,
        post,
        comb,
        fn,
        scale,
        bias,
        1.0e-5,
        1.0e-6,
        1.0e-6,
        2.0,
        20,
        1,
        1,
        norm_weight,
        1.0e-5,
    )


def elapsed_us(function, inputs: tuple[torch.Tensor, ...], iterations: int) -> float:
    for _ in range(10):
        function(inputs)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function(inputs)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 2, 4, 7, 8, 16))
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise SystemExit(f"requires DGX Spark SM121, got SM{capability[0]}{capability[1]}")

    results = []
    for tokens in args.tokens:
        inputs = make_inputs(tokens, 202_609_01 + tokens)
        b12x_outputs = run_b12x(inputs)
        tilelang_outputs = run_tilelang(inputs)
        torch.cuda.synchronize()
        # B12x and TileLang use different reduction orders.  Keep the same
        # decode-oracle bounds used by the GLM integration, with 0.5e-3 of
        # headroom for the fp32 post/comb reductions, and report the observed
        # deltas so a passing tolerance cannot hide numerical drift.
        tolerances = (2.0e-2, 2.5e-3, 2.5e-3, 1.6e-2)
        max_abs_diffs = []
        for actual, expected, atol in zip(
            b12x_outputs, tilelang_outputs, tolerances, strict=True
        ):
            max_abs_diffs.append(float((actual.float() - expected.float()).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=atol)

        b12x_us = elapsed_us(run_b12x, inputs, args.iterations)
        tilelang_us = elapsed_us(run_tilelang, inputs, args.iterations)
        results.append(
            {
                "tokens": tokens,
                "b12x_us": b12x_us,
                "tilelang_us": tilelang_us,
                "b12x_speedup": tilelang_us / b12x_us,
                "max_abs_diffs": max_abs_diffs,
            }
        )

    print(
        json.dumps(
            {
                "schema": "deepseek-v4-mhc-b12x-vs-tilelang.v1",
                "device_capability": list(capability),
                "iterations": args.iterations,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
