#!/usr/bin/env python3
"""Measure one uniform K2 Trellis layer at DeepSeek V4 Flash geometry."""

from __future__ import annotations

import argparse
import statistics

import torch

from b12x.moe import fused_moe
from benchmark_mixed_trellis_tiles import (
    HIDDEN_SIZE,
    INTERMEDIATE_SIZE,
    NUM_EXPERTS,
    TOP_K,
    _parse_tile,
    _require_spark,
    _timings,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument("--tile", type=_parse_tile, default=(64, 256, 64, 256))
    args = parser.parse_args()
    if args.m <= 32:
        raise ValueError("this benchmark targets packed prefill and requires m > 32")

    device = _require_spark()
    props = torch.cuda.get_device_properties(device)
    torch.manual_seed(20260901)
    bits = 2
    last = 16 * bits

    def payload(shape: tuple[int, ...]) -> torch.Tensor:
        value = torch.empty(shape, dtype=torch.int16, device=device)
        value.random_(-32768, 32768)
        return value

    w13 = payload(
        (
            2,
            NUM_EXPERTS,
            HIDDEN_SIZE // 16,
            INTERMEDIATE_SIZE // 16,
            last,
        )
    )
    w2 = payload(
        (
            NUM_EXPERTS,
            INTERMEDIATE_SIZE // 16,
            HIDDEN_SIZE // 16,
            last,
        )
    )
    gate_suh = torch.ones(
        (NUM_EXPERTS, HIDDEN_SIZE), dtype=torch.float16, device=device
    )
    up_suh = torch.ones_like(gate_suh)
    intermediate_rotations = torch.ones(
        (NUM_EXPERTS, 3 * INTERMEDIATE_SIZE),
        dtype=torch.float16,
        device=device,
    )
    down_svh = torch.ones_like(gate_suh)
    x = (
        torch.randn((args.m, HIDDEN_SIZE), dtype=torch.float32, device=device)
        * 1.0e-3
    ).to(torch.bfloat16)
    row = torch.arange(args.m, dtype=torch.int64, device=device).unsqueeze(1)
    offsets = torch.tensor(
        (0, 41, 83, 127, 173, 229), dtype=torch.int64, device=device
    ).unsqueeze(0)
    topk_ids = (row * 17 + offsets).remainder(NUM_EXPERTS)
    topk_weights = torch.softmax(
        torch.randn((args.m, TOP_K), dtype=torch.float32, device=device), dim=-1
    )

    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="exl3_trellis_mcg",
        activation="silu",
        params_dtype=torch.float16,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        w13_layout="w13",
        trellis_bits=bits,
        trellis_tile_config=args.tile,
        trellis_codebook="mcg",
    )
    experts = fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.float16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        trellis_mcg=0xCBAC1FED,
    )
    caps = fused_moe.Caps(
        max_tokens=args.m,
        num_topk=TOP_K,
        route_num_experts=0,
        device=device,
        weight_plan=experts.plan,
        quant_mode="w4a16",
        w4a16_block_size_m=args.block_m,
        swiglu_limit=10.0,
        full_rotation_output_dtype=torch.bfloat16,
    )
    plan = fused_moe.plan(caps)
    launch = plan._prewarmed_fused_launches[0][1]
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(
        scratch_spec.shape,
        dtype=scratch_spec.dtype,
        device=scratch_spec.device,
    )

    def invoke() -> torch.Tensor:
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        return fused_moe.run(binding=binding)

    for _ in range(args.warmup):
        output = invoke()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = invoke()
    graph.replay()
    torch.cuda.synchronize(device)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("uniform K2 produced a non-finite output")
    measured = _timings(graph, args.replays)
    print(
        f"device={props.name} sm={props.major}{props.minor} m={args.m} "
        f"H={HIDDEN_SIZE} N={INTERMEDIATE_SIZE} E={NUM_EXPERTS} topk={TOP_K} "
        f"uniform=K2 tile={args.tile} block_m={args.block_m} "
        f"median={statistics.median(measured):.2f}us "
        f"min={min(measured):.2f}us blocks_per_sm={launch.blocks_per_sm} "
        f"threads={launch.cta_threads} smem={launch.shared_memory_bytes}B scratch="
        f"{scratch.numel() * scratch.element_size() / float(1 << 20):.1f}MiB"
    )


if __name__ == "__main__":
    main()
