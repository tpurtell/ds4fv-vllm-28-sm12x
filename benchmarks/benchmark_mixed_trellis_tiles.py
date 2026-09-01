#!/usr/bin/env python3
"""Sweep one-grid K2/K3 Trellis tiles at DeepSeek V4 Flash geometry.

This is a kernel qualification benchmark, not a release benchmark.  It must
run inside the recipe image on a DGX Spark; the architecture checks deliberately
reject the local RTX/SM120 development host.
"""

from __future__ import annotations

import argparse
import platform
import statistics
from dataclasses import dataclass

import torch

from b12x.moe import fused_moe
from b12x.moe.fused_moe.trellis import ProjectionTrellisTierWeights


HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 2048
NUM_EXPERTS = 256
TOP_K = 6
BASE_LAYER_COUNTS = {
    2: (242, 232, 218),
    3: (14, 24, 38),
}
DEFAULT_TILES = (
    (128, 128, 128, 128),
    (128, 128, 64, 256),
    (64, 256, 128, 128),
    (64, 256, 64, 256),
    (64, 128, 64, 128),
    (64, 128, 128, 64),
    (32, 256, 64, 128),
    (32, 256, 32, 256),
    (128, 64, 64, 128),
    (128, 64, 128, 64),
    (64, 256, 64, 256),
    (128, 128, 128, 128),
)


@dataclass(frozen=True)
class Result:
    tile: tuple[int, int, int, int]
    median_us: float
    minimum_us: float
    scratch_mib: float
    relative_error: float
    cosine: float


def _parse_tile(value: str) -> tuple[int, int, int, int]:
    try:
        tile = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid tile {value!r}") from exc
    if len(tile) != 4 or any(part <= 0 for part in tile):
        raise argparse.ArgumentTypeError(
            "tiles must be four positive comma-separated integers"
        )
    return tile  # type: ignore[return-value]


def _require_spark() -> torch.device:
    if platform.machine() != "aarch64":
        raise RuntimeError(
            "this GPU benchmark is Spark-only and refuses non-aarch64 hosts"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(device)
    if props.major != 12 or props.minor != 1 or "GB10" not in props.name:
        raise RuntimeError(
            "this benchmark requires DGX Spark GB10/SM121; "
            f"got {props.name} SM{props.major}{props.minor}"
        )
    return device


def _projection_ids(bits: int, projection: int) -> tuple[int, ...]:
    """Build complementary K2/K3 expert maps with production slot counts."""

    k3_count = BASE_LAYER_COUNTS[3][projection]
    return (
        tuple(range(k3_count, NUM_EXPERTS))
        if bits == 2
        else tuple(range(k3_count))
    )


def _make_tier(bits: int, device: torch.device) -> ProjectionTrellisTierWeights:
    gate_count, up_count, down_count = BASE_LAYER_COUNTS[bits]
    last = 16 * int(bits)

    def payload(shape: tuple[int, ...]) -> torch.Tensor:
        value = torch.empty(shape, dtype=torch.int16, device=device)
        value.random_(-32768, 32768)
        return value

    return ProjectionTrellisTierWeights(
        bits=bits,
        w13=payload(
            (
                gate_count + up_count,
                HIDDEN_SIZE // 16,
                INTERMEDIATE_SIZE // 16,
                last,
            )
        ),
        w2=payload(
            (
                down_count,
                INTERMEDIATE_SIZE // 16,
                HIDDEN_SIZE // 16,
                last,
            )
        ),
        gate_experts=_projection_ids(bits, 0),
        up_experts=_projection_ids(bits, 1),
        down_experts=_projection_ids(bits, 2),
    )


def _timings(graph: torch.cuda.CUDAGraph, replays: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(replays)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end) * 1000.0
        for start, end in zip(starts, ends, strict=True)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--replays", type=int, default=12)
    parser.add_argument(
        "--route-mode",
        choices=("balanced", "k2", "k3"),
        default="balanced",
    )
    parser.add_argument(
        "--tile",
        type=_parse_tile,
        action="append",
        dest="tiles",
        help="FC1_K,FC1_N,FC2_K,FC2_N; repeat to compare candidates",
    )
    args = parser.parse_args()
    if args.m <= 32:
        raise ValueError("this sweep targets packed prefill and requires m > 32")
    if args.warmup < 1 or args.replays < 3:
        raise ValueError("warmup must be >=1 and replays must be >=3")

    device = _require_spark()
    props = torch.cuda.get_device_properties(device)
    tiles = tuple(args.tiles or DEFAULT_TILES)
    torch.manual_seed(20260901)

    tiers = (_make_tier(2, device), _make_tier(3, device))
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
    if args.route_mode == "balanced":
        offsets = torch.tensor(
            (0, 41, 83, 127, 173, 229), dtype=torch.int64, device=device
        ).unsqueeze(0)
        topk_ids = (row * 17 + offsets).remainder(NUM_EXPERTS)
    elif args.route_mode == "k2":
        offsets = torch.tensor(
            (0, 31, 63, 95, 127, 159), dtype=torch.int64, device=device
        ).unsqueeze(0)
        topk_ids = (row * 17 + offsets).remainder(192).add_(64)
    else:
        offsets = torch.tensor(
            (0, 1, 3, 5, 7, 11), dtype=torch.int64, device=device
        ).unsqueeze(0)
        topk_ids = (row * 7 + offsets).remainder(14)
    topk_weights = torch.softmax(
        torch.randn((args.m, TOP_K), dtype=torch.float32, device=device), dim=-1
    )

    print(
        f"device={props.name} sm={props.major}{props.minor} m={args.m} "
        f"H={HIDDEN_SIZE} N={INTERMEDIATE_SIZE} E={NUM_EXPERTS} topk={TOP_K} "
        f"tiers={BASE_LAYER_COUNTS} routes={args.route_mode} block_m={args.block_m}"
    )
    baseline: torch.Tensor | None = None
    baseline_norm = 0.0
    results: list[Result] = []

    for index, tile in enumerate(tiles):
        print(f"planning tile={tile}", flush=True)
        weight_plan = fused_moe.plan_weights(
            quant_modes="w4a16",
            source_format="exl3_trellis_mcg",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=NUM_EXPERTS,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            w13_layout="w13",
            trellis_bits=2,
            trellis_tile_config=tile,
            trellis_codebook="mcg",
            trellis_rate_granularity="per_expert_projection",
        )
        experts = fused_moe.prepare_weights(
            plan=weight_plan,
            params_dtype=torch.bfloat16,
            projection_tiers=tiers,
            gate_suh=gate_suh,
            up_suh=up_suh,
            intermediate_rotations=intermediate_rotations,
            down_svh=down_svh,
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
            mixed_trellis_route_id_dtypes=(torch.int64,),
            mixed_trellis_broadcast_suh=(False,),
            mixed_trellis_broadcast_svh=(False,),
        )
        try:
            plan = fused_moe.plan(caps)
        except ValueError as exc:
            print(f"rejected tile={tile}: {exc}", flush=True)
            continue
        launch = plan._mixed_trellis_launches[0][4]
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
            raise RuntimeError(f"tile {tile} produced a non-finite output")

        measured = _timings(graph, args.replays)
        candidate = output.clone()
        torch.cuda.synchronize(device)
        if baseline is None:
            baseline = candidate
            baseline_norm = float(baseline.float().norm().clamp_min(1.0e-12))
            relative_error = 0.0
            cosine = 1.0
        else:
            difference = (candidate.float() - baseline.float()).norm()
            relative_error = float(difference) / baseline_norm
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    candidate.flatten().float(), baseline.flatten().float(), dim=0
                )
            )
            if relative_error > 2.0e-2 or cosine < 0.999:
                raise RuntimeError(
                    f"tile {tile} failed cross-tile correctness: "
                    f"relative_error={relative_error:.6g}, cosine={cosine:.8f}"
                )

        result = Result(
            tile=tile,
            median_us=statistics.median(measured),
            minimum_us=min(measured),
            scratch_mib=(
                scratch.numel() * scratch.element_size() / float(1 << 20)
            ),
            relative_error=relative_error,
            cosine=cosine,
        )
        results.append(result)
        print(
            f"tile={tile} median={result.median_us:.2f}us "
            f"min={result.minimum_us:.2f}us scratch={result.scratch_mib:.1f}MiB "
            f"blocks_per_sm={launch.blocks_per_sm} "
            f"smem={launch.shared_memory_bytes}B "
            f"rel={result.relative_error:.3e} cos={result.cosine:.8f}",
            flush=True,
        )
        del graph, output, candidate, scratch, plan, experts, weight_plan
        torch.cuda.empty_cache()

    best = min(results, key=lambda result: result.median_us)
    first = results[0]
    print(
        f"best={best.tile} median={best.median_us:.2f}us "
        f"vs_first={(best.median_us / first.median_us - 1.0) * 100.0:+.2f}%"
    )


if __name__ == "__main__":
    main()
