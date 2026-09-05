#!/usr/bin/env python3
"""Measure DeepSeek-V4's exact-global C4 top-K DCP merge on two Sparks."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
    pack_dcp_topk_candidates_cutedsl,
    stable_topk_from_gathered_candidates_cutedsl,
)


LAYERS = 20
TOPK = 512
WORLD_SIZE = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29878)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--valid-candidates", type=int, default=35)
    return parser.parse_args()


def all_gather_dim1(tensor: torch.Tensor) -> torch.Tensor:
    """Mirror CudaCommunicator.all_gather(..., dim=1), including its copy."""
    input_size = tensor.shape
    rank_major = torch.empty(
        (WORLD_SIZE * input_size[0], *input_size[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    dist.all_gather_into_tensor(rank_major, tensor)
    return (
        rank_major.reshape(WORLD_SIZE, *input_size)
        .movedim(0, 1)
        .reshape(input_size[0], WORLD_SIZE * input_size[1], *input_size[2:])
    )


def measure(operation, warmups: int, iterations: int) -> float:
    for _ in range(warmups):
        for _ in range(LAYERS):
            operation()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        for _ in range(LAYERS):
            operation()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("run this benchmark only on the DGX Sparks")
    if not 0 < args.valid_candidates <= TOPK:
        raise ValueError(f"valid candidates must be in [1, {TOPK}]")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError("this benchmark requires a DGX Spark SM121 GPU")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=WORLD_SIZE,
    )

    results: list[dict[str, float | int]] = []
    for concurrency in (1, 2, 4):
        rows = 4 * concurrency
        logits = torch.randn((rows, TOPK), dtype=torch.float32, device="cuda")
        topk_indices = torch.full(
            (rows, TOPK), -1, dtype=torch.int32, device="cuda"
        )
        topk_indices[:, : args.valid_candidates] = torch.arange(
            args.valid_candidates, dtype=torch.int32, device="cuda"
        )
        packed = torch.empty((rows, TOPK, 2), dtype=torch.float32, device="cuda")
        gathered = torch.empty(
            (rows, WORLD_SIZE * TOPK, 2), dtype=torch.float32, device="cuda"
        )
        output = torch.empty((rows, TOPK), dtype=torch.int32, device="cuda")

        def pack() -> None:
            pack_dcp_topk_candidates_cutedsl(
                logits,
                topk_indices,
                packed,
                args.rank,
                WORLD_SIZE,
                1,
                None,
            )

        def gather() -> None:
            gathered.copy_(all_gather_dim1(packed))

        def select() -> None:
            stable_topk_from_gathered_candidates_cutedsl(
                gathered, TOPK, out=output
            )

        def merge() -> None:
            local_packed = torch.empty_like(packed)
            pack_dcp_topk_candidates_cutedsl(
                logits,
                topk_indices,
                local_packed,
                args.rank,
                WORLD_SIZE,
                1,
                None,
            )
            global_candidates = all_gather_dim1(local_packed)
            stable_topk_from_gathered_candidates_cutedsl(
                global_candidates, TOPK, out=output
            )

        pack()
        gathered.copy_(all_gather_dim1(packed))
        select()
        torch.cuda.synchronize()
        results.append(
            {
                "concurrency": concurrency,
                "rows": rows,
                "valid_candidates_per_rank": args.valid_candidates,
                "pack_ms": measure(pack, args.warmups, args.iterations),
                "gather_ms": measure(gather, args.warmups, args.iterations),
                "stable_topk_ms": measure(select, args.warmups, args.iterations),
                "full_merge_ms": measure(merge, args.warmups, args.iterations),
            }
        )

    if args.rank == 0:
        print(
            json.dumps(
                {
                    "schema": "ds4fv-dcp-indexer-merge.v1",
                    "layers": LAYERS,
                    "topk": TOPK,
                    "world_size": WORLD_SIZE,
                    "warmups": args.warmups,
                    "iterations": args.iterations,
                    "results": results,
                },
                indent=2,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
