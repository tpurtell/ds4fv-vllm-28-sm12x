#!/usr/bin/env python3
"""Measure the three NCCL collectives in one DeepSeek-V4 DCP attention pass."""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


LAYERS = 43
LOCAL_HEADS = 32
GLOBAL_HEADS = 64
HEAD_DIM = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29876)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("run this benchmark only on the DGX Sparks")
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=2,
    )

    results: list[dict[str, float | int]] = []
    for concurrency in (1, 2, 4):
        # Greedy K3 verifies four target rows per live sequence.
        rows = 4 * concurrency
        query = torch.zeros(
            (rows, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )
        gathered_query = torch.empty(
            (2 * rows, LOCAL_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        partial_lse = torch.zeros(
            (rows, GLOBAL_HEADS), dtype=torch.float32, device="cuda"
        )
        gathered_lse = torch.empty(
            (2 * rows, GLOBAL_HEADS), dtype=torch.float32, device="cuda"
        )
        partial_output = torch.zeros(
            (2 * rows, LOCAL_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reduced_output = torch.empty(
            (rows, LOCAL_HEADS, HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )

        def one_model_pass() -> None:
            for _ in range(LAYERS):
                dist.all_gather_into_tensor(gathered_query, query)
                dist.all_gather_into_tensor(gathered_lse, partial_lse)
                dist.reduce_scatter_tensor(reduced_output, partial_output)

        def measure_component(operation) -> float:
            for _ in range(args.warmups):
                for _ in range(LAYERS):
                    operation()
            torch.cuda.synchronize()
            dist.barrier()
            component_started = time.perf_counter()
            for _ in range(args.iterations):
                for _ in range(LAYERS):
                    operation()
            torch.cuda.synchronize()
            dist.barrier()
            return (
                (time.perf_counter() - component_started)
                * 1000.0
                / args.iterations
            )

        for _ in range(args.warmups):
            one_model_pass()
        torch.cuda.synchronize()
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.iterations):
            one_model_pass()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - started
        milliseconds = elapsed * 1000.0 / args.iterations
        query_gather_ms = measure_component(
            lambda: dist.all_gather_into_tensor(gathered_query, query)
        )
        lse_gather_ms = measure_component(
            lambda: dist.all_gather_into_tensor(gathered_lse, partial_lse)
        )
        output_reduce_scatter_ms = measure_component(
            lambda: dist.reduce_scatter_tensor(reduced_output, partial_output)
        )
        results.append(
            {
                "concurrency": concurrency,
                "rows": rows,
                "model_pass_ms": milliseconds,
                "layer_triplet_us": milliseconds * 1000.0 / LAYERS,
                "query_gather_ms": query_gather_ms,
                "lse_gather_ms": lse_gather_ms,
                "output_reduce_scatter_ms": output_reduce_scatter_ms,
                "component_sum_ms": (
                    query_gather_ms + lse_gather_ms + output_reduce_scatter_ms
                ),
            }
        )

    if args.rank == 0:
        print(
            json.dumps(
                {
                    "schema": "ds4fv-dcp-collectives.v1",
                    "nccl_algo": os.getenv("NCCL_ALGO", "auto"),
                    "nccl_proto": os.getenv("NCCL_PROTO", "auto"),
                    "layers": LAYERS,
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
