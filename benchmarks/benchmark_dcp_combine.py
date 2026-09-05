#!/usr/bin/env python3
"""Compare vLLM's generic DCP AG/RS combine with a packed-output variant."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.distributed as dist

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import CPTritonContext, correct_attn_out


LAYERS = 43
LOCAL_HEADS = 32
GLOBAL_HEADS = 64
HEAD_DIM = 512
WORLD_SIZE = 2


@triton.jit
def _correct_and_pack_kernel(
    partial_output_ptr,
    gathered_lse_ptr,
    packed_output_ptr,
    output_stride_token,
    output_stride_head,
    output_stride_dim,
    lse_stride_rank,
    lse_stride_token,
    lse_stride_head,
    rank,
    num_tokens,
    HEAD_DIM: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim = tl.arange(0, HEAD_DIM)
    peer = tl.arange(0, WORLD_SIZE)

    lse_offsets = (
        peer * lse_stride_rank
        + token * lse_stride_token
        + head * lse_stride_head
    )
    lses = tl.load(gathered_lse_ptr + lse_offsets).to(tl.float32)
    lses = tl.where(
        (lses != lses) | (lses == float("inf")), -float("inf"), lses
    )
    lse_max = tl.max(lses, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    shifted = lses - lse_max
    global_lse = tl.log2(tl.sum(tl.exp2(shifted), axis=0)) + lse_max
    local_lse = tl.load(
        gathered_lse_ptr
        + rank * lse_stride_rank
        + token * lse_stride_token
        + head * lse_stride_head
    ).to(tl.float32)
    correction = local_lse - global_lse
    correction = tl.where(
        (correction != correction) | (correction == float("inf")),
        -float("inf"),
        correction,
    )
    factor = tl.exp2(correction)

    source_offsets = (
        token * output_stride_token
        + head * output_stride_head
        + dim * output_stride_dim
    )
    values = tl.load(partial_output_ptr + source_offsets) * factor
    values = tl.where(factor == 0.0, 0.0, values)

    destination_rank = head // LOCAL_HEADS
    local_head = head - destination_rank * LOCAL_HEADS
    packed_offsets = (
        destination_rank * num_tokens * LOCAL_HEADS * HEAD_DIM
        + token * LOCAL_HEADS * HEAD_DIM
        + local_head * HEAD_DIM
        + dim
    )
    tl.store(packed_output_ptr + packed_offsets, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29877)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def gather_lse(partial_lse: torch.Tensor) -> torch.Tensor:
    gathered = torch.empty(
        (WORLD_SIZE * partial_lse.shape[0], partial_lse.shape[1]),
        dtype=partial_lse.dtype,
        device=partial_lse.device,
    )
    dist.all_gather_into_tensor(gathered, partial_lse)
    return gathered.view(WORLD_SIZE, *partial_lse.shape)


def current_query_gather(local_query: torch.Tensor) -> torch.Tensor:
    """Mirror CudaCommunicator.all_gather(dim=1), including its final copy."""
    rows, local_heads, head_dim = local_query.shape
    gathered = torch.empty(
        (WORLD_SIZE * rows, local_heads, head_dim),
        dtype=local_query.dtype,
        device=local_query.device,
    )
    dist.all_gather_into_tensor(gathered, local_query.contiguous())
    return (
        gathered.view(WORLD_SIZE, rows, local_heads, head_dim)
        .movedim(0, 1)
        .reshape(rows, WORLD_SIZE * local_heads, head_dim)
    )


def strided_query_gather(local_query: torch.Tensor) -> torch.Tensor:
    """Gather head-major and return a zero-copy token/head transpose view."""
    rows, local_heads, head_dim = local_query.shape
    head_major = local_query.movedim(0, 1).contiguous()
    gathered = torch.empty(
        (WORLD_SIZE * local_heads, rows, head_dim),
        dtype=local_query.dtype,
        device=local_query.device,
    )
    dist.all_gather_into_tensor(gathered, head_major)
    return gathered.movedim(0, 1)


def generic_combine(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    destination: torch.Tensor,
    rank: int,
    context: CPTritonContext,
) -> None:
    gathered_lse = gather_lse(partial_lse)
    corrected, _ = correct_attn_out(
        partial_output,
        gathered_lse,
        rank,
        context,
        is_lse_base_on_e=False,
    )
    rank_major = corrected.movedim(0, 1).contiguous()
    reduced = torch.empty(
        (LOCAL_HEADS, partial_output.shape[0], HEAD_DIM),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    dist.reduce_scatter_tensor(reduced, rank_major)
    destination.copy_(reduced.movedim(0, 1).contiguous())


def packed_combine(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    destination: torch.Tensor,
    rank: int,
) -> None:
    gathered_lse = gather_lse(partial_lse)
    num_tokens = partial_output.shape[0]
    packed = torch.empty(
        (WORLD_SIZE * num_tokens, LOCAL_HEADS, HEAD_DIM),
        dtype=partial_output.dtype,
        device=partial_output.device,
    )
    _correct_and_pack_kernel[(num_tokens, GLOBAL_HEADS)](
        partial_output,
        gathered_lse,
        packed,
        *partial_output.stride(),
        *gathered_lse.stride(),
        rank,
        num_tokens,
        HEAD_DIM=HEAD_DIM,
        LOCAL_HEADS=LOCAL_HEADS,
        WORLD_SIZE=WORLD_SIZE,
    )
    dist.reduce_scatter_tensor(destination, packed)


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
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=args.rank,
        world_size=WORLD_SIZE,
    )

    results: list[dict[str, float | int]] = []
    for concurrency in (1, 2, 4):
        # Greedy K3 runs one target pass over four rows per live sequence.
        rows = 4 * concurrency
        generator = torch.Generator(device="cuda").manual_seed(1000 + args.rank)
        correctness_output = torch.randn(
            (rows, GLOBAL_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        partial_lse = torch.randn(
            (rows, GLOBAL_HEADS),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        local_query = torch.randn(
            (rows, LOCAL_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        current_query = current_query_gather(local_query)
        strided_query = strided_query_gather(local_query)
        if not current_query.is_contiguous() or strided_query.is_contiguous():
            raise AssertionError(
                "query gather layouts do not match the expected copy/view contract"
            )
        if not torch.equal(current_query, strided_query):
            raise AssertionError("strided query gather changed global head ordering")
        generic_result = torch.empty(
            (rows, LOCAL_HEADS, HEAD_DIM),
            dtype=torch.bfloat16,
            device="cuda",
        )
        packed_result = torch.empty_like(generic_result)
        generic_combine(
            correctness_output.clone(),
            partial_lse,
            generic_result,
            args.rank,
            CPTritonContext(),
        )
        packed_combine(
            correctness_output.clone(), partial_lse, packed_result, args.rank
        )
        torch.cuda.synchronize()
        max_abs_error = float(
            (generic_result.float() - packed_result.float()).abs().max().item()
        )
        if not torch.equal(generic_result, packed_result):
            raise AssertionError(
                f"packed combine differs from generic combine: {max_abs_error=}"
            )

        # Zero is stable under repeated in-place correction and keeps the timed
        # path focused on combine mechanics rather than regenerating inputs.
        timed_output = torch.zeros_like(correctness_output)
        destination = torch.empty_like(generic_result)
        generic_context = CPTritonContext()
        generic_ms = measure(
            lambda: generic_combine(
                timed_output,
                partial_lse,
                destination,
                args.rank,
                generic_context,
            ),
            args.warmups,
            args.iterations,
        )
        packed_ms = measure(
            lambda: packed_combine(
                timed_output, partial_lse, destination, args.rank
            ),
            args.warmups,
            args.iterations,
        )
        current_query_ms = measure(
            lambda: current_query_gather(local_query),
            args.warmups,
            args.iterations,
        )
        strided_query_ms = measure(
            lambda: strided_query_gather(local_query),
            args.warmups,
            args.iterations,
        )
        results.append(
            {
                "concurrency": concurrency,
                "rows": rows,
                "generic_model_pass_ms": generic_ms,
                "packed_model_pass_ms": packed_ms,
                "saved_model_pass_ms": generic_ms - packed_ms,
                "speedup_percent": (generic_ms / packed_ms - 1.0) * 100.0,
                "current_query_gather_ms": current_query_ms,
                "strided_query_gather_ms": strided_query_ms,
                "query_gather_saved_ms": current_query_ms - strided_query_ms,
                "max_abs_error": max_abs_error,
            }
        )

    if args.rank == 0:
        print(
            json.dumps(
                {
                    "schema": "ds4fv-dcp-combine.v2",
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
