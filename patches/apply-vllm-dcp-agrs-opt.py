#!/usr/bin/env python3
"""Fuse DeepSeek-V4's DCP AG/RS correction, packing, and output scatter."""

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
        raise SystemExit("usage: apply-vllm-dcp-agrs-opt.py VLLM_ROOT")
    root = Path(sys.argv[1]).resolve()
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    if not path.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")

    replace_once(
        path,
        '''from vllm.v1.attention.ops.dcp_utils import MLADCPManager
''',
        '''from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.common import mask_dcp_empty_shards_
from vllm.v1.attention.ops.dcp_utils import MLADCPManager
''',
        "packed AG/RS imports",
    )
    replace_once(
        path,
        '''def _attention_sink_for_shard(attention) -> torch.Tensor:
''',
        '''@triton.jit
def _dcp_correct_and_pack_rs_kernel(
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
    """LSE-correct local partials directly into NCCL destination chunks."""
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
    global_lse = tl.log2(tl.sum(tl.exp2(lses - lse_max), axis=0)) + lse_max

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


def _dcp_ag_rs_combine_into(
    dcp_manager: MLADCPManager,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    destination: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None:
    """Fuse LSE correction with rank-major packing and scatter in place."""
    group = dcp_manager.group
    world_size = group.world_size
    if dcp_manager.use_a2a or world_size <= 1:
        raise RuntimeError("packed AG/RS combine requires multi-rank ag_rs")
    num_tokens, global_heads, head_dim = partial_output.shape
    if global_heads % world_size != 0:
        raise RuntimeError(
            f"DCP global head count {global_heads} is not divisible by {world_size}."
        )
    local_heads = global_heads // world_size
    destination = destination[:num_tokens, :local_heads]
    if not destination.is_contiguous():
        raise RuntimeError("DeepSeek-V4 DCP output destination must be contiguous.")

    partial_lse = partial_lse.contiguous()
    mask_dcp_empty_shards_(partial_lse, seq_lens, query_start_loc)
    gathered_lse = group.all_gather(partial_lse, dim=0).reshape(
        world_size, num_tokens, global_heads
    )
    packed_output = partial_output.new_empty(
        (world_size * num_tokens, local_heads, head_dim)
    )
    _dcp_correct_and_pack_rs_kernel[(num_tokens, global_heads)](
        partial_output,
        gathered_lse,
        packed_output,
        *partial_output.stride(),
        *gathered_lse.stride(),
        group.rank_in_group,
        num_tokens,
        HEAD_DIM=head_dim,
        LOCAL_HEADS=local_heads,
        WORLD_SIZE=world_size,
    )

    communicator = group.device_communicator
    pynccl_comm = getattr(communicator, "pynccl_comm", None)
    if pynccl_comm is None or pynccl_comm.disabled:
        raise RuntimeError(
            "DeepSeek-V4 packed DCP AG/RS requires the active PyNccl communicator."
        )
    pynccl_comm.reduce_scatter(destination, packed_output)


def _attention_sink_for_shard(attention) -> torch.Tensor:
''',
        "packed AG/RS helpers",
    )
    replace_once(
        path,
        '''        if self.dcp_manager is not None:
            # Make vLLM's peak-memory profiler account for the live gathered Q,
            # partial O/LSE, and final local combine result. The caching
            # allocator reuses these blocks across sequential attention layers.
            num_tokens = int(q.shape[0])
            _profile_gathered_q = torch.empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            _profile_partial_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            _profile_partial_lse = torch.empty(
                (num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=device,
            )
            _profile_combined = output.new_empty(
                (num_tokens, self.padded_heads, self.head_dim)
            )
            _ = (
                _profile_gathered_q,
                _profile_partial_output,
                _profile_partial_lse,
                _profile_combined,
            )
''',
        '''        if self.dcp_manager is not None:
            # Mirror the two DCP peak-live stages. Gathered Q is freed after
            # attention so its block can be reused by the packed AG/RS input.
            num_tokens = int(q.shape[0])
            _profile_gathered_q = torch.empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            _profile_partial_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            _profile_partial_lse = torch.empty(
                (num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=device,
            )
            del _profile_gathered_q
            _profile_gathered_lse = torch.empty(
                (self.dcp_world_size, num_tokens, self._dcp_attention_heads),
                dtype=torch.float32,
                device=device,
            )
            _profile_packed_output = output.new_empty(
                (num_tokens, self._dcp_attention_heads, self.head_dim)
            )
            _ = (
                _profile_partial_output,
                _profile_partial_lse,
                _profile_gathered_lse,
                _profile_packed_output,
            )
''',
        "profile packed AG/RS buffer lifetimes",
    )
    replace_once(
        path,
        '''                swa_only=swa_only,
                lse_output=partial_lse,
            )
            if swa_metadata.dcp_local_seq_lens is None:
                raise RuntimeError("DeepSeek-V4 DCP local sequence lengths missing.")
            if swa_metadata.query_start_loc is None:
                raise RuntimeError("DeepSeek-V4 DCP query offsets missing.")
            num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
            combined = self.dcp_manager.combine(
                partial_output,
                partial_lse,
                seq_lens=swa_metadata.dcp_local_seq_lens[:num_reqs],
                query_start_loc=swa_metadata.query_start_loc[: num_reqs + 1],
            )
            output[:num_tokens].copy_(combined)
''',
        '''                swa_only=swa_only,
                lse_output=partial_lse,
            )
            # The packed combine input has the same size as gathered Q. Drop
            # the last reference now so the CUDA allocator can reuse that block.
            del gathered_query
            if swa_metadata.dcp_local_seq_lens is None:
                raise RuntimeError("DeepSeek-V4 DCP local sequence lengths missing.")
            if swa_metadata.query_start_loc is None:
                raise RuntimeError("DeepSeek-V4 DCP query offsets missing.")
            num_reqs = swa_metadata.num_decodes + swa_metadata.num_prefills
            seq_lens = swa_metadata.dcp_local_seq_lens[:num_reqs]
            query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
            if self.dcp_manager.use_a2a:
                combined = self.dcp_manager.combine(
                    partial_output,
                    partial_lse,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
                output[:num_tokens].copy_(combined)
            else:
                _dcp_ag_rs_combine_into(
                    self.dcp_manager,
                    partial_output,
                    partial_lse,
                    output[:num_tokens],
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                )
''',
        "direct packed AG/RS combine",
    )


if __name__ == "__main__":
    main()
