# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native 432-byte DeepSeek-V4 NVFP4 MLA cache producers.

The installed vLLM wheel's fused CUDA producer writes only the 584-byte
FP8/UE8M0 record.  These Triton kernels preserve its Q-normalization/RoPE
contract while writing B12x's compact record directly through the live packed
cache strides:

* 256 bytes of packed E2M1 values for the complete rotated 512-D latent;
* 32 E4M3 group-of-16 scales;
* 16 reserved zero bytes; and
* 128 bytes of BF16 RoPE data.

The cache block stride is deliberately independent of the logical page width;
vLLM may expose a view into a larger model-wide packed slab.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

from .fused_indexer_q import _fp32x2_to_fp4x2


DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_NVFP4_RECORD_BYTES = 432
DSV4_NVFP4_SCALE_OFFSET = 256
DSV4_NVFP4_PAD_OFFSET = 288
DSV4_NVFP4_ROPE_OFFSET = 304


@triton.jit
def _store_nvfp4_latent(
    output,
    cache_u8,
    cache_bf16,
    data_base_bytes,
    dim,
    rope_mask,
):
    """Store one already-rotated BF16-rounded DSV4 latent."""
    quant_3d = tl.reshape(output, (32, 8, 2))
    lo, hi = tl.split(quant_3d)
    max_abs = tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi), axis=1))
    raw_scale = tl.maximum(max_abs * (1.0 / 6.0), 1.1754943508222875e-38)
    scale_fp8 = raw_scale.to(tl.float8e4nv)
    decoded_scale = scale_fp8.to(tl.float32)
    inv_scale = tl.reshape(1.0 / decoded_scale, (32, 1))
    packed = _fp32x2_to_fp4x2(lo * inv_scale, hi * inv_scale)

    tl.store(
        cache_u8 + data_base_bytes + tl.arange(0, 256),
        tl.reshape(packed, (256,)),
    )
    tl.store(
        cache_u8 + data_base_bytes + 256 + tl.arange(0, 32),
        scale_fp8.to(tl.uint8, bitcast=True),
    )
    tl.store(
        cache_u8 + data_base_bytes + 288 + tl.arange(0, 16),
        0,
    )
    rope_local = dim - 448
    tl.store(
        cache_bf16 + (data_base_bytes + 304) // 2 + rope_local,
        output.to(tl.bfloat16),
        mask=rope_mask,
    )


@triton.jit
def _qnorm_rope_kv_insert_kernel(
    q,
    kv,
    q_out,
    cache_u8,
    cache_bf16,
    slot_mapping,
    positions,
    cos_sin,
    q_stride_t,
    q_stride_h,
    q_out_stride_t,
    q_out_stride_h,
    kv_stride_t,
    cache_block_stride,
    cache_token_stride,
    cos_sin_stride_pos,
    eps,
    NUM_HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    task = tl.program_id(1)
    dim = tl.arange(0, 512)
    rope_mask = dim >= 448
    rope_dim = dim - 448
    partner_dim = 448 + (rope_dim ^ 1)
    position = tl.load(positions + token)
    pair = tl.maximum(rope_dim >> 1, 0)
    cs = cos_sin + position * cos_sin_stride_pos
    cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
    sin_v = tl.load(
        cs + 32 + pair,
        mask=rope_mask,
        other=0.0,
    )

    if task < PADDED_HEADS:
        real_head = task < NUM_HEADS
        q_base = q + token * q_stride_t + task * q_stride_h
        values = tl.load(q_base + dim, mask=real_head, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(values * values, axis=0) / 512 + eps)
        normalized = (values * inv).to(tl.bfloat16).to(tl.float32)
        partner = tl.load(
            q_base + partner_dim,
            mask=rope_mask & real_head,
            other=0.0,
        ).to(tl.float32)
        partner = (partner * inv).to(tl.bfloat16).to(tl.float32)
        rotated = tl.where(
            (rope_dim & 1) == 0,
            normalized * cos_v - partner * sin_v,
            normalized * cos_v + partner * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized)
        out_base = q_out + token * q_out_stride_t + task * q_out_stride_h
        tl.store(out_base + dim, output.to(tl.bfloat16))
    else:
        slot = tl.load(slot_mapping + token).to(tl.int64)
        if slot < 0:
            return
        kv_base = kv + token * kv_stride_t
        normalized = tl.load(kv_base + dim).to(tl.float32)
        partner = tl.load(
            kv_base + partner_dim,
            mask=rope_mask,
            other=0.0,
        ).to(tl.float32)
        rotated = tl.where(
            (rope_dim & 1) == 0,
            normalized * cos_v - partner * sin_v,
            normalized * cos_v + partner * sin_v,
        )
        output = tl.where(rope_mask, rotated, normalized)
        output = output.to(tl.bfloat16).to(tl.float32)
        block = slot // CACHE_BLOCK_SIZE
        row = slot - block * CACHE_BLOCK_SIZE
        data_base_bytes = block * cache_block_stride + row * cache_token_stride
        _store_nvfp4_latent(
            output,
            cache_u8,
            cache_bf16,
            data_base_bytes,
            dim,
            rope_mask,
        )


@triton.jit
def _kv_insert_kernel(
    kv,
    cache_u8,
    cache_bf16,
    slot_mapping,
    positions,
    cos_sin,
    kv_stride_t,
    cache_block_stride,
    cache_token_stride,
    cos_sin_stride_pos,
    CACHE_BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    slot = tl.load(slot_mapping + token).to(tl.int64)
    if slot < 0:
        return

    dim = tl.arange(0, 512)
    rope_mask = dim >= 448
    rope_dim = dim - 448
    partner_dim = 448 + (rope_dim ^ 1)
    kv_base = kv + token * kv_stride_t
    normalized = tl.load(kv_base + dim).to(tl.float32)
    partner = tl.load(
        kv_base + partner_dim,
        mask=rope_mask,
        other=0.0,
    ).to(tl.float32)
    position = tl.load(positions + token)
    pair = tl.maximum(rope_dim >> 1, 0)
    cs = cos_sin + position * cos_sin_stride_pos
    cos_v = tl.load(cs + pair, mask=rope_mask, other=1.0)
    sin_v = tl.load(
        cs + 32 + pair,
        mask=rope_mask,
        other=0.0,
    )
    rotated = tl.where(
        (rope_dim & 1) == 0,
        normalized * cos_v - partner * sin_v,
        normalized * cos_v + partner * sin_v,
    )
    output = tl.where(rope_mask, rotated, normalized)
    output = output.to(tl.bfloat16).to(tl.float32)
    block = slot // CACHE_BLOCK_SIZE
    row = slot - block * CACHE_BLOCK_SIZE
    data_base_bytes = block * cache_block_stride + row * cache_token_stride
    _store_nvfp4_latent(
        output,
        cache_u8,
        cache_bf16,
        data_base_bytes,
        dim,
        rope_mask,
    )


def _validate_cache(kv_cache: torch.Tensor, block_size: int) -> None:
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"NVFP4 DS-MLA cache must be uint8, got {kv_cache.dtype}")
    if kv_cache.ndim == 4:
        if int(kv_cache.shape[-2]) != 1:
            raise ValueError(
                "NVFP4 DS-MLA cache requires one shared KV head, got "
                f"{tuple(kv_cache.shape)}"
            )
    elif kv_cache.ndim != 3:
        raise ValueError(
            "NVFP4 DS-MLA cache must have rank 3 or 4, got "
            f"{tuple(kv_cache.shape)}"
        )
    if int(kv_cache.shape[1]) != int(block_size):
        raise ValueError(
            f"NVFP4 cache block size mismatch: shape={tuple(kv_cache.shape)}, "
            f"metadata={block_size}"
        )
    if int(kv_cache.shape[-1]) != DSV4_NVFP4_RECORD_BYTES:
        raise ValueError(
            "NVFP4 DS-MLA requires a 432-byte record, got "
            f"{int(kv_cache.shape[-1])}"
        )
    if int(kv_cache.stride(-1)) != 1:
        raise ValueError(
            f"NVFP4 DS-MLA record must be contiguous, got {kv_cache.stride()}"
        )
    if int(kv_cache.stride(1)) != DSV4_NVFP4_RECORD_BYTES:
        raise ValueError(
            "NVFP4 DS-MLA token stride must be 432 bytes, got "
            f"{int(kv_cache.stride(1))}"
        )


def qnorm_rope_kv_insert_nvfp4_ds_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_out: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    padded_heads: int,
    eps: float,
    block_size: int,
) -> torch.Tensor:
    """Normalize/RoPE Q and write normalized KV to the compact cache."""
    _validate_cache(kv_cache, block_size)
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise TypeError(f"NVFP4 DS-MLA producer requires BF16 q/kv, got {q.dtype}/{kv.dtype}")
    tokens, heads, head_dim = map(int, q.shape)
    if head_dim != DSV4_HEAD_DIM or tuple(kv.shape) != (tokens, DSV4_HEAD_DIM):
        raise ValueError(f"invalid DSV4 q/kv shapes: q={tuple(q.shape)}, kv={tuple(kv.shape)}")
    if tuple(q_out.shape[:3]) != (tokens, int(padded_heads), DSV4_HEAD_DIM):
        raise ValueError(
            f"invalid padded Q output {tuple(q_out.shape)} for "
            f"{tokens}/{padded_heads}/{DSV4_HEAD_DIM}"
        )
    if q_out.dtype != torch.bfloat16:
        raise TypeError(f"NVFP4 DS-MLA Q output must be BF16, got {q_out.dtype}")
    if tokens == 0:
        return q_out

    _qnorm_rope_kv_insert_kernel[(tokens, int(padded_heads) + 1)](
        q,
        kv,
        q_out,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        slot_mapping,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        q_out.stride(0),
        q_out.stride(1),
        kv.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        cos_sin_cache.stride(0),
        float(eps),
        NUM_HEADS=heads,
        PADDED_HEADS=int(padded_heads),
        CACHE_BLOCK_SIZE=int(block_size),
        num_warps=8,
    )
    return q_out


def insert_nvfp4_ds_mla_kv(
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    block_size: int,
) -> None:
    """RoPE and insert already-RMS-normalized context KV for dSpark."""
    _validate_cache(kv_cache, block_size)
    if kv.dtype != torch.bfloat16 or kv.ndim != 2 or int(kv.shape[1]) != 512:
        raise TypeError(
            "NVFP4 DS-MLA context KV must be [N,512] BF16, got "
            f"dtype={kv.dtype}, shape={tuple(kv.shape)}"
        )
    tokens = int(kv.shape[0])
    if tokens == 0:
        return
    _kv_insert_kernel[(tokens,)](
        kv,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        slot_mapping,
        positions,
        cos_sin_cache,
        kv.stride(0),
        kv_cache.stride(0),
        kv_cache.stride(1),
        cos_sin_cache.stride(0),
        CACHE_BLOCK_SIZE=int(block_size),
        num_warps=8,
    )


__all__ = [
    "DSV4_NVFP4_RECORD_BYTES",
    "insert_nvfp4_ds_mla_kv",
    "qnorm_rope_kv_insert_nvfp4_ds_mla",
]
