#!/usr/bin/env python3
"""Add the native 432-byte DeepSeek-V4 NVFP4 MLA path to vLLM 0.28.

Every edit is anchored against the already-patched production source.  The
image build therefore fails closed if either upstream vLLM or an earlier recipe
patch changes one of the execution or accounting contracts below.
"""

from __future__ import annotations

import sys
from pathlib import Path


MARKER = "Using DeepSeek's native 432-byte NVFP4 DS-MLA KV cache format."


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def patch_cache_dtype(root: Path) -> None:
    cache = root / "config/cache.py"
    replace_once(
        cache,
        '    "nvfp4",\n    "nvfp4_4over6",\n',
        '    "nvfp4",\n    "nvfp4_ds_mla",\n    "nvfp4_4over6",\n',
        "NVFP4 DS-MLA CLI dtype",
    )
    replace_once(
        cache,
        '    "nvfp4_4over6" uses the NVFP4 layout and selects between max/6 and max/4\n'
        '    scales per 16 values by minimizing squared reconstruction error.\n',
        '    "nvfp4_ds_mla" selects DeepSeek-V4\'s 432-byte packed MLA record.\n'
        '    "nvfp4_4over6" uses the NVFP4 layout and selects between max/6 and max/4\n'
        '    scales per 16 values by minimizing squared reconstruction error.\n',
        "NVFP4 DS-MLA cache dtype documentation",
    )

    torch_utils = root / "utils/torch_utils.py"
    replace_once(
        torch_utils,
        '    "nvfp4": torch.uint8,\n    "nvfp4_4over6": torch.uint8,\n',
        '    "nvfp4": torch.uint8,\n'
        '    "nvfp4_ds_mla": torch.uint8,\n'
        '    "nvfp4_4over6": torch.uint8,\n',
        "NVFP4 DS-MLA torch storage dtype",
    )

    config = root / "config/vllm.py"
    replace_once(
        config,
        '            self.cache_config.cache_dtype.startswith("nvfp4")\n'
        '            and self.model_config.use_mla\n',
        '            self.cache_config.cache_dtype.startswith("nvfp4")\n'
        '            and self.cache_config.cache_dtype != "nvfp4_ds_mla"\n'
        '            and self.model_config.use_mla\n',
        "permit model-specific NVFP4 MLA dtype",
    )


def patch_attention(root: Path) -> None:
    path = root / "models/deepseek_v4/attention.py"
    replace_once(
        path,
        '''    if use_fp8_ds_mla_layout:
        # fp8_ds_mla block format: UE8M0 block-scaled fp8 packed as uint8.
        assert kv_cache_dtype.startswith("fp8"), (
            f"DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, "
            f"got {kv_cache_dtype}"
        )
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
''',
        '''    if use_fp8_ds_mla_layout:
        # Both sparse-MLA formats are byte-addressed paged records. Generic
        # NVFP4 is a different K/V layout and remains rejected by the backend.
        assert kv_cache_dtype.startswith("fp8") or kv_cache_dtype == "nvfp4_ds_mla", (
            "DeepseekV4 packed MLA layout supports only fp8 or "
            f"nvfp4_ds_mla, got {kv_cache_dtype}"
        )
        if kv_cache_dtype == "nvfp4_ds_mla":
            logger.info_once(
                "Using DeepSeek's native 432-byte NVFP4 DS-MLA KV cache format."
            )
            return kv_cache_dtype, torch.uint8
        if kv_cache_dtype != "fp8_ds_mla":
            if cache_config is not None:
                cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        return kv_cache_dtype, torch.uint8
''',
        "packed MLA dtype resolution",
    )
    replace_once(
        path,
        '''                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
                eager_scratch_pool=eager_scratch_pool,
''',
        '''                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.prefix,
                use_fp4_cache=self.kv_cache_dtype == "nvfp4_ds_mla",
                eager_scratch_pool=eager_scratch_pool,
''',
        "compact compressor selection",
    )
    replace_once(
        path,
        '''        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if cache_dtype == torch.uint8:
''',
        '''        # kv is unchanged; attention reads kv solely via swa_kv_cache.
        if self.kv_cache_dtype == "nvfp4_ds_mla":
            from vllm.models.deepseek_v4.common.ops.nvfp4_ds_mla import (
                qnorm_rope_kv_insert_nvfp4_ds_mla,
            )

            if self.eager_scratch_pool is not None:
                q_out = self.eager_scratch_pool.q_out(q.shape[0])
            else:
                q_out = torch.empty(
                    (q.shape[0], self.padded_heads, self.head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
            return qnorm_rope_kv_insert_nvfp4_ds_mla(
                q,
                kv,
                q_out,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions,
                cos_sin_cache,
                padded_heads=self.padded_heads,
                eps=self.eps,
                block_size=swa_metadata.block_size,
            )

        if cache_dtype == torch.uint8:
''',
        "native target Q/KV producer dispatch",
    )
    replace_once(
        path,
        '''        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token;
            # head_size stays semantic (512).
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )
''',
        '''        uses_fp8_ds_mla_layout = self.kv_cache_dtype == "fp8_ds_mla"
        uses_nvfp4_ds_mla_layout = self.kv_cache_dtype == "nvfp4_ds_mla"
        uses_packed_mla_layout = (
            uses_fp8_ds_mla_layout or uses_nvfp4_ds_mla_layout
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_packed_mla_layout else self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            alignment=(
                432
                if uses_nvfp4_ds_mla_layout
                else 576
                if uses_fp8_ds_mla_layout
                else 512
            ),
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            # DeepseekV4 packed records keep head_size semantic (512).
            state_content_bytes=(
                432
                if uses_nvfp4_ds_mla_layout
                else 584
                if uses_fp8_ds_mla_layout
                else None
            ),
        )
''',
        "native compact main-cache accounting",
    )


def patch_compressor(root: Path) -> None:
    path = root / "models/deepseek_v4/compressor.py"
    replace_once(
        path,
        '''        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
''',
        '''        if self.head_dim == 512:
            self._quant_block = 16 if use_fp4_cache else 64
            self._token_stride = (
                432
                if use_fp4_cache
                else self.nope_head_dim + self.rope_head_dim * 2
            )
            self._scale_dim = (
                32 if use_fp4_cache else self.nope_head_dim // 64 + 1
            )
''',
        "native compact compressor geometry",
    )
    replace_once(
        path,
        '''        if current_platform.is_cuda() and self.head_dim == 512:
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
''',
        '''        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and not self.use_fp4_cache
        ):
            from .nvidia.ops.sparse_attn_compress_cutedsl import (
''',
        "Triton compact compressor dispatch",
    )

    op = root / "models/deepseek_v4/common/ops/fused_compress_quant_cache.py"
    replace_once(
        op,
        '''    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    abs_2d = tl.abs(quant_2d)
    block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
    block_absmax = tl.maximum(block_absmax, 1e-4)

    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_scaled = quant_2d * inv_scales_col
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = exponents + 127.0
    max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
    encoded = tl.maximum(tl.minimum(encoded, max_encoded), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)  # each [NUM_PAIRS] fp32

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32
    if SANITIZE_CACHE_NANS:
        result = tl.where(result == result, result, 0.0)

    # Store rotated rope portion as bf16 into the cache's bf16 area.
    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)
''',
        '''    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    value_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # Register-based GPT-J RoPE in fp32.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)
    if SANITIZE_CACHE_NANS:
        result = tl.where(result == result, result, 0.0)
    quant_input = result.to(tl.bfloat16).to(tl.float32)

    if QUANT_BLOCK == 16:
        # Native DSV4 NVFP4 record: 256B E2M1 + 32B E4M3 group scales
        # + 16B reserved + 128B BF16 RoPE.
        tl.static_assert(TOKEN_STRIDE == 432)
        tl.static_assert(SCALE_DIM == 32)
        quant_3d = tl.reshape(quant_input, (32, 8, 2))
        lo, hi = tl.split(quant_3d)
        amax = tl.maximum(tl.max(tl.abs(lo), axis=1), tl.max(tl.abs(hi), axis=1))
        raw_scale = tl.maximum(amax * (1.0 / 6.0), 1.1754943508222875e-38)
        scale_fp8 = raw_scale.to(tl.float8e4nv)
        inv_scale = tl.reshape(1.0 / scale_fp8.to(tl.float32), (32, 1))
        packed = _fp32x2_to_fp4x2(lo * inv_scale, hi * inv_scale)
        tl.store(value_ptr + tl.arange(0, 256), tl.reshape(packed, (256,)))
        tl.store(
            value_ptr + 256 + tl.arange(0, 32),
            scale_fp8.to(tl.uint8, bitcast=True),
        )
        tl.store(value_ptr + 288 + tl.arange(0, 16), 0)
        rope_ptr = (value_ptr + 304).to(tl.pointer_type(tl.bfloat16))
        rope_local = block - NOPE_HEAD_DIM
        tl.store(
            rope_ptr + rope_local,
            result.to(tl.bfloat16),
            mask=(block >= NOPE_HEAD_DIM) & mask,
        )
    else:
        # FP8/UE8M0 keeps page-major 576B payload rows followed by an
        # 8B-per-token scale footer.
        scale_ptr = (
            cache_block_ptr
            + kv_cache_block_size * TOKEN_STRIDE
            + kv_pos_in_block * SCALE_DIM
        )
        N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
        N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
        quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
        block_absmax = tl.maximum(tl.max(tl.abs(quant_2d), axis=1), 1e-4)
        exponents = tl.ceil(tl.log2(block_absmax * (1.0 / FP8_MAX)))
        inv_scales = tl.reshape(tl.exp2(-exponents), (N_QUANT_BLOCKS, 1))
        x_fp8 = tl.clamp(
            quant_2d * inv_scales, -FP8_MAX, FP8_MAX
        ).to(tl.float8e4nv)
        x_uint8 = tl.reshape(
            x_fp8.to(tl.uint8, bitcast=True), (TRITON_BLOCK_SIZE,)
        )
        tl.store(value_ptr + block, x_uint8, mask=block < NOPE_HEAD_DIM)
        scale_idx = tl.arange(0, N_QUANT_BLOCKS)
        max_encoded: tl.constexpr = 254.0 if SANITIZE_CACHE_NANS else 255.0
        encoded = tl.maximum(
            tl.minimum(exponents + 127.0, max_encoded), 0.0
        )
        tl.store(
            scale_ptr + scale_idx,
            encoded.to(tl.uint8),
            mask=scale_idx < N_NOPE_BLOCKS,
        )
        tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))
        bf16_ptr = (value_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
        rope_local = block - NOPE_HEAD_DIM
        tl.store(
            bf16_ptr + rope_local,
            result.to(tl.bfloat16),
            mask=(block >= NOPE_HEAD_DIM) & mask,
        )
''',
        "native compact compressed-cache producer",
    )


def patch_cache_specs(root: Path) -> None:
    sparse = root / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        sparse,
        '        "fp8_ds_mla",\n        "fp8",  # alias for fp8_ds_mla\n',
        '        "fp8_ds_mla",\n'
        '        "nvfp4_ds_mla",\n'
        '        "fp8",  # alias for fp8_ds_mla\n',
        "sparse MLA supported compact dtype",
    )
    replace_once(
        sparse,
        '''        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
''',
        '''        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        elif cache_dtype_str == "nvfp4_ds_mla":
            return (num_blocks, block_size, 432)
        else:
''',
        "sparse MLA compact cache shape",
    )

    swa = root / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        swa,
        '''        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
''',
        '''        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        uses_nvfp4_ds_mla_layout = (
            self.cache_config.cache_dtype == "nvfp4_ds_mla"
        )
        return SlidingWindowMLASpec(
''',
        "SWA compact dtype selection",
    )
    replace_once(
        swa,
        '''            # DeepseekV4 fp8_ds_mla: 584B per token (448B NoPE + 128B RoPE + 8B scales)
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
''',
        '''            # DeepseekV4 packed record bytes; head_size stays semantic.
            state_content_bytes=(
                432
                if uses_nvfp4_ds_mla_layout
                else 584
                if uses_fp8_ds_mla_layout
                else None
            ),
            alignment=(
                432
                if uses_nvfp4_ds_mla_layout
                else 576
                if uses_fp8_ds_mla_layout
                else 512
            ),
''',
        "SWA compact cache accounting",
    )
    replace_once(
        swa,
        '''        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
''',
        '''        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 SWA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        elif cache_dtype_str == "nvfp4_ds_mla":
            return (num_blocks, block_size, 432)
        else:
''',
        "SWA compact cache shape",
    )


def patch_dspark(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/dspark.py"
    replace_once(
        path,
        '''    if cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged layout
''',
        '''    if attn.kv_cache_dtype == "nvfp4_ds_mla":
        from vllm.models.deepseek_v4.common.ops.nvfp4_ds_mla import (
            insert_nvfp4_ds_mla_kv,
        )

        insert_nvfp4_ds_mla_kv(
            kv,
            swa_cache,
            slot_mapping,
            positions,
            cos_sin_cache,
            block_size=block_size,
        )
    elif cache_dtype == torch.uint8:
        # fp8_ds_mla UE8M0 paged layout
''',
        "dSpark native compact context priming",
    )


def patch_b12x_reads(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        '''            sm_scale=self.scale,
            page_block_size=int(swa_kv_cache.shape[1]),
            attn_sink=self.attn_sink,
''',
        '''            sm_scale=self.scale,
            page_block_size=int(swa_kv_cache.shape[1]),
            scale_format=(
                2 if self.kv_cache_dtype == "nvfp4_ds_mla" else None
            ),
            fp8_rope=(
                False if self.kv_cache_dtype == "nvfp4_ds_mla" else None
            ),
            attn_sink=self.attn_sink,
''',
        "explicit B12x NVFP4 prefill format",
    )
    replace_once(
        path,
        '''        self._b12x_o_proj_weights = None
        self._b12x_compressed_mla_enabled = (
            self._b12x_o_proj_enabled
            and self.use_fp8_ds_mla_layout
            and _use_b12x_compressed_mla_decode()
        )
''',
        '''        self._b12x_o_proj_weights = None
        if self.kv_cache_dtype == "nvfp4_ds_mla" and not self._b12x_o_proj_enabled:
            raise RuntimeError(
                "nvfp4_ds_mla requires the B12x linear/attention backend"
            )
        self._b12x_compressed_mla_enabled = (
            self._b12x_o_proj_enabled
            and self.use_fp8_ds_mla_layout
            and (
                self.kv_cache_dtype == "nvfp4_ds_mla"
                or _use_b12x_compressed_mla_decode()
            )
        )
''',
        "automatic B12x compact read path",
    )
    replace_once(
        path,
        '''        indexed_lengths: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        from b12x.attention import compressed_sparse_mla
        from b12x.attention._shared.mla.compressed_reference import (
            COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN,
        )

        swa_cache = self._as_b12x_sparse_cache(swa_cache)
        if int(swa_cache.shape[1]) % COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN:
            raise ValueError(
                "B12x SWA cache page is not an integral compressed-MLA page"
            )
        swa_page_size = (
            int(swa_cache.shape[1]) // COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
        )
        indexed_page_size = None
        if indexed_cache is not None:
            indexed_cache = self._as_b12x_sparse_cache(indexed_cache)
            if (
                int(indexed_cache.shape[1])
                % COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
            ):
                raise ValueError(
                    "B12x indexed cache page is not an integral "
                    "compressed-MLA page"
                )
            indexed_page_size = (
                int(indexed_cache.shape[1])
                // COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
            )

''',
        '''        indexed_lengths: torch.Tensor | None,
        swa_page_size: int,
        indexed_page_size: int | None,
        output: torch.Tensor,
    ) -> None:
        from b12x.attention import compressed_sparse_mla

        record_bytes = 432 if self.kv_cache_dtype == "nvfp4_ds_mla" else 584
        swa_cache = self._as_b12x_sparse_cache(swa_cache)
        expected_swa_width = int(swa_page_size) * record_bytes
        if int(swa_cache.shape[1]) != expected_swa_width:
            raise ValueError(
                "B12x SWA cache width disagrees with its dtype/page contract: "
                f"width={int(swa_cache.shape[1])}, expected={expected_swa_width}"
            )
        if indexed_cache is not None:
            if indexed_page_size is None:
                raise ValueError("indexed page size is required with indexed cache")
            indexed_cache = self._as_b12x_sparse_cache(indexed_cache)
            expected_indexed_width = int(indexed_page_size) * record_bytes
            if int(indexed_cache.shape[1]) != expected_indexed_width:
                raise ValueError(
                    "B12x indexed cache width disagrees with its dtype/page contract: "
                    f"width={int(indexed_cache.shape[1])}, "
                    f"expected={expected_indexed_width}"
                )

''',
        "B12x dtype-aware page validation",
    )
    replace_once(
        path,
        '''                indexed_indices=extra_sparse_indices,
                indexed_lengths=extra_sparse_lengths,
                output=output,
''',
        '''                indexed_indices=extra_sparse_indices,
                indexed_lengths=extra_sparse_lengths,
                swa_page_size=swa_metadata.block_size,
                indexed_page_size=(
                    attn_metadata.block_size // self.compress_ratio
                    if extra_cache is not None and attn_metadata is not None
                    else None
                ),
                output=output,
''',
        "B12x explicit live page sizes",
    )
    replace_once(
        path,
        '''            use_b12x_prefill = (
                self._b12x_o_proj_enabled
                and int(swa_indices_chunk.shape[-1]) in (128, 512)
            )
''',
        '''            b12x_prefill_width = int(swa_indices_chunk.shape[-1])
            if (
                self.kv_cache_dtype == "nvfp4_ds_mla"
                and b12x_prefill_width not in (128, 512, 1024, 2048)
            ):
                raise RuntimeError(
                    "NVFP4 DS-MLA prefill requires a B12x-supported width; "
                    f"got {b12x_prefill_width}"
                )
            use_b12x_prefill = (
                self._b12x_o_proj_enabled
                and b12x_prefill_width in (128, 512, 1024, 2048)
            )
''',
        "B12x compact prefill enforcement",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-dsv4-nvfp4.py VLLM_ROOT")
    root = Path(sys.argv[1]).resolve()
    attention = root / "models/deepseek_v4/attention.py"
    if not attention.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")
    if MARKER in attention.read_text():
        raise RuntimeError(f"NVFP4 DS-MLA patch already applied to {root}")

    patch_cache_dtype(root)
    patch_attention(root)
    patch_compressor(root)
    patch_cache_specs(root)
    patch_dspark(root)
    patch_b12x_reads(root)


if __name__ == "__main__":
    main()
