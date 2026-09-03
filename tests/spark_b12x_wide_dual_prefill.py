#!/usr/bin/env python3
"""Spark-only numeric qualification for Vision's wide dual-cache MLA prefill.

This intentionally imports B12x's independent PyTorch oracle from its pinned
source tree.  It must only be run in the image on a GB10 / SM121 Spark.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


B12X_ROOT = Path("/opt/b12x")
if not (B12X_ROOT / "tests/_reference/dsv4_extra_ref.py").is_file():
    raise RuntimeError("pinned B12x source tree is not available at /opt/b12x")
sys.path.insert(0, str(B12X_ROOT))

from b12x.attention._shared.mla.compressed_reference import (  # noqa: E402
    compressed_sparse_mla_page_nbytes,
)
from b12x.attention._shared.mla.prefill import (  # noqa: E402
    run_unified_prefill,
)
from tests._reference import dsv4_extra_ref, dsv4_ref  # noqa: E402


def repack_compressed(packed: torch.Tensor, page_size: int) -> torch.Tensor:
    num_blocks = int(packed.shape[0])
    payload = page_size * dsv4_ref.DSV4_KV_GMEM_STRIDE
    out = torch.zeros(
        (num_blocks, compressed_sparse_mla_page_nbytes(page_size)),
        dtype=torch.uint8,
        device=packed.device,
    )
    out[:, :payload] = packed.reshape(num_blocks, payload)
    return out


def cosine(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.double().flatten()
    expected = expected.double().flatten()
    return float((actual @ expected) / (actual.norm() * expected.norm()))


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("this qualification must run on a Spark GPU")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 1):
        raise RuntimeError(f"expected Spark SM121, got capability {capability}")

    num_tokens = 2
    num_heads = 32
    topk = 512
    extra_topk = 512
    page_size = 64
    case = dsv4_extra_ref.make_dsv4_extra_decode_case(
        num_heads=num_heads,
        topk=topk,
        extra_topk=extra_topk,
        num_tokens=num_tokens,
        num_blocks=8,
        page_block_size=page_size,
        pbs_extra=page_size,
        extra_num_blocks=8,
        invalidate_half=False,
        device="cuda",
        seed=121512,
    )
    main_cache = repack_compressed(case["kv_cache"], page_size)
    extra_cache = repack_compressed(case["extra_kv_cache"], page_size)
    extra_lengths = torch.tensor([512, 257], dtype=torch.int32, device="cuda")
    length_cases = {
        "mixed": torch.tensor([384, 128], dtype=torch.int32, device="cuda"),
        # Rate-aware DCP owns replicated SWA on rank 0.  A C4 prefill on every
        # other rank is therefore an all-extra union: the main/SWA section is
        # empty while compressed C4 records remain live.  Qualify that exact
        # boundary instead of relying on the ordinary mixed-length case.
        "all-extra": torch.zeros(num_tokens, dtype=torch.int32, device="cuda"),
    }

    for label, main_lengths in length_cases.items():
        expected, expected_lse = dsv4_extra_ref.dsv4_extra_decode_reference(
            case["q"],
            case["kv_cache"],
            case["topk_indices"],
            case["sm_scale"],
            case["extra_kv_cache"],
            case["extra_indices"],
            page_block_size=page_size,
            pbs_extra=page_size,
            topk_length=main_lengths,
            extra_topk_length=extra_lengths,
            main_kv_dequant=case["kv_dequant"],
            extra_kv_dequant=case["extra_kv_dequant"],
        )

        output, lse = run_unified_prefill(
            q=case["q"].contiguous(),
            kv_cache=main_cache,
            topk_indices=case["topk_indices"].contiguous(),
            topk_length=main_lengths,
            sm_scale=case["sm_scale"],
            page_block_size=page_size,
            extra_kv_cache=extra_cache,
            extra_indices=case["extra_indices"].contiguous(),
            extra_topk_length=extra_lengths,
            extra_page_block_size=page_size,
        )
        torch.cuda.synchronize()

        actual = output.float()
        expected = expected.float()
        actual_lse = lse.float()
        expected_lse = expected_lse.float()
        cos = cosine(actual, expected)
        max_abs = float((actual - expected).abs().max())
        max_lse = float((actual_lse - expected_lse).abs().max())
        print(
            "wide dual-cache prefill qualification: "
            f"case={label} capability={capability} heads={num_heads} "
            f"main={topk} extra={extra_topk} cos={cos:.8f} "
            f"max_abs={max_abs:.8f} max_lse={max_lse:.8f}"
        )
        if not math.isfinite(cos) or cos <= 0.999:
            raise AssertionError(f"{label} output cosine {cos} <= 0.999")
        if max_abs >= 2e-2:
            raise AssertionError(f"{label} output max abs {max_abs} >= 2e-2")
        if max_lse >= 5e-2:
            raise AssertionError(f"{label} LSE max abs {max_lse} >= 5e-2")


if __name__ == "__main__":
    main()
