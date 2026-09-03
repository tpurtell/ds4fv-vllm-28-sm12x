#!/usr/bin/env python3
"""CUDA-hidden contract for position-aware DeepSeek-V4 Vision encoder caching."""

from __future__ import annotations

import os
from types import SimpleNamespace

from vllm.model_executor.models import deepseek_v4_vision as vision


if os.uname().machine != "aarch64":
    raise RuntimeError("this smoke test is Spark/arm64-only")
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA must be hidden for the no-GPU smoke test")

types_0, _ = vision.build_image_block(11, 11, 0)
types_1, _ = vision.build_image_block(11, 11, 1)
types_4, _ = vision.build_image_block(11, 11, 4)
hash_0 = vision._position_aware_encoder_hash("same-image", types_0)
hash_1 = vision._position_aware_encoder_hash("same-image", types_1)
hash_4 = vision._position_aware_encoder_hash("same-image", types_4)

assert hash_0 != hash_1
assert hash_0 == hash_4
assert hash_0 != vision._position_aware_encoder_hash("other-image", types_0)

mm_info = vision.MultiModalProcessingInfo(
    kwargs={
        "image": [
            {
                "image_block_types": SimpleNamespace(data=types_0),
            }
        ]
    },
    hashes={"image": ["same-image"]},
    prompt_updates={"image": [[]]},
)
keyed_info = vision._with_position_aware_encoder_hashes(mm_info)
assert keyed_info.hashes == {"image": [hash_0]}
assert keyed_info.kwargs is mm_info.kwargs
assert keyed_info.prompt_updates is mm_info.prompt_updates

empty = vision.MultiModalProcessingInfo(
    kwargs={},
    hashes={},
    prompt_updates={},
)
assert vision._with_position_aware_encoder_hashes(empty) is empty

print("Spark CUDA-hidden Vision encoder-cache identity contract passed")
