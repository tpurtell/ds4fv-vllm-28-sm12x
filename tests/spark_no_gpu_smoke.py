#!/usr/bin/env python3
"""Spark-side import and preprocessing smoke test; never run on the build host."""

from __future__ import annotations

import base64
import importlib.util
import inspect
import io
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from vllm.model_executor.models import deepseek_v4_vision as vision
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4SparseMLABackend
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.engine.input_processor import _model_max_input_token_id


def load_reference():
    path = Path(os.environ["DS4FV_REFERENCE_IMAGE_PROCESSOR"])
    spec = importlib.util.spec_from_file_location("ds4fv_reference_image", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference image processor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_image(width: int, height: int) -> Image.Image:
    pixels = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes((x % 251, y % 241, (x + y) % 239))
    return Image.frombytes("RGB", (width, height), bytes(pixels))


def main() -> None:
    if os.uname().machine != "aarch64":
        raise RuntimeError("this smoke test is Spark/arm64-only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA must be hidden for the no-GPU smoke test")

    reference = load_reference()
    config = SimpleNamespace(
        vocab_size=129280,
        vision_patch_size=14,
        vision_downsample_ratio=3,
        vision_max_n_token=384,
        vision_min_pixels=147456,
        vision_max_wh_ratio=8,
    )

    for height, width in ((384, 384), (240, 320), (100, 1200), (1800, 240)):
        best_width = ((width + 13) // 14) * 14
        best_height = ((height + 13) // 14) * 14
        ours = vision._safe_resize(
            height, width, best_height, best_width, 14, 3, 384
        )
        theirs = reference.safe_resize(
            height, width, best_height, best_width, 14, 3, 384
        )
        assert ours == theirs, (height, width, ours, theirs)

        n_h, n_w, _, _ = ours
        for start_pos in range(8):
            our_types, our_perm = vision.build_image_block(n_h, n_w, start_pos)
            ref_types, ref_perm = reference.build_image_block(n_h, n_w, start_pos)
            assert torch.equal(our_types, ref_types)
            assert torch.equal(our_perm, ref_perm)
            assert len(our_types) <= config.vision_max_n_token
            assert len(our_types) + 127 <= 512

    source_image = make_image(320, 240)
    encoded = io.BytesIO()
    source_image.save(encoded, format="PNG")
    ref_patches, ref_vit_h, ref_vit_w, ref_llm_h, ref_llm_w = (
        reference.load_image(
            {"data": base64.b64encode(encoded.getvalue()).decode()}, config
        )
    )
    patches, vit_grid, llm_grid = vision._prepare_image(source_image, config)
    assert torch.equal(patches, ref_patches)
    assert vit_grid.tolist() == [ref_vit_h, ref_vit_w]
    assert llm_grid.tolist() == [ref_llm_h, ref_llm_w]

    assert (
        _MULTIMODAL_MODELS["DeepseekV4VisionForConditionalGeneration"][0]
        == "deepseek_v4_vision"
    )
    factory_params = inspect.signature(FusedMoEFactory).parameters
    assert "vision_e_score_correction_bias" in factory_params
    assert "vision_vocab_size" in factory_params
    assert DeepseekV4SparseMLABackend.supports_mm_prefix()
    assert DeepseekSparseSWABackend.supports_mm_prefix()

    vision_model_config = SimpleNamespace(
        get_vocab_size=lambda: config.vocab_size,
        hf_config=SimpleNamespace(
            architectures=["DeepseekV4VisionForConditionalGeneration"]
        ),
    )
    assert (
        _model_max_input_token_id(vision_model_config, is_multimodal=False)
        == 129279
    )
    assert (
        _model_max_input_token_id(vision_model_config, is_multimodal=True)
        == 129284
    )

    # The wrapper adds only the child-module prefix. The native language-model
    # loader must own its checkpoint renames so they cannot run twice (the
    # first full Vision load caught head.weight becoming lm_lm_head.weight).
    wrapper_mapper = vision._vision_weights_mapper()
    assert wrapper_mapper.apply_list(
        [
            "layers.0.attn_norm.weight",
            "embed.weight",
            "norm.weight",
            "head.weight",
            "hc_head_base",
            "mtp.0.norm.weight",
            "vision.patch_embed.weight",
            "aligner.w1.weight",
            "image_start",
        ]
    ) == [
        "language_model.layers.0.attn_norm.weight",
        "language_model.embed.weight",
        "language_model.norm.weight",
        "language_model.head.weight",
        "language_model.hc_head_base",
        "language_model.mtp.0.norm.weight",
        "vision.patch_embed.weight",
        "aligner.w1.weight",
        "image_start",
    ]
    native_mapper = vision.DeepseekV4ForCausalLM.hf_to_vllm_mapper
    assert native_mapper.apply_list(["head.weight"]) == ["lm_head.weight"]

    for heads in (8, 16, 32, 64, 128):
        assert has_flashinfer_sparse_mla_sm120_config(heads, 128)
        assert has_flashinfer_sparse_mla_sm120_config(heads, 512)

    print("Spark no-GPU Vision smoke test passed")


if __name__ == "__main__":
    main()
