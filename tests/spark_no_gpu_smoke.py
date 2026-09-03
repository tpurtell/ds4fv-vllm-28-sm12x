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
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    DeepseekV4FlashInferMLASparseBackend,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _compute_global_topk_indices_and_lens_kernel,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4SparseMLABackend,
    DeepseekV4SparseMLAMetadataBuilder,
)
from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120_config
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadataBuilder,
    DeepseekV4IndexerBackend,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWABackend,
    DeepseekSparseSWAMetadataBuilder,
)
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
    mapped_names = vision._vision_weights_mapper().apply_list(
        [
            "layers.0.attn.q_norm.weight",
            "model.layers.0.mlp.experts.0.gate_proj.trellis",
            "vision.norm.weight",
        ]
    )
    assert mapped_names == [
        "language_model.layers.0.attn.q_norm.weight",
        "language_model.model.layers.0.mlp.experts.0.gate_proj.trellis",
        "vision.norm.weight",
    ]
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

    # Encoder outputs contain the position-dependent alignment pads and must
    # not alias in vLLM's raw-image-keyed cache. Equivalent modulo-four layouts
    # should retain their cache hit.
    hash_types_0, _ = vision.build_image_block(11, 11, 0)
    hash_types_1, _ = vision.build_image_block(11, 11, 1)
    hash_types_4, _ = vision.build_image_block(11, 11, 4)
    hash_0 = vision._position_aware_encoder_hash("same-image", hash_types_0)
    hash_1 = vision._position_aware_encoder_hash("same-image", hash_types_1)
    hash_4 = vision._position_aware_encoder_hash("same-image", hash_types_4)
    assert hash_0 != hash_1
    assert hash_0 == hash_4
    assert hash_0 != vision._position_aware_encoder_hash("other-image", hash_types_0)
    mm_info = vision.MultiModalProcessingInfo(
        kwargs={
            "image": [
                {
                    "image_block_types": SimpleNamespace(data=hash_types_0),
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

    # Equal-grid images must share one real ViT batch dimension without
    # allowing full attention to cross image boundaries. Compare the batched
    # tower+aligner path directly with independent single-image executions.
    torch.manual_seed(121)
    tiny_vision_config = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=16,
        vision_n_layers=2,
        vision_rope_theta=10000.0,
        vision_downsample_ratio=2,
        hidden_size=12,
    )
    tower = vision.DeepseekV4VisionTower(tiny_vision_config).eval()
    aligner = vision.DeepseekV4VisionAligner(tiny_vision_config).eval()
    image_batch = torch.randn(3, 4, 3, 2, 2)
    with torch.no_grad():
        batched_features = aligner(tower(image_batch, 2, 2), 2, 2)
        serial_features = torch.stack(
            [aligner(tower(image, 2, 2), 2, 2) for image in image_batch]
        )
    assert batched_features.shape == (3, 1, 12)
    torch.testing.assert_close(
        batched_features, serial_features, rtol=1e-5, atol=1e-5
    )

    assert (
        _MULTIMODAL_MODELS["DeepseekV4VisionForConditionalGeneration"][0]
        == "deepseek_v4_vision"
    )
    factory_params = inspect.signature(FusedMoEFactory).parameters
    assert "vision_e_score_correction_bias" in factory_params
    assert "vision_vocab_size" in factory_params
    assert DeepseekV4SparseMLABackend.supports_mm_prefix()
    assert DeepseekSparseSWABackend.supports_mm_prefix()
    assert DeepseekV4IndexerBackend.supports_device_cpu_query_lens_mismatch()
    assert (
        DeepseekV4FlashInferMLASparseBackend
        .supports_device_cpu_query_lens_mismatch()
    )
    assert DeepseekSparseSWABackend.supports_device_cpu_query_lens_mismatch()
    indexer_builder_source = inspect.getsource(DeepseekV32IndexerMetadataBuilder)
    assert "self.enable_adaptive_verification" in indexer_builder_source
    assert "not self.enable_adaptive_verification" in indexer_builder_source
    assert "enable_adaptive_verification" in inspect.getsource(
        DeepseekV4SparseMLAMetadataBuilder.get_cudagraph_support
    )
    assert "enable_adaptive_verification" in inspect.getsource(
        DeepseekSparseSWAMetadataBuilder.get_cudagraph_support
    )
    global_topk_source = inspect.getsource(
        _compute_global_topk_indices_and_lens_kernel.fn
    )
    assert "safe_req_idx" in global_topk_source
    assert "(local_idx >= 0) & is_valid_token" in global_topk_source
    assert supports_eagle3(vision.DeepseekV4VisionForConditionalGeneration)

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
