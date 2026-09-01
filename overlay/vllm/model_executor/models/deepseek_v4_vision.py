# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash-Vision support for the pinned vLLM 0.28 image.

The experimental checkpoint exposes a text-only HF architecture even though
it contains a native DeepSeek ViT, aligner, image sentinel embeddings, visual
MoE routing biases, and bidirectional image-block attention.  This module
ports the checkpoint's preprocessing and vision tower without remote code;
the small accompanying source patch teaches the existing DeepSeek-V4 text
stack about the visual routing and attention contracts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch import nn
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.processing.context import TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.multimodal.processing.processor import MultiModalProcessingInfo
from vllm.sequence import IntermediateTensors


IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
IMAGE_TOKEN_ID = 129264

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
NUM_IMAGE_TYPES = 5
COMPRESS_PAD_TO = 4


def _required_vision_value(config: Any, name: str) -> Any:
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(
            f"DeepSeek-V4 Vision checkpoint is missing required field {name!r}"
        )
    return value


def _grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def _solve_resize_ratio(
    height: int,
    width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("image token budget is too small")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(
            max_w * patch_size * downsample_ratio / width,
            max_h * patch_size * downsample_ratio / height,
        )
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def _safe_resize(
    height: int,
    width: int,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    # Reserve the worst-case three position-alignment pads.
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = _grid_tokens(
        best_height, best_width, patch_size, downsample_ratio
    )
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = (
            _solve_resize_ratio(
                height,
                width,
                patch_size,
                downsample_ratio,
                budget,
            )
        )
        budget -= 1
    return n_llm_h, n_llm_w, best_height, best_width


def build_image_block(
    n_llm_h: int, n_llm_w: int, start_pos: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return checkpoint-exact N-layout types and aligner-row permutation."""
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2

    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
        + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = (
        torch.arange(rows * row_len)
        .view(rows // 2, 2, row_len)
        .transpose(1, 2)
        .reshape(-1)
    )
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(
        n_llm_h * n_llm_w
    ).view(n_llm_h, n_llm_w)
    permutation = image_idx[order]
    permutation = permutation[permutation >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START], dtype=torch.int64),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END], dtype=torch.int64),
        ]
    )
    return types, permutation


def _as_pil_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim != 3:
            raise ValueError(f"image array must have rank 3, got {array.ndim}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"unsupported DeepSeek-V4 image type: {type(value).__name__}")


def _prepare_image(image_value: object, config: Any) -> tuple[torch.Tensor, ...]:
    image = _as_pil_image(image_value)
    patch_size = int(_required_vision_value(config, "vision_patch_size"))
    downsample = int(_required_vision_value(config, "vision_downsample_ratio"))
    max_tokens = int(_required_vision_value(config, "vision_max_n_token"))
    min_pixels = int(_required_vision_value(config, "vision_min_pixels"))
    max_wh_ratio = getattr(config, "vision_max_wh_ratio", None)

    width, height = image.size
    if max_wh_ratio is not None and width > height * max_wh_ratio:
        width = int(height * max_wh_ratio)
    if 0 < width * height < min_pixels:
        ratio = math.sqrt(min_pixels / (width * height))
        width = int(width * ratio)
        height = int(height * ratio)

    best_width = math.ceil(width / patch_size) * patch_size
    best_height = math.ceil(height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = _safe_resize(
        height,
        width,
        best_height,
        best_width,
        patch_size,
        downsample,
        max_tokens,
    )
    n_vit_h = best_height // patch_size
    n_vit_w = best_width // patch_size
    if max_wh_ratio is not None and image.width >= max_wh_ratio * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(
            image, (best_width, best_height), color=(127, 127, 127)
        )

    pixels = (
        torch.from_numpy(np.array(image, dtype=np.float32, copy=True))
        .permute(2, 0, 1)
        .div_(255.0)
    )
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
        .contiguous()
    )
    return (
        patches,
        torch.tensor([n_vit_h, n_vit_w], dtype=torch.int64),
        torch.tensor([n_llm_h, n_llm_w], dtype=torch.int64),
    )


class DeepseekV4VisionProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int]:
        del seq_len, mm_counts
        return {"image": int(self.get_hf_config().vision_max_n_token)}

    def get_image_size_with_most_features(self) -> ImageSize:
        config = self.get_hf_config()
        patch = int(config.vision_patch_size)
        ratio = int(config.vision_downsample_ratio)
        grid = max(2, math.isqrt(int(config.vision_max_n_token) - 2))
        side = patch * ratio * grid
        return ImageSize(width=side, height=side)


class DeepseekV4VisionDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        size = self.info.get_image_size_with_most_features()
        return {
            "image": self._get_dummy_images(
                width=size.width,
                height=size.height,
                num_images=mm_counts.get("image", 0),
                overrides=mm_options.get("image"),
            )
        }


class DeepseekV4VisionMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionProcessingInfo]
):
    def _cached_apply_hf_processor(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> tuple[list[int], MultiModalProcessingInfo, bool]:
        # Leading alignment pads depend on the placeholder's prompt position,
        # so per-image processor/encoder cache reuse would be incorrect.
        return self._apply_hf_processor(inputs, timing_ctx)

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        del mm_kwargs
        tokenizer = self.info.get_tokenizer()
        config = self.info.get_hf_config()
        prompt_ids = tokenizer.encode(prompt, **tok_kwargs)
        placeholder_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if placeholder_id != IMAGE_TOKEN_ID:
            raise ValueError(
                f"expected {IMAGE_PLACEHOLDER}={IMAGE_TOKEN_ID}, got {placeholder_id}"
            )

        images = mm_data.get("images", ())
        if not isinstance(images, Sequence):
            raise TypeError("DeepSeek-V4 Vision images must be a sequence")
        if prompt_ids.count(placeholder_id) != len(images):
            raise ValueError(
                "the number of DeepSeek image placeholders does not match images"
            )

        prepared = [_prepare_image(image, config) for image in images]
        combined_ids: list[int] = []
        patches: list[torch.Tensor] = []
        patch_counts: list[int] = []
        vit_grids: list[torch.Tensor] = []
        llm_grids: list[torch.Tensor] = []
        block_types: list[torch.Tensor] = []
        block_counts: list[int] = []
        permutations: list[torch.Tensor] = []
        feature_counts: list[int] = []
        image_idx = 0

        for token_id in prompt_ids:
            if token_id != placeholder_id:
                combined_ids.append(token_id)
                continue
            image_patches, vit_grid, llm_grid = prepared[image_idx]
            types, permutation = build_image_block(
                int(llm_grid[0]), int(llm_grid[1]), len(combined_ids)
            )
            combined_ids.extend((int(config.vocab_size) + types).tolist())
            patches.append(image_patches)
            patch_counts.append(len(image_patches))
            vit_grids.append(vit_grid)
            llm_grids.append(llm_grid)
            block_types.append(types)
            block_counts.append(len(types))
            permutations.append(permutation)
            feature_counts.append(len(permutation))
            image_idx += 1

        data: dict[str, object] = {"input_ids": [combined_ids]}
        if patches:
            data.update(
                image_patches=torch.cat(patches),
                image_patch_counts=torch.tensor(patch_counts, dtype=torch.int64),
                image_vit_grids=torch.stack(vit_grids),
                image_llm_grids=torch.stack(llm_grids),
                image_block_types=torch.cat(block_types),
                image_block_counts=torch.tensor(block_counts, dtype=torch.int64),
                image_permutations=torch.cat(permutations),
                image_feature_counts=torch.tensor(feature_counts, dtype=torch.int64),
            )
        return BatchFeature(data=data, tensor_type=None)

    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        del prompt_text, mm_items, hf_processor_mm_kwargs, tokenization_kwargs
        return True

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        del hf_processor_mm_kwargs
        patch_counts = hf_inputs.get("image_patch_counts", torch.empty(0))
        block_counts = hf_inputs.get("image_block_counts", torch.empty(0))
        feature_counts = hf_inputs.get("image_feature_counts", torch.empty(0))
        return {
            "image_patches": MultiModalFieldConfig.flat_from_sizes(
                "image", patch_counts
            ),
            "image_patch_counts": MultiModalFieldConfig.batched("image"),
            "image_vit_grids": MultiModalFieldConfig.batched("image"),
            "image_llm_grids": MultiModalFieldConfig.batched("image"),
            "image_block_types": MultiModalFieldConfig.flat_from_sizes(
                "image", block_counts
            ),
            "image_block_counts": MultiModalFieldConfig.batched("image"),
            "image_permutations": MultiModalFieldConfig.flat_from_sizes(
                "image", feature_counts
            ),
            "image_feature_counts": MultiModalFieldConfig.batched("image"),
        }

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        del mm_items, hf_processor_mm_kwargs
        vocab_size = int(self.info.get_hf_config().vocab_size)

        def replacement(item_idx: int) -> list[int]:
            item = out_mm_kwargs["image"][item_idx]
            types = item["image_block_types"].data
            if not isinstance(types, torch.Tensor):
                raise TypeError("image_block_types must be a tensor")
            return (types.to(torch.int64) + vocab_size).tolist()

        return [
            PromptReplacement(
                modality="image",
                target=[IMAGE_TOKEN_ID],
                replacement=replacement,
            )
        ]


@lru_cache(maxsize=32)
def _vision_rope_cpu(
    n_h: int, n_w: int, dim: int, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    )
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    dtype = x.dtype
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin], dim=-1
    ).to(dtype)


class _VisionRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        normalized = x.float()
        normalized *= torch.rsqrt(
            normalized.square().mean(-1, keepdim=True) + self.eps
        )
        return (self.weight * normalized).to(dtype)


class _PatchEmbed(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        patch = int(config.vision_patch_size)
        self.proj = nn.Linear(3 * patch**2, int(config.vision_dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        return self.proj(patches.flatten(1))


class _VisionAttention(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.n_heads = int(config.vision_n_heads)
        dim = int(config.vision_dim)
        self.head_dim = dim // self.n_heads
        self.wqkv = nn.Linear(dim, 3 * dim)
        self.wo = nn.Linear(dim, dim)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        n_tokens = x.size(0)
        q, k, v = (
            tensor.view(n_tokens, self.n_heads, self.head_dim)
            for tensor in self.wqkv(x).chunk(3, dim=-1)
        )
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        )
        return self.wo(output.transpose(0, 1).reshape(n_tokens, -1))


class _VisionMLP(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        dim = int(config.vision_dim)
        intermediate = int(config.vision_inter_dim)
        self.w1 = nn.Linear(dim, 2 * intermediate, bias=False)
        self.w2 = nn.Linear(intermediate, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


class _VisionBlock(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        dim = int(config.vision_dim)
        self.norm1 = _VisionRMSNorm(dim)
        self.attn = _VisionAttention(config)
        self.norm2 = _VisionRMSNorm(dim)
        self.mlp = _VisionMLP(config)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV4VisionTower(nn.Module):
    """Checkpoint-exact full-attention ViT with two-dimensional RoPE."""

    def __init__(self, config: Any):
        super().__init__()
        dim = int(config.vision_dim)
        heads = int(config.vision_n_heads)
        self.rope_dim = dim // heads // 2
        self.rope_theta = float(config.vision_rope_theta)
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(
            [_VisionBlock(config) for _ in range(int(config.vision_n_layers))]
        )
        self.norm = _VisionRMSNorm(dim)

    def forward(
        self, patches: torch.Tensor, n_h: int, n_w: int
    ) -> torch.Tensor:
        x = self.patch_embed(patches)
        cos, sin = _vision_rope_cpu(n_h, n_w, self.rope_dim, self.rope_theta)
        cos = cos.to(device=x.device, non_blocking=True)
        sin = sin.to(device=x.device, non_blocking=True)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV4VisionAligner(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.downsample_ratio = int(config.vision_downsample_ratio)
        hidden_size = int(config.hidden_size)
        input_size = int(config.vision_dim) * self.downsample_ratio**2
        self.w1 = nn.Linear(input_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        ratio = self.downsample_ratio
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)
        x = F.pad(x, (0, -n_w % ratio, 0, -n_h % ratio))
        x = (
            F.unfold(x.unsqueeze(0), ratio, stride=ratio)
            .squeeze(0)
            .transpose(0, 1)
        )
        return self.w2(F.gelu(self.w1(x)))


def _vision_weights_mapper() -> WeightsMapper:
    # AutoWeightsLoader delegates every ``language_model.*`` group to the
    # native DeepseekV4ForCausalLM.load_weights implementation. Keep those
    # names in checkpoint form here so its mapper runs exactly once. Applying
    # the native renames in this outer wrapper as well turns ``head.weight``
    # into ``lm_lm_head.weight`` and similarly double-maps the model subtree.
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "language_model.layers.",
            "embed.": "language_model.embed.",
            "norm.": "language_model.norm.",
            "head.": "language_model.head.",
            "hc_head": "language_model.hc_head",
            "mtp.": "language_model.mtp.",
        },
    )


@MULTIMODAL_REGISTRY.register_processor(
    DeepseekV4VisionMultiModalProcessor,
    info=DeepseekV4VisionProcessingInfo,
    dummy_inputs=DeepseekV4VisionDummyInputsBuilder,
)
class DeepseekV4VisionForConditionalGeneration(
    nn.Module, SupportsEagle3, SupportsMultiModal, SupportsPP
):
    requires_raw_input_tokens = True
    mm_prefix_clamp_sliding_window = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        del i
        if modality.startswith("image"):
            return IMAGE_PLACEHOLDER
        raise ValueError("DeepSeek-V4 Vision supports only image inputs")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        for field in (
            "vision_n_layers",
            "vision_dim",
            "vision_n_heads",
            "vision_inter_dim",
            "vision_patch_size",
            "vision_rope_theta",
            "vision_downsample_ratio",
            "vision_max_n_token",
            "vision_min_pixels",
        ):
            _required_vision_value(config, field)
        if int(config.vision_n_layers) <= 0:
            raise ValueError("DeepSeek-V4 Vision requires vision_n_layers > 0")
        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            raise ValueError(
                "DeepSeek-V4 Vision visual routing is not supported by mega-MoE"
            )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision = DeepseekV4VisionTower(config)
            self.aligner = DeepseekV4VisionAligner(config)
            hidden_size = int(config.hidden_size)
            self.image_start = nn.Parameter(torch.empty(hidden_size))
            self.image_end = nn.Parameter(torch.empty(hidden_size))
            self.image_newline = nn.Parameter(torch.empty(hidden_size))
            self.image_pad = nn.Parameter(torch.empty(hidden_size))

        with self._mark_language_model(vllm_config):
            self.language_model = DeepseekV4ForCausalLM(
                vllm_config=vllm_config,
                prefix="language_model" if not prefix else f"{prefix}.language_model",
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.configure_mm_token_handling(
            int(config.vocab_size),
            [int(config.vocab_size) + token_type for token_type in range(NUM_IMAGE_TYPES)],
        )
        self.hf_to_vllm_mapper = _vision_weights_mapper()

    def _encode_one_image(
        self,
        patches: torch.Tensor,
        vit_grid: torch.Tensor,
        block_types: torch.Tensor,
        permutation: torch.Tensor,
    ) -> torch.Tensor:
        n_vit_h, n_vit_w = (int(value) for value in vit_grid.tolist())
        features = self.aligner(
            self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w
        )
        permutation = permutation.to(device=features.device, dtype=torch.int64)
        features = features[permutation]
        block_types = block_types.to(device=features.device, dtype=torch.int64)
        sentinels = torch.stack(
            [
                self.image_start,
                self.image_pad,
                self.image_pad,
                self.image_newline,
                self.image_end,
            ]
        )
        block = sentinels[block_types]
        block[block_types == IMAGE] = features
        return block

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        patches = kwargs.pop("image_patches", None)
        if patches is None:
            return []
        if not isinstance(patches, torch.Tensor):
            raise TypeError("image_patches must be a tensor")

        patch_counts = torch.as_tensor(kwargs.pop("image_patch_counts"))
        vit_grids = torch.as_tensor(kwargs.pop("image_vit_grids"))
        block_types = torch.as_tensor(kwargs.pop("image_block_types"))
        block_counts = torch.as_tensor(kwargs.pop("image_block_counts"))
        permutations = torch.as_tensor(kwargs.pop("image_permutations"))
        feature_counts = torch.as_tensor(kwargs.pop("image_feature_counts"))
        # Retained for validation/provenance even though the aligner derives its
        # output shape from the ViT grid.
        llm_grids = torch.as_tensor(kwargs.pop("image_llm_grids"))

        counts = [int(value) for value in patch_counts.flatten().tolist()]
        blocks = [int(value) for value in block_counts.flatten().tolist()]
        features = [int(value) for value in feature_counts.flatten().tolist()]
        vit_grids = vit_grids.reshape(-1, 2)
        llm_grids = llm_grids.reshape(-1, 2)
        if not (
            len(counts)
            == len(blocks)
            == len(features)
            == len(vit_grids)
            == len(llm_grids)
        ):
            raise ValueError("inconsistent DeepSeek-V4 Vision image metadata")

        outputs: list[torch.Tensor] = []
        patch_offset = block_offset = feature_offset = 0
        for index, (patch_count, block_count, feature_count) in enumerate(
            zip(counts, blocks, features)
        ):
            expected_features = int(llm_grids[index].prod().item())
            if feature_count != expected_features:
                raise ValueError(
                    f"image {index} has {feature_count} features, "
                    f"expected {expected_features}"
                )
            outputs.append(
                self._encode_one_image(
                    patches[patch_offset : patch_offset + patch_count],
                    vit_grids[index],
                    block_types[block_offset : block_offset + block_count],
                    permutations[
                        feature_offset : feature_offset + feature_count
                    ],
                )
            )
            patch_offset += patch_count
            block_offset += block_count
            feature_offset += feature_count
        return tuple(outputs)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs
        if intermediate_tensors is not None:
            inputs_embeds = None
        if input_ids is None:
            raise ValueError("DeepSeek-V4 Vision requires raw input token IDs")
        return self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self, skip_substrs=["language_model.model.mtp."]
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.language_model.model.finalize_mega_moe_weights()
        self.language_model.model.finalize_mhc_broadcast_weights()
        return loaded

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()
