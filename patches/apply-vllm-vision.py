#!/usr/bin/env python3
"""Apply the DS4FV Vision delta to one exact vLLM 0.28 source tree.

Every edit is anchored to the pinned source and must match exactly once.  This
is intentionally not a fuzzy patch: a base-image drift fails the image build
instead of producing a subtly text-only or causally-masked Vision service.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: {label} expected exactly one source anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


def patch_registry(root: Path) -> None:
    path = root / "model_executor/models/registry.py"
    replace_once(
        path,
        '    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),\n',
        '    "DeepseekOCR2ForCausalLM": ("deepseek_ocr2", "DeepseekOCR2ForCausalLM"),\n'
        '    "DeepseekV4VisionForConditionalGeneration": (\n'
        '        "deepseek_v4_vision",\n'
        '        "DeepseekV4VisionForConditionalGeneration",\n'
        '    ),\n',
        "multimodal registry",
    )


def patch_model_config(root: Path) -> None:
    path = root / "model_executor/models/config.py"
    replace_once(
        path,
        '    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLMConfig,\n',
        '    "DeepseekV4ForCausalLM": DeepseekV4ForCausalLMConfig,\n'
        '    "DeepseekV4VisionForConditionalGeneration": '
        'DeepseekV4ForCausalLMConfig,\n',
        "Vision config hook",
    )

    path = root / "config/model.py"
    replace_once(
        path,
        '            elif arch == "DeepseekV4ForCausalLM":\n'
        '                self.tokenizer_mode = "deepseek_v4"\n',
        '            elif arch in (\n'
        '                "DeepseekV4ForCausalLM",\n'
        '                "DeepseekV4VisionForConditionalGeneration",\n'
        '            ):\n'
        '                self.tokenizer_mode = "deepseek_v4"\n',
        "DeepSeek-V4 tokenizer mode",
    )

    path = root / "config/vllm.py"
    replace_once(
        path,
        '        "DeepseekV4ForCausalLM",\n'
        '        "GraniteMoeForCausalLM",\n',
        '        "DeepseekV4ForCausalLM",\n'
        '        "DeepseekV4VisionForConditionalGeneration",\n'
        '        "GraniteMoeForCausalLM",\n',
        "MRV2 default",
    )
    replace_once(
        path,
        '        return DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES - '
        '{"DeepseekV4ForCausalLM"}\n',
        '        return DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES - {\n'
        '            "DeepseekV4ForCausalLM",\n'
        '            "DeepseekV4VisionForConditionalGeneration",\n'
        '        }\n',
        "ROCm MRV2 exclusion",
    )
    replace_once(
        path,
        '                    "DeepseekV4ForCausalLM",\n'
        '                    "DeepSeekV4MTPModel",\n',
        '                    "DeepseekV4ForCausalLM",\n'
        '                    "DeepseekV4VisionForConditionalGeneration",\n'
        '                    "DeepSeekV4MTPModel",\n',
        "breakable CUDA graph default",
    )


def patch_moe_factory(root: Path) -> None:
    path = root / "model_executor/layers/fused_moe/layer.py"
    replace_once(
        path,
        '    hash_indices_table: torch.Tensor | None = None,\n'
        '    runner_cls: type[MoERunner] | None = None,\n',
        '    hash_indices_table: torch.Tensor | None = None,\n'
        '    vision_e_score_correction_bias: torch.Tensor | None = None,\n'
        '    vision_vocab_size: int | None = None,\n'
        '    runner_cls: type[MoERunner] | None = None,\n',
        "FusedMoE Vision arguments",
    )
    replace_once(
        path,
        '            hash_indices_table=hash_indices_table,\n'
        '        )\n\n'
        '    if params_dtype is None:\n',
        '            hash_indices_table=hash_indices_table,\n'
        '            vision_e_score_correction_bias=vision_e_score_correction_bias,\n'
        '            vision_vocab_size=vision_vocab_size,\n'
        '        )\n\n'
        '    if params_dtype is None:\n',
        "router Vision arguments",
    )

    path = root / "model_executor/layers/fused_moe/router/router_factory.py"
    replace_once(
        path,
        '    hash_indices_table: torch.Tensor | None = None,\n'
        ') -> FusedMoERouter:\n',
        '    hash_indices_table: torch.Tensor | None = None,\n'
        '    vision_e_score_correction_bias: torch.Tensor | None = None,\n'
        '    vision_vocab_size: int | None = None,\n'
        ') -> FusedMoERouter:\n',
        "router factory Vision arguments",
    )
    replace_once(
        path,
        '    if e_score_correction_bias is not None or hash_indices_table is not None:\n'
        '        return FusedTopKBiasRouter(\n',
        '    if (\n'
        '        e_score_correction_bias is not None\n'
        '        or hash_indices_table is not None\n'
        '        or vision_e_score_correction_bias is not None\n'
        '    ):\n'
        '        return FusedTopKBiasRouter(\n',
        "Vision router selection",
    )
    replace_once(
        path,
        '            shared_expert_weight=shared_expert_weight,\n'
        '        )\n\n'
        '    if (\n'
        '        num_fused_shared_experts > 0\n',
        '            shared_expert_weight=shared_expert_weight,\n'
        '            vision_e_score_correction_bias=vision_e_score_correction_bias,\n'
        '            vision_vocab_size=vision_vocab_size,\n'
        '        )\n\n'
        '    if (\n'
        '        num_fused_shared_experts > 0\n',
        "Vision biased router construction",
    )


def patch_router(root: Path) -> None:
    path = root / "model_executor/layers/fused_moe/router/fused_topk_bias_router.py"
    replace_once(
        path,
        'import torch\n\n'
        'import vllm._custom_ops as ops\n',
        'import torch\n'
        'import torch.nn.functional as F\n\n'
        'import vllm._custom_ops as ops\n',
        "functional import",
    )
    replace_once(
        path,
        '        shared_expert_weight: float = 1.0,\n'
        '    ):\n',
        '        shared_expert_weight: float = 1.0,\n'
        '        vision_e_score_correction_bias: torch.Tensor | None = None,\n'
        '        vision_vocab_size: int | None = None,\n'
        '    ):\n',
        "Vision router constructor arguments",
    )
    replace_once(
        path,
        '        self.num_fused_shared_experts = num_fused_shared_experts\n'
        '        self.shared_expert_weight = shared_expert_weight\n',
        '        self.num_fused_shared_experts = num_fused_shared_experts\n'
        '        self.shared_expert_weight = shared_expert_weight\n'
        '        self.vision_e_score_correction_bias = (\n'
        '            vision_e_score_correction_bias\n'
        '        )\n'
        '        self.vision_vocab_size = vision_vocab_size\n'
        '        if (vision_e_score_correction_bias is None) != (\n'
        '            vision_vocab_size is None\n'
        '        ):\n'
        '            raise ValueError(\n'
        '                "visual routing bias and vocabulary size must be set together"\n'
        '            )\n',
        "Vision router state",
    )
    old = '''        """Compute routing using fused top-k with bias."""
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias.data
            if self.e_score_correction_bias is not None
            else None,
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            input_tokens=input_ids,
            hash_indices_table=self._hash_indices_table,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        if self.num_fused_shared_experts > 0:
'''
    new = '''        """Compute text routing with the fused path and overlay visual routes."""
        visual_bias = self.vision_e_score_correction_bias
        image_mask = None
        safe_input_ids = input_ids
        if visual_bias is not None:
            if input_ids is None or self.vision_vocab_size is None:
                raise ValueError("DeepSeek-V4 visual routing requires input_ids")
            flat_ids = input_ids.reshape(-1)
            image_mask = flat_ids >= self.vision_vocab_size
            safe_input_ids = flat_ids.masked_fill(image_mask, 0)

        # Hash layers carry the ordinary bias solely because the Vision
        # checkpoint stores it. Text selection in those layers remains tid2eid.
        text_bias = (
            None
            if self._hash_indices_table is not None
            else self.e_score_correction_bias
        )
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=(
                text_bias.data if text_bias is not None else None
            ),
            topk=self.top_k,
            renormalize=self.renormalize,
            indices_type=indices_type,
            input_tokens=safe_input_ids,
            hash_indices_table=self._hash_indices_table,
            routed_scaling_factor=self.routed_scaling_factor,
        )

        if visual_bias is not None:
            assert image_mask is not None
            if self.scoring_func != "sqrtsoftplus":
                raise ValueError(
                    "DeepSeek-V4 Vision requires sqrtsoftplus routing"
                )
            scores = F.softplus(router_logits.float()).sqrt()
            use_sorted = envs.VLLM_BATCH_INVARIANT
            visual_ids = torch.topk(
                scores + visual_bias.data.unsqueeze(0),
                k=self.top_k,
                dim=-1,
                sorted=use_sorted,
            )[1]
            visual_weights = scores.gather(1, visual_ids)
            if self.renormalize:
                visual_weights /= visual_weights.sum(dim=-1, keepdim=True)
            visual_weights *= self.routed_scaling_factor
            topk_weights = torch.where(
                image_mask.unsqueeze(-1), visual_weights, topk_weights
            )
            topk_ids = torch.where(
                image_mask.unsqueeze(-1),
                visual_ids.to(topk_ids.dtype),
                topk_ids,
            )

        if self.num_fused_shared_experts > 0:
'''
    replace_once(path, old, new, "visual route overlay")


def patch_deepseek_model(root: Path) -> None:
    path = root / "models/deepseek_v4/nvidia/model.py"
    replace_once(
        path,
        '        self.gate.e_score_correction_bias = None\n'
        '        self.gate.tid2eid = None\n'
        '        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers\n'
        '        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32\n'
        '        if is_hash_moe:\n',
        '        self.gate.e_score_correction_bias = None\n'
        '        self.gate.tid2eid = None\n'
        '        self.gate.bias_vl = None\n'
        '        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers\n'
        '        is_vision = int(getattr(config, "vision_n_layers", 0)) > 0\n'
        '        if self.use_mega_moe and is_vision:\n'
        '            raise NotImplementedError(\n'
        '                "DeepSeek-V4 Vision routing is not supported by mega-MoE"\n'
        '            )\n'
        '        self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32\n'
        '        if is_hash_moe:\n',
        "Vision gate initialization",
    )
    replace_once(
        path,
        '                requires_grad=False,\n'
        '            )\n'
        '        elif getattr(config, "topk_method", None) == "noaux_tc":\n'
        '            self.gate.e_score_correction_bias = nn.Parameter(\n'
        '                torch.empty(config.n_routed_experts, dtype=torch.float32),\n'
        '                requires_grad=False,\n'
        '            )\n\n'
        '        if config.n_shared_experts is None:\n',
        '                requires_grad=False,\n'
        '            )\n'
        '        if (\n'
        '            not is_hash_moe\n'
        '            and getattr(config, "topk_method", None) == "noaux_tc"\n'
        '        ) or is_vision:\n'
        '            self.gate.e_score_correction_bias = nn.Parameter(\n'
        '                torch.empty(config.n_routed_experts, dtype=torch.float32),\n'
        '                requires_grad=False,\n'
        '            )\n'
        '        if is_vision:\n'
        '            self.gate.bias_vl = nn.Parameter(\n'
        '                torch.empty(config.n_routed_experts, dtype=torch.float32),\n'
        '                requires_grad=False,\n'
        '            )\n\n'
        '        if config.n_shared_experts is None:\n',
        "Vision gate biases",
    )
    replace_once(
        path,
        '            hash_indices_table=self.gate.tid2eid,\n'
        '            swiglu_limit=self.swiglu_limit,\n',
        '            hash_indices_table=self.gate.tid2eid,\n'
        '            vision_e_score_correction_bias=self.gate.bias_vl,\n'
        '            vision_vocab_size=(\n'
        '                config.vocab_size if self.gate.bias_vl is not None else None\n'
        '            ),\n'
        '            swiglu_limit=self.swiglu_limit,\n',
        "Vision route factory binding",
    )


def patch_attention(root: Path) -> None:
    path = root / "models/deepseek_v4/attention.py"
    replace_once(
        path,
        '        self.window_size = config.sliding_window\n',
        '        # Vision widens physical cache storage while retaining the\n'
        '        # checkpoint\'s trained 128-token text window.\n'
        '        self.window_size = getattr(\n'
        '            config, "vision_text_sliding_window", config.sliding_window\n'
        '        )\n'
        '        self.swa_cache_window_size = config.sliding_window\n',
        "semantic Vision SWA window",
    )
    replace_once(
        path,
        '            window_size=self.window_size,\n'
        '            dtype=self.kv_cache_torch_dtype,\n',
        '            window_size=self.swa_cache_window_size,\n'
        '            dtype=self.kv_cache_torch_dtype,\n',
        "physical Vision SWA cache",
    )

    path = root / "models/deepseek_v4/sparse_mla.py"
    replace_once(
        path,
        '    @classmethod\n'
        '    def supports_sink(cls) -> bool:\n'
        '        return True\n\n'
        '    @classmethod\n'
        '    def supports_compute_capability',
        '    @classmethod\n'
        '    def supports_sink(cls) -> bool:\n'
        '        return True\n\n'
        '    @classmethod\n'
        '    def supports_mm_prefix(cls) -> bool:\n'
        '        # The compressed indexer remains causal; the paired SWA cache\n'
        '        # implements the multimodal bidirectional span.\n'
        '        return True\n\n'
        '    @classmethod\n'
        '    def supports_compute_capability',
        "sparse MLA mm-prefix capability",
    )

    path = root / "models/deepseek_v4/nvidia/flashinfer_sparse.py"
    replace_once(
        path,
        '        required_topk = _required_sm120_sparse_topk(vllm_config, self.window_size)\n'
        '        if not has_flashinfer_sparse_mla_sm120_config(self.padded_heads, required_topk):\n'
        '            raise RuntimeError(\n'
        '                "FLASHINFER_MLA_SPARSE_DSV4 on SM120 requires a FlashInfer "\n'
        '                "DSV4 sparse MLA decode specialization for "\n'
        '                f"(num_q_heads={self.padded_heads}, top_k={required_topk}). "\n'
        '                "Install a FlashInfer build containing "\n'
        '                "flashinfer-ai/flashinfer#4380."\n'
        '            )\n',
        '        required_topks = {\n'
        '            _required_sm120_sparse_topk(vllm_config, self.window_size),\n'
        '            self.swa_cache_window_size,\n'
        '        }\n'
        '        for required_topk in sorted(required_topks):\n'
        '            if has_flashinfer_sparse_mla_sm120_config(\n'
        '                self.padded_heads, required_topk\n'
        '            ):\n'
        '                continue\n'
        '            raise RuntimeError(\n'
        '                "FLASHINFER_MLA_SPARSE_DSV4 on SM12x requires a "\n'
        '                "FlashInfer DSV4 sparse MLA specialization for "\n'
        '                f"(num_q_heads={self.padded_heads}, top_k={required_topk}). "\n'
        '                "Install a FlashInfer build containing that dispatch width."\n'
        '            )\n',
        "SM12x semantic and physical sparse widths",
    )


def patch_input_validation(root: Path) -> None:
    path = root / "v1/engine/input_processor.py"
    replace_once(
        path,
        "logger = init_logger(__name__)\n\n\nclass InputProcessor:\n",
        "logger = init_logger(__name__)\n\n\n"
        "def _model_max_input_token_id(\n"
        "    model_config, *, is_multimodal: bool\n"
        ") -> int:\n"
        "    max_token_id = int(model_config.get_vocab_size()) - 1\n"
        "    architectures = getattr(\n"
        "        model_config.hf_config, \"architectures\", ()\n"
        "    ) or ()\n"
        "    if (\n"
        "        is_multimodal\n"
        "        and \"DeepseekV4VisionForConditionalGeneration\" in architectures\n"
        "    ):\n"
        "        # The native checkpoint owns five learned image sentinel/type\n"
        "        # embeddings immediately above the text vocabulary. They are\n"
        "        # masked before text embedding and replaced by multimodal\n"
        "        # embeddings, so permit exactly that bounded OOV interval.\n"
        "        max_token_id += 5\n"
        "    return max_token_id\n\n\n"
        "class InputProcessor:\n",
        "DeepSeek-V4 Vision OOV input bound helper",
    )
    replace_once(
        path,
        "            model_vocab_size = model_config.get_vocab_size()\n"
        "            # A negative id is out of vocabulary just like an over-large one,\n",
        "            model_max_input_id = _model_max_input_token_id(\n"
        "                model_config,\n"
        '                is_multimodal=prompt_input["type"] == "multimodal",\n'
        "            )\n"
        "            # A negative id is out of vocabulary just like an over-large one,\n",
        "DeepSeek-V4 Vision OOV input bound",
    )
    replace_once(
        path,
        "            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):\n",
        "            if max_input_id > max(tokenizer.max_token_id, model_max_input_id):\n",
        "DeepSeek-V4 Vision OOV validation",
    )


def patch_sparse_swa(root: Path) -> None:
    path = root / "v1/attention/backends/mla/sparse_swa.py"
    replace_once(
        path,
        'from typing import ClassVar, cast\n\n'
        'import torch\n',
        'from typing import ClassVar, cast\n\n'
        'import numpy as np\n'
        'import torch\n',
        "NumPy import",
    )
    replace_once(
        path,
        'from vllm.utils.math_utils import cdiv, next_power_of_2\n',
        'from vllm.utils.math_utils import cdiv, next_power_of_2\n'
        'from vllm.utils.torch_utils import PIN_MEMORY\n',
        "pinned memory import",
    )
    replace_once(
        path,
        'from vllm.v1.attention.backends.utils import split_decodes_and_prefills\n',
        'from vllm.v1.attention.backends.utils import (\n'
        '    fill_mm_prefix_query_ranges,\n'
        '    split_decodes_and_prefills,\n'
        ')\n',
        "mm-prefix range helper import",
    )
    replace_once(
        path,
        '    @classmethod\n'
        '    def get_supported_head_sizes(cls) -> list[int]:\n'
        '        return [512]\n\n'
        '    @staticmethod\n'
        '    def get_builder_cls',
        '    @classmethod\n'
        '    def get_supported_head_sizes(cls) -> list[int]:\n'
        '        return [512]\n\n'
        '    @classmethod\n'
        '    def supports_mm_prefix(cls) -> bool:\n'
        '        return True\n\n'
        '    @staticmethod\n'
        '    def get_builder_cls',
        "SWA mm-prefix capability",
    )
    replace_once(
        path,
        '        self.window_size = hf_config.sliding_window\n\n'
        '        # Detect which DeepseekV4 layer types',
        '        self.window_size = getattr(\n'
        '            hf_config, "vision_text_sliding_window", hf_config.sliding_window\n'
        '        )\n'
        '        self.index_width = hf_config.sliding_window\n'
        '        if self.index_width < self.window_size:\n'
        '            raise ValueError("physical SWA window cannot be smaller than text SWA")\n\n'
        '        # Detect which DeepseekV4 layer types',
        "SWA semantic and physical widths",
    )
    replace_once(
        path,
        '        self.prefill_swa_indices = torch.zeros(\n'
        '            max_tokens,\n'
        '            1,\n'
        '            self.window_size,\n'
        '            dtype=torch.int32,\n'
        '            device=self.device,\n'
        '        )\n',
        '        self.prefill_swa_indices = torch.zeros(\n'
        '            max_tokens,\n'
        '            1,\n'
        '            self.index_width,\n'
        '            dtype=torch.int32,\n'
        '            device=self.device,\n'
        '        )\n',
        "widened Vision prefill indices",
    )
    replace_once(
        path,
        '        self.is_valid_token = torch.zeros(\n'
        '            max_tokens,\n'
        '            dtype=torch.bool,\n'
        '            device=self.device,\n'
        '        )\n\n'
        '        # DSpark draft:',
        '        self.is_valid_token = torch.zeros(\n'
        '            max_tokens,\n'
        '            dtype=torch.bool,\n'
        '            device=self.device,\n'
        '        )\n'
        '        self.mm_prefix_query_ranges_cpu: torch.Tensor | None = None\n'
        '        self.mm_prefix_query_ranges_np: np.ndarray | None = None\n'
        '        self.mm_prefix_query_ranges_gpu: torch.Tensor | None = None\n'
        '        if self.vllm_config.model_config.is_mm_prefix_lm:\n'
        '            self.mm_prefix_query_ranges_cpu = torch.empty(\n'
        '                (max_tokens, 2), dtype=torch.int32, pin_memory=PIN_MEMORY\n'
        '            )\n'
        '            self.mm_prefix_query_ranges_np = (\n'
        '                self.mm_prefix_query_ranges_cpu.numpy()\n'
        '            )\n'
        '            self.mm_prefix_query_ranges_gpu = torch.empty(\n'
        '                (max_tokens, 2), dtype=torch.int32, device=self.device\n'
        '            )\n\n'
        '        # DSpark draft:',
        "mm-prefix staging buffers",
    )

    # Decode calls retain the narrow semantic window and never contain image tokens.
    replace_once(
        path,
        '                    self.window_size,\n'
        '                    query_start_loc,\n'
        '                    seq_lens,\n'
        '                    token_to_req_indices,\n'
        '                    is_valid_token,\n'
        '                    block_table,\n'
        '                    block_table.stride(0),\n'
        '                    self.block_size,\n'
        '                    token_offset=0,\n'
        '                    TRITON_BLOCK_SIZE=1024,\n'
        '                )\n\n'
        '        # Prefill SWA indices',
        '                    self.window_size,\n'
        '                    self.window_size,\n'
        '                    query_start_loc,\n'
        '                    seq_lens,\n'
        '                    token_to_req_indices,\n'
        '                    is_valid_token,\n'
        '                    block_table,\n'
        '                    block_table.stride(0),\n'
        '                    self.block_size,\n'
        '                    None,\n'
        '                    token_offset=0,\n'
        '                    HAS_MM_PREFIX=False,\n'
        '                    TRITON_BLOCK_SIZE=1024,\n'
        '                )\n\n'
        '        # Prefill SWA indices',
        "decode SWA kernel arguments",
    )

    old = '''        # Prefill SWA indices live in paged coordinates. `token_offset` lets
        # the kernel read is_valid_token / token_to_req_indices at absolute
        # prefill positions while writing output starting at index 0.
        if num_prefill_tokens > 0:
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
                prefill_swa_indices,
                prefill_swa_indices.stride(0),
                prefill_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                token_offset=num_decode_tokens,
                TRITON_BLOCK_SIZE=1024,
            )
'''
    new = '''        # Prefill SWA indices live in paged coordinates. Image queries use
        # the full inclusive mm-prefix range while text retains the trained
        # semantic window. The physical matrix is wide enough for both.
        mm_query_ranges = None
        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None and self.mm_prefix_query_ranges_np is not None:
            assert common_attn_metadata.seq_lens_cpu_upper_bound is not None
            num_mm_tokens = fill_mm_prefix_query_ranges(
                self.mm_prefix_query_ranges_np,
                mm_ranges,
                common_attn_metadata.query_start_loc_cpu,
                common_attn_metadata.seq_lens_cpu_upper_bound,
            )
            if num_mm_tokens > 0:
                assert self.mm_prefix_query_ranges_cpu is not None
                assert self.mm_prefix_query_ranges_gpu is not None
                mm_query_ranges = self.mm_prefix_query_ranges_gpu[:num_mm_tokens]
                mm_query_ranges.copy_(
                    self.mm_prefix_query_ranges_cpu[:num_mm_tokens],
                    non_blocking=True,
                )

        if num_prefill_tokens > 0:
            prefill_swa_indices = self.prefill_swa_indices[:num_prefill_tokens]
            prefill_swa_lens = self.prefill_swa_lens[:num_prefill_tokens]
            _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
                prefill_swa_indices,
                prefill_swa_indices.stride(0),
                prefill_swa_lens,
                self.window_size,
                self.index_width,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                mm_query_ranges,
                token_offset=num_decode_tokens,
                HAS_MM_PREFIX=mm_query_ranges is not None,
                TRITON_BLOCK_SIZE=1024,
            )
'''
    replace_once(path, old, new, "Vision prefill SWA ranges")

    replace_once(
        path,
        '            metadata.decode_swa_indices.shape[-1],\n'
        '            metadata.query_start_loc,\n',
        '            metadata.decode_swa_indices.shape[-1],\n'
        '            metadata.decode_swa_indices.shape[-1],\n'
        '            metadata.query_start_loc,\n',
        "draft decode index width",
    )
    replace_once(
        path,
        '            self.block_size,\n'
        '            token_offset=0,\n'
        '            TRITON_BLOCK_SIZE=1024,\n'
        '        )\n'
        '        tile_sched = self.build_tile_scheduler',
        '            self.block_size,\n'
        '            None,\n'
        '            token_offset=0,\n'
        '            HAS_MM_PREFIX=False,\n'
        '            TRITON_BLOCK_SIZE=1024,\n'
        '        )\n'
        '        tile_sched = self.build_tile_scheduler',
        "draft decode mm-prefix arguments",
    )

    old_kernel = '''def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    token_offset,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        # Clear the row so a padded token cannot gather through stale indices.
        for i in range(0, window_size, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < window_size,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start

    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1

    swa_len = end_pos - start_pos
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, window_size, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)

        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets

        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < window_size,
        )
'''
    new_kernel = '''def _compute_swa_indices_and_lens_kernel(
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    window_size,
    index_width,
    query_start_loc_ptr,
    seq_lens_ptr,
    token_to_req_indices_ptr,
    is_valid_token_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    mm_prefix_query_ranges_ptr,
    token_offset,
    HAS_MM_PREFIX: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid + token_offset
    is_valid = tl.load(is_valid_token_ptr + token_idx)
    if not is_valid:
        tl.store(swa_lens_ptr + pid, 0)
        for i in range(0, index_width, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(
                swa_indices_ptr + pid * swa_indices_stride + offset,
                -1,
                mask=offset < index_width,
            )
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    query_start = tl.load(query_start_loc_ptr + req_idx)
    query_end = tl.load(query_start_loc_ptr + req_idx + 1)
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + req_idx)
    prefix_len = seq_len - query_len

    pos = prefix_len + token_idx - query_start
    start_pos = tl.maximum(pos - window_size + 1, 0)
    end_pos = pos + 1
    if HAS_MM_PREFIX:
        mm_start = tl.load(mm_prefix_query_ranges_ptr + token_idx * 2)
        mm_end = tl.load(mm_prefix_query_ranges_ptr + token_idx * 2 + 1)
        in_image = mm_start >= 0
        start_pos = tl.where(in_image, tl.minimum(start_pos, mm_start), start_pos)
        end_pos = tl.where(in_image, tl.maximum(end_pos, mm_end + 1), end_pos)

    swa_len = end_pos - start_pos
    tl.device_assert(swa_len <= index_width, "SWA index width is too small")
    tl.store(swa_lens_ptr + pid, swa_len)

    for i in range(0, index_width, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        pos_offset = start_pos + offset
        block_indices = pos_offset // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=pos_offset < end_pos,
        )
        block_offsets = pos_offset % block_size
        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(offset < swa_len, slot_ids, -1)
        tl.store(
            swa_indices_ptr + pid * swa_indices_stride + offset,
            slot_ids,
            mask=offset < index_width,
        )
'''
    replace_once(path, old_kernel, new_kernel, "bidirectional image SWA kernel")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1]).resolve()
    expected = root / "models/deepseek_v4/nvidia/model.py"
    if not expected.is_file():
        raise RuntimeError(f"not a vLLM 0.28 package tree: {root}")
    marker = root / ".ds4fv-vision-patched"
    if marker.exists():
        raise RuntimeError(f"refusing to patch the same source tree twice: {root}")

    patch_registry(root)
    patch_model_config(root)
    patch_moe_factory(root)
    patch_router(root)
    patch_deepseek_model(root)
    patch_attention(root)
    patch_input_validation(root)
    patch_sparse_swa(root)
    marker.write_text("ds4fv-vllm-28-sm12x Vision patch v1\n")


if __name__ == "__main__":
    main()
