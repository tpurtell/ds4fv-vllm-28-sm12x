#!/usr/bin/env python3
"""CUDA-hidden concurrency contract for the DeepSeek-V4 renderer tokenizer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import TokenizersBackend

from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.hf import (
    ThreadSafeHFTokenizerMixin,
    get_cached_tokenizer,
)


raw = Tokenizer(
    WordLevel(
        {
            "[UNK]": 0,
            "alpha": 1,
            "beta": 2,
            "gamma": 3,
            "delta": 4,
        },
        unk_token="[UNK]",
    )
)
raw.pre_tokenizer = Whitespace()
hf_tokenizer = TokenizersBackend(tokenizer_object=raw, unk_token="[UNK]")
tokenizer = get_cached_tokenizer(get_deepseek_v4_tokenizer(hf_tokenizer))

config = SimpleNamespace(
    model_config=SimpleNamespace(
        renderer_num_workers=1,
        is_multimodal_model=False,
    ),
    parallel_config=SimpleNamespace(_api_process_rank=0),
    observability_config=None,
)
renderer = DeepseekV4Renderer(config, tokenizer)
pooled = renderer.get_tokenizer()
assert isinstance(pooled, ThreadSafeHFTokenizerMixin)

text = "alpha beta gamma delta " * 1024


def stress(worker: int) -> None:
    for iteration in range(200):
        truncate = (worker + iteration) % 2 == 0
        pooled(
            text,
            truncation=truncate,
            max_length=128 if truncate else None,
            padding=False,
        )
        pooled.apply_chat_template(
            [{"role": "user", "content": "alpha beta"}],
            tokenize=True,
            truncation=not truncate,
            max_length=64,
        )


with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(stress, range(8)))

renderer._executor.shutdown(wait=True)
renderer._mm_executor.shutdown(wait=True)
print("Spark CUDA-hidden DeepSeek-V4 tokenizer concurrency contract passed")
