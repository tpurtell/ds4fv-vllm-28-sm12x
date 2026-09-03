#!/usr/bin/env python3
"""Give the DeepSeek-V4 renderer vLLM's thread-safe HF tokenizer pool."""

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
        raise SystemExit("usage: apply-vllm-dsv4-tokenizer-threadsafe.py VLLM_ROOT")

    root = Path(sys.argv[1]).resolve()
    path = root / "renderers/deepseek_v4.py"
    if not path.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")

    replace_once(
        path,
        '''# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
''',
        '''# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

from vllm.config import VllmConfig
''',
        "copy import",
    )
    replace_once(
        path,
        '''from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.utils.async_utils import make_async
''',
        '''from vllm.tokenizers.deepseek_v4 import DeepseekV4Tokenizer
from vllm.tokenizers.hf import maybe_make_thread_pool
from vllm.utils.async_utils import make_async
''',
        "thread-safe tokenizer helper import",
    )
    replace_once(
        path,
        '''    ) -> None:
        super().__init__(config, tokenizer)

        self._apply_chat_template_async = make_async(
''',
        '''    ) -> None:
        # DeepSeek-V4 still uses a Transformers fast tokenizer underneath its
        # custom chat encoder. Chat rendering, prompt tokenization, and the
        # multimodal processor can run concurrently, while Transformers
        # mutates Rust truncation/padding state on every call. Mirror
        # HfRenderer's independent-backend pool instead of sharing one backend.
        tokenizer = copy.copy(tokenizer)
        super().__init__(config, tokenizer)

        if self.tokenizer is not None:
            maybe_make_thread_pool(
                self.tokenizer, config.model_config.renderer_num_workers + 1
            )

        self._apply_chat_template_async = make_async(
''',
        "DeepSeek-V4 renderer tokenizer pool",
    )


if __name__ == "__main__":
    main()
