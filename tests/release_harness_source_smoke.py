#!/usr/bin/env python3
"""Source-only release harness checks; this file must never import vLLM/CUDA."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_content_benchmark():
    path = ROOT / "scripts/benchmark-content-types.py"
    spec = importlib.util.spec_from_file_location("ds4fv_content_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    content = load_content_benchmark()
    assert sum(arm.score_weight for arm in content.ARMS.values()) == 7.0
    assert content.ARMS["structured-json-normal"].score_weight == 0.5
    assert content.ARMS["structured-json-constrained"].score_weight == 0.5
    assert content.ARMS["topic"].max_tokens == 384
    assert content.ARMS["multilingual"].max_tokens == 384

    valid_json = (
        '{"path":"src/cache.rs","operation":"replace","line_start":41,'
        '"line_end":47,"rationale":"Remove a redundant copy."}'
    )
    for arm_id in ("structured-json-normal", "structured-json-constrained"):
        passed, issues = content.validate_semantic_contract(arm_id, valid_json)
        assert passed, issues
    assert not content.validate_semantic_contract("topic", "- Paging only.")[0]

    launcher = (ROOT / "scripts/start-native.sh").read_text()
    for fragment in (
        "--tokenizer-mode deepseek_v4",
        "--tool-call-parser deepseek_v4",
        "--enable-auto-tool-choice",
        "--reasoning-parser deepseek_v4",
        "--limit-mm-per-prompt '{\"image\":16}'",
    ):
        assert fragment in launcher

    runner = (ROOT / "scripts/run-release-suite.sh").read_text()
    for script in (
        "benchmark-decode.py",
        "benchmark-prefill.py",
        "benchmark-content-types.py",
        "test-tool-call.py",
        "test-native-vision-vllm.py",
        "test-long-context.py",
        "test-prefix-replay.py",
        "soak-api.py",
    ):
        assert script in runner
    assert "docker run" not in runner
    assert "vllm serve" not in runner

    dockerfile = (ROOT / "Dockerfile").read_text()
    assert 'org.opencontainers.image.revision="${RECIPE_COMMIT}"' in dockerfile
    print("release harness source smoke passed")


if __name__ == "__main__":
    main()
