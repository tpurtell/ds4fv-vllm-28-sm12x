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
    for fragment in (
        "run_vllm_with_warmup \"native-${model_kind}\"",
        "run_vllm_with_warmup exl3",
        "/tmp/ds4fv-release-ready",
        "DS4FV_STARTUP_WARMUP",
        "DS4FV release startup warmup complete; container is ready.",
    ):
        assert fragment in launcher

    warmup = (ROOT / "scripts/release-warmup.py").read_text()
    for fragment in (
        'choices=("native-text", "native-vision", "exl3")',
        "for block_size in (8, 16, 32, 64, 128, 256)",
        "args.base_url, model, 9500, args.request_timeout",
        "image_counts = (1, 4, 16)",
        '"structured output"',
        '"tool parser"',
        "timed_concurrent_requests(",
        "for index in range(2)",
        "for index in range(4)",
    ):
        assert fragment in warmup

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
    assert "scripts/release-warmup.py /opt/ds4fv/bin/release-warmup" in dockerfile
    assert "scripts/container-healthcheck.py" in dockerfile
    assert "TILELANG_CACHE_DIR=/cache/huggingface/tilelang-cache" in dockerfile
    assert "TRITON_CACHE_DIR=/cache/huggingface/triton-cache" in dockerfile
    assert "--start-period=60m" in dockerfile

    for name in ("launch-two-spark.sh", "launch-one-spark-exl3.sh"):
        host_launcher = (ROOT / "scripts" / name).read_text()
        for fragment in (
            "DS4FV_STARTUP_WARMUP",
            "DS4FV_ENGINE_READY_TIMEOUT_S",
            "DS4FV_STARTUP_WARMUP_TIMEOUT_S",
            "TILELANG_CACHE_DIR",
            "TRITON_CACHE_DIR",
        ):
            assert fragment in host_launcher

    audit = (ROOT / "scripts/audit-startup-jit.py").read_text()
    assert "ds4fv-release-startup-jit-audit.v1" in audit
    assert '"passed": not post_ready_jit' in audit
    print("release harness source smoke passed")


if __name__ == "__main__":
    main()
