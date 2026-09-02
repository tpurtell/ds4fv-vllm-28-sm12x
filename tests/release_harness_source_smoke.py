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
    assert content.ARMS["code"].max_tokens == 512
    assert content.ARMS["topic"].max_tokens == 256
    assert content.ARMS["multilingual"].max_tokens == 384
    assert "exactly three" in content.ARMS["code"].prompt
    assert "150 to 165 words" in content.ARMS["fable"].prompt
    assert "Moral:" in content.ARMS["fable"].prompt
    assert "正好四個單行條列" in content.ARMS["multilingual"].prompt

    valid_json = (
        '{"path":"src/cache.rs","operation":"replace","line_start":41,'
        '"line_end":47,"rationale":"Remove a redundant copy."}'
    )
    for arm_id in ("structured-json-normal", "structured-json-constrained"):
        passed, issues = content.validate_semantic_contract(arm_id, valid_json)
        assert passed, issues
    assert not content.validate_semantic_contract("topic", "- Paging only.")[0]
    truncated = content.compact_record(
        arm_id="hello",
        arm=content.ARMS["hello"],
        repeat=1,
        timed=True,
        raw={
            "content": "Hello",
            "prompt_tokens": 1,
            "completion_tokens": 32,
            "finish_reason": "length",
            "ttft_s": 0.1,
            "decode_s": 0.5,
            "decode_tok_s": 62.0,
        },
    )
    assert not truncated["quality_contract_passed"]
    assert "response hit the output token limit" in truncated[
        "quality_contract_issues"
    ]
    summary = content.summarize(
        [
            {
                "timed": True,
                "arm": "code",
                "category": "code",
                "score_weight": 1.0,
                "decode_tok_s": 50.0,
                "completion_tokens": 32,
                "quality_contract_passed": True,
            },
            {
                "timed": True,
                "arm": "orchid",
                "category": "low-entropy-showcase",
                "score_weight": 1.0,
                "decode_tok_s": 65.0,
                "completion_tokens": 128,
                "orchid_only": True,
                "orchid_minimum_reached": True,
            },
        ]
    )
    assert summary["quality_contract_passed"]
    assert summary["orchid_contract_passed"]

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
        'run_vllm_with_warmup "${warmup_role}"',
        "/tmp/ds4fv-release-ready",
        "DS4FV_STARTUP_WARMUP",
        "DS4FV release startup warmup complete; container is ready.",
    ):
        assert fragment in launcher

    warmup = (ROOT / "scripts/release-warmup.py").read_text()
    for fragment in (
        'choices=("native-text", "native-vision", "exl3", "exl3-vision")',
        "for block_size in (8, 16, 32, 64, 128, 256)",
        "args.base_url, model, 9500, args.request_timeout",
        "image_counts = (1, 4, 16)",
        '"structured output"',
        '"tool parser"',
        "timed_concurrent_requests(",
        "for choice_count in (2, 4)",
        '"temperature": 0.2',
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
    assert '"${role}" == native-vision || "${role}" == exl3-vision' in runner
    assert '"${role}" == exl3' in runner

    for fragment in (
        "DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1",
        "8aab722f04f7e8963af83de5acb16138474e0228",
        "deepseek-v4-flash-vision-exp-exl3-k2.2-d2-v1",
        "DeepseekV4VisionForConditionalGeneration",
        "warmup_role=exl3-vision",
        'configure_dspark_args speculative_args "${dspark_default_tokens}"',
    ):
        assert fragment in launcher

    dockerfile = (ROOT / "Dockerfile").read_text()
    assert 'org.opencontainers.image.revision="${RECIPE_COMMIT}"' in dockerfile
    assert "scripts/release-warmup.py /opt/ds4fv/bin/release-warmup" in dockerfile
    assert "scripts/container-healthcheck.py" in dockerfile
    assert "TILELANG_CACHE_DIR=/cache/huggingface/tilelang-cache" in dockerfile
    assert "TRITON_CACHE_DIR=/cache/huggingface/triton-cache" in dockerfile
    assert "--start-period=60m" in dockerfile
    assert "apply-vllm-long-prefill-jit.py" in dockerfile
    assert "apply-vllm-indexer-workspace.py" in dockerfile
    assert "apply-vllm-dsv4-kv-groups.py" in dockerfile

    long_prefill_patch = (
        ROOT / "patches/apply-vllm-long-prefill-jit.py"
    ).read_text()
    assert '"            query_slice_start=WarmupIntRange(0, 2),\\n"' in (
        long_prefill_patch
    )
    assert '"            query_slice_start=WarmupIntRange(0, 3),\\n"' in (
        long_prefill_patch
    )

    indexer_workspace_patch = (
        ROOT / "patches/apply-vllm-indexer-workspace.py"
    ).read_text()
    assert "max_num_seqs = vllm_config.scheduler_config.max_num_seqs" in (
        indexer_workspace_patch
    )
    assert "max(1, min(40, max_num_seqs))" in indexer_workspace_patch
    assert "self.max_decode_tokens = max(" in indexer_workspace_patch
    assert "scheduler_config.max_num_seqs * next_n" in indexer_workspace_patch
    assert "max_cudagraph_capture_size" in indexer_workspace_patch
    assert "num_decode_tokens <= self.max_decode_tokens" in (
        indexer_workspace_patch
    )
    assert "self.c128a_max_decode_tokens = max(" in indexer_workspace_patch
    assert "num_decode_tokens <= self.c128a_max_decode_tokens" in (
        indexer_workspace_patch
    )
    assert "prefill buffer below correctly retains the full 8K budget" in (
        indexer_workspace_patch
    )

    kv_group_patch = (
        ROOT / "patches/apply-vllm-dsv4-kv-groups.py"
    ).read_text()
    for fragment in (
        "max_groups = 5",
        "candidate_strides = sorted(",
        "block_stride * required_blocks",
        "for index in range(tuple_count)",
        "num_tuple_groups = cdiv(tuple_count, tuple_width)",
        "layer_tuples[group_index::num_tuple_groups]",
        "required_bytes = packed_stride * sum(",
        '"block_stride=%d bytes, one_request=%d bytes"',
        "block_stride, _ = _get_packed_kv_cache_layout(kv_cache_groups)",
    ):
        assert fragment in kv_group_patch

    vision = (
        ROOT / "overlay/vllm/model_executor/models/deepseek_v4_vision.py"
    ).read_text()
    assert "def _encode_image_batch(" in vision
    assert "patch_batch = torch.stack(" in vision
    assert "q.transpose(-3, -2)" in vision
    assert '"model.": "language_model.model."' in vision

    vision_patch = (ROOT / "patches/apply-vllm-vision.py").read_text()
    assert '"deepseek_v4": ("hash_moe", "moe")' in vision_patch
    assert '"ALLOWED_MLP_LAYER_TYPES"' in vision_patch

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

    one_spark_launcher = (ROOT / "scripts/launch-one-spark-exl3.sh").read_text()
    assert 'if [[ "${model_kind}" == vision ]]' in one_spark_launcher
    assert "gpu_memory_utilization=0.86" in one_spark_launcher
    assert "max_model_len=500000" in one_spark_launcher
    assert "max_num_batched_tokens=2048" in one_spark_launcher
    assert "gpu_memory_default=0.86" in launcher
    assert "max_model_len_default=500000" in launcher
    assert "max_num_batched_tokens_default=2048" in launcher

    audit = (ROOT / "scripts/audit-startup-jit.py").read_text()
    assert "ds4fv-release-startup-jit-audit.v1" in audit
    assert '"passed": not post_ready_jit' in audit
    print("release harness source smoke passed")


if __name__ == "__main__":
    main()
