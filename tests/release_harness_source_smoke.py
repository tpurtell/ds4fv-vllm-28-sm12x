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
    assert not summary["structured_contract_passed"]
    assert summary["structured_contract_passes"] == 0
    assert summary["structured_contract_total"] == 0
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

    dockerfile = (ROOT / "Dockerfile").read_text()
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
        "test-vision-prefix-replay.py",
        "test-long-context.py",
        "test-prefix-replay.py",
        "soak-api.py",
    ):
        assert script in runner
    assert "docker run" not in runner
    assert "vllm serve" not in runner
    assert '"${role}" == native-vision || "${role}" == exl3-vision' in runner
    assert '--role "${role}"' in runner
    assert '"kv_cache_dtype": kv_cache_dtype' in runner
    assert 'content_contract_floor=38' in runner
    assert 'content_contract_floor=34' in runner
    assert '--minimum-contract-passes "${content_contract_floor}"' in runner
    assert '--dspark-tokens "${dspark_tokens}"' in runner
    assert "--draft-sample-method greedy" in runner

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
    assert "apply-vllm-dcp-swa.py" in dockerfile
    assert "apply-vllm-dsv4-nvfp4.py" in dockerfile
    assert "apply-vllm-dcp-dsv4.py" in dockerfile
    assert "apply-vllm-dcp-rate-aware.py" in dockerfile
    assert "apply-vllm-dsv4-tokenizer-threadsafe.py" in dockerfile
    assert "3fc8d1491d1313c0ca64b2b95772972b7f42ee9d" in dockerfile
    assert "tests/spark_b12x_no_gpu_smoke.py" in dockerfile
    assert "tests/spark_dcp_swa_no_gpu_smoke.py" in dockerfile
    assert "tests/spark_dcp_dsv4_no_gpu_smoke.py" in dockerfile
    assert "tests/spark_dcp_rate_aware_no_gpu_smoke.py" in dockerfile
    assert "tests/spark_dsv4_tokenizer_threadsafe_no_gpu.py" in dockerfile
    assert "tests/spark_vision_layout_hash_no_gpu_smoke.py" in dockerfile
    assert "CUDA_VISIBLE_DEVICES='' python3 /opt/ds4fv/tests/spark_dcp_swa_no_gpu_smoke.py" in dockerfile

    release_suite = (ROOT / "scripts/run-release-suite.sh").read_text()
    for fragment in (
        "tool-eval-bench.json",
        "--parallel 1",
        "--temperature 0.0",
        "--max-turns 8",
        'result.get("total_scenarios") != 69',
        'scores.get("max_points") != 138',
        "2.3.2.dev3+g5df1e9e0c",
        'tool_eval_version_output=$("${tool_eval_cmd[@]}" --version 2>&1)',
        "tool-eval-bench result version mismatch",
    ):
        assert fragment in release_suite

    dcp_swa_patch = (ROOT / "patches/apply-vllm-dcp-swa.py").read_text()
    for fragment in (
        "effective_block_size = self.block_size * dcp_world_size",
        "kv_cache_spec.block_size * dcp_world_size",
        "(FullAttentionSpec, SlidingWindowSpec, MambaSpec)",
        'admission_kwargs["dcp_world_size"]',
    ):
        assert fragment in dcp_swa_patch

    dcp_dsv4_patch = (ROOT / "patches/apply-vllm-dcp-dsv4.py").read_text()
    for fragment in (
        "Compression is semantic: first select the completed C4/C128 record",
        "A real query row exists on every DCP rank",
        "self.dcp_manager = MLADCPManager(",
        "gathered_query = self.dcp_manager.query_gather(",
        "local_query.contiguous()",
        "def _dcp_ag_rs_combine_into(",
        "pynccl_comm.reduce_scatter(destination, packed_output)",
        "combined = self.dcp_manager.combine(",
        'return_lse=True, lse_scale="base2"',
        "self.dcp_world_size if self.compress_ratio == 4 else 1",
        "supports_dcp_with_varlen=True",
        "ctx_virtual_block_size = block_size * CP_SIZE",
        "q_virtual_block_size = block_size * CP_SIZE",
        "self.block_tables.cp_size,",
    ):
        assert fragment in dcp_dsv4_patch
    assert "def _dcp_head_major_query_gather(" not in dcp_dsv4_patch

    dcp_rate_aware_patch = (
        ROOT / "patches/apply-vllm-dcp-rate-aware.py"
    ).read_text()
    for fragment in (
        "get_kv_cache_dcp_world_size(",
        "split the original mixed full-MLA family",
        "self.block_tables.kv_cache_cp_sizes[gid]",
        "if self.block_tables.kv_cache_cp_sizes[gid] > 1",
        'model_version="deepseek_v4"',
        "def _swa_lengths_for_shard(",
        "swa_lens = _swa_lengths_for_shard(self, swa_lens)",
    ):
        assert fragment in dcp_rate_aware_patch

    tokenizer_patch = (
        ROOT / "patches/apply-vllm-dsv4-tokenizer-threadsafe.py"
    ).read_text()
    for fragment in (
        "from vllm.tokenizers.hf import maybe_make_thread_pool",
        "tokenizer = copy.copy(tokenizer)",
        "config.model_config.renderer_num_workers + 1",
    ):
        assert fragment in tokenizer_patch

    nvfp4_patch = (ROOT / "patches/apply-vllm-dsv4-nvfp4.py").read_text()
    nvfp4_producer = (
        ROOT
        / "overlay/vllm/models/deepseek_v4/common/ops/nvfp4_ds_mla.py"
    ).read_text()
    for fragment in (
        '"nvfp4_ds_mla": torch.uint8',
        "return (num_blocks, block_size, 432)",
        'record_bytes = 432 if self.kv_cache_dtype == "nvfp4_ds_mla" else 584',
        '2 if self.kv_cache_dtype == "nvfp4_ds_mla" else None',
        'False if self.kv_cache_dtype == "nvfp4_ds_mla" else None',
        "b12x_prefill_width in (128, 512, 1024, 2048)",
        "nvfp4_ds_mla requires the B12x linear/attention backend",
    ):
        assert fragment in nvfp4_patch
    for fragment in (
        "DSV4_NVFP4_RECORD_BYTES = 432",
        "DSV4_NVFP4_SCALE_OFFSET = 256",
        "DSV4_NVFP4_PAD_OFFSET = 288",
        "DSV4_NVFP4_ROPE_OFFSET = 304",
        "cache_block_stride",
        "cache_token_stride",
    ):
        assert fragment in nvfp4_producer

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
        "vllm_config.parallel_config.tensor_parallel_size > 1",
        "else 5",
        "candidate_strides = sorted(",
        "block_stride * required_blocks",
        "vllm_config.parallel_config.tensor_parallel_size == 1",
        "and tuple_width >= grouped_spec.get_num_layer_tuples()",
        "for index in range(tuple_count)",
        "num_tuple_groups = cdiv(tuple_count, tuple_width)",
        "layer_tuples[group_index::num_tuple_groups]",
        "required_bytes = packed_stride * sum(",
        '"block_stride=%d bytes, one_request=%d bytes"',
        "block_stride, _ = _get_packed_kv_cache_layout(kv_cache_groups)",
    ):
        assert fragment in kv_group_patch

    b12x_patch = (ROOT / "patches/apply-vllm-b12x.py").read_text()
    assert "patch_b12x_wide_dual_prefill" not in b12x_patch
    b12x_stream_patch = (
        ROOT / "patches/apply-vllm-b12x-shared-stream.py"
    ).read_text()
    assert "disable_aux_stream_overlap=self._uses_b12x_moe_kernel" in (
        b12x_stream_patch
    )
    assert "or disable_aux_stream_overlap" in b12x_stream_patch
    assert "apply-vllm-b12x-shared-stream.py" in dockerfile
    b12x_configured_stream_patch = (
        ROOT / "patches/apply-vllm-b12x-configured-stream.py"
    ).read_text()
    assert 'getattr(kernel_config, "moe_backend", None) == "b12x"' in (
        b12x_configured_stream_patch
    )
    assert "apply-vllm-b12x-configured-stream.py" in dockerfile

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
    assert '-e KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"' in one_spark_launcher
    assert "gpu_memory_default=0.86" in launcher
    assert "max_model_len_default=500000" in launcher
    assert "max_num_batched_tokens_default=2048" in launcher
    assert 'role=${DS4FV_ROLE:-exl3}' in launcher
    assert 'local model_kind=${MODEL_KIND:-vision}' in launcher
    assert '--decode-context-parallel-size "${DCP_SIZE:-1}"' in launcher
    assert '--dcp-comm-backend "${dcp_comm_backend}"' in launcher
    assert 'DCP_COMM_BACKEND:-ag_rs' in launcher
    assert '--max-model-len "${MAX_MODEL_LEN:-500000}"' in launcher
    assert 'if [[ "${model_kind}" == vision ||' not in launcher
    assert launcher.count('if [[ "${ENABLE_PREFIX_CACHING:-1}" == 1 ]]') == 2

    two_spark_launcher = (ROOT / "scripts/launch-two-spark.sh").read_text()
    assert '-e MODEL_KIND="${MODEL_KIND:-vision}"' in two_spark_launcher
    assert '-e DCP_SIZE="${DCP_SIZE:-1}"' in two_spark_launcher
    assert '-e MAX_MODEL_LEN="${MAX_MODEL_LEN:-500000}"' in two_spark_launcher
    assert (
        '-e VLLM_DISABLE_SHARED_EXPERTS_STREAM="${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}"'
        in two_spark_launcher
    )
    assert (
        '-e VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD="${VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD:-1024}"'
        in two_spark_launcher
    )
    assert 'model_kind=${MODEL_KIND:-vision}' in one_spark_launcher
    assert "max_model_len=500000" in one_spark_launcher

    audit = (ROOT / "scripts/audit-startup-jit.py").read_text()
    assert "ds4fv-release-startup-jit-audit.v1" in audit
    assert '"passed": not post_ready_jit' in audit
    print("release harness source smoke passed")


if __name__ == "__main__":
    main()
