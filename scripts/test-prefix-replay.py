#!/usr/bin/env python3
"""Replay one exact 128K prompt and verify real prefix-cache hits."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import urllib.request
import uuid
from pathlib import Path

from ds4fv_benchmark_common import health, server_root


def load_long_context():
    path = Path(__file__).with_name("test-long-context.py")
    spec = importlib.util.spec_from_file_location("ds4fv_long_context", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prefix_hits(base_url: str) -> float:
    with urllib.request.urlopen(server_root(base_url) + "/metrics", timeout=300) as response:
        body = response.read().decode("utf-8", errors="replace")
    total = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total{"):
            total += float(line.rsplit(" ", 1)[-1])
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--role", required=True, choices=("native-vision", "exl3", "exl3-vision")
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--tokens", type=int, default=128000)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 8192 <= args.tokens <= 128000:
        parser.error("prefix replay tokens must stay within 8192..128000")

    health(args.base_url)
    long_context = load_long_context()
    nonce = f"prefix-replay-{uuid.uuid4().hex}"
    cache_salt = uuid.uuid4().hex
    prompt, placements = long_context.build_exact_prompt(
        args.base_url, args.model, args.tokens, nonce, args.timeout
    )
    before_hits = prefix_hits(args.base_url)
    passes = []
    expected = [f"{key}={value}" for _, key, value in long_context.FACTS]
    for pass_index in range(2):
        output, usage, ttft_seconds, request_seconds = long_context.stream_request(
            args.base_url,
            {
                "model": args.model,
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0,
                "cache_salt": cache_salt,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            args.timeout,
        )
        passes.append(
            {
                "pass": pass_index + 1,
                "passed": all(record in output for record in expected),
                "ttft_seconds": ttft_seconds,
                "request_seconds": request_seconds,
                "usage": usage,
                "output": output,
            }
        )
    after_hits = prefix_hits(args.base_url)
    report = {
        "schema": "ds4fv-prefix-replay.v1",
        "passed": all(item["passed"] for item in passes) and after_hits > before_hits,
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "target_prompt_tokens": args.tokens,
        "cache_salt": cache_salt,
        "placements": placements,
        "prefix_cache_hits_before": before_hits,
        "prefix_cache_hits_after": after_hits,
        "prefix_cache_hit_delta": after_hits - before_hits,
        "passes": passes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
