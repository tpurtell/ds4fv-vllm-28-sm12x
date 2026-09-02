#!/usr/bin/env python3
"""Prove multimodal prefix reuse without reusing KV across changed images."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from ds4fv_benchmark_common import health, server_root


def load_vision_helpers():
    path = Path(__file__).with_name("test-native-vision-vllm.py")
    spec = importlib.util.spec_from_file_location("ds4fv_vision_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def prefix_hits(base_url: str, timeout: float) -> float:
    with urllib.request.urlopen(
        server_root(base_url) + "/metrics", timeout=timeout
    ) as response:
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
        "--role", required=True, choices=("native-vision", "exl3-vision")
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    health(args.base_url)
    vision = load_vision_helpers()
    cache_salt = f"vision-prefix-replay-{uuid.uuid4().hex}"

    def request(values: list[int]) -> dict:
        content = [
            {
                "type": "image_url",
                "image_url": {"url": vision.image_url(value)},
            }
            for value in values
        ]
        content.append(
            {
                "type": "text",
                "text": (
                    "Each image contains one large decimal digit. Read all four "
                    "images in order and return only the comma-separated digits, "
                    "with no prose."
                ),
            }
        )
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 64,
            "cache_salt": cache_salt,
            "chat_template_kwargs": {
                "thinking": False,
                "reasoning_effort": "low",
            },
        }
        req = urllib.request.Request(
            server_root(args.base_url) + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                status = response.status
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = json.loads(exc.read().decode("utf-8", errors="replace"))
        elapsed = time.perf_counter() - started
        if status != 200:
            raise RuntimeError(f"Vision replay failed with HTTP {status}: {body}")
        output = body["choices"][0]["message"].get("content") or ""
        observed = [int(value) for value in re.findall(r"\d+", output)]
        return {
            "passed": observed == values,
            "expected": values,
            "observed": observed,
            "response": output,
            "request_seconds": elapsed,
            "usage": body.get("usage"),
        }

    before_hits = prefix_hits(args.base_url, args.timeout)
    seed = request([1, 2, 3, 4])
    after_seed_hits = prefix_hits(args.base_url, args.timeout)
    exact_replay = request([1, 2, 3, 4])
    after_replay_hits = prefix_hits(args.base_url, args.timeout)
    changed_images = request([5, 6, 7, 8])
    after_changed_hits = prefix_hits(args.base_url, args.timeout)
    replay_hit_delta = after_replay_hits - after_seed_hits

    report = {
        "schema": "ds4fv-vision-prefix-replay.v1",
        "passed": (
            seed["passed"]
            and exact_replay["passed"]
            and changed_images["passed"]
            and replay_hit_delta > 0
        ),
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "cache_salt": cache_salt,
        "prefix_cache_hits_before": before_hits,
        "prefix_cache_hits_after_seed": after_seed_hits,
        "prefix_cache_hits_after_exact_replay": after_replay_hits,
        "prefix_cache_hits_after_changed_images": after_changed_hits,
        "exact_replay_hit_delta": replay_hit_delta,
        "seed": seed,
        "exact_replay": exact_replay,
        "changed_images_collision_guard": changed_images,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
