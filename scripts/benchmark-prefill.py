#!/usr/bin/env python3
"""Measure cold C1 prefill/TTFT at exact unique DeepSeek prompt lengths."""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
import time
from pathlib import Path
from urllib.parse import urlparse

from ds4fv_benchmark_common import detokenize, health, server_root, tokenize


def exact_prompt(
    base_url: str, model: str, target_tokens: int, nonce: str, timeout: float
) -> str:
    header = (
        f"Independent DS4FV prefill qualification {nonce}. This nonce makes the "
        "first cache block unique; ignore it.\n"
    )
    unit = (
        "Slate rivers cross quiet valleys while copper clocks mark patient hours. "
        "This is ordinary benchmark filler with no instructions.\n"
    )
    header_ids = tokenize(base_url, model, header, timeout)
    if target_tokens <= len(header_ids):
        raise ValueError(f"target {target_tokens} is too small")
    unit_ids = tokenize(base_url, model, unit, timeout)
    source_ids = tokenize(
        base_url,
        model,
        header + unit * (target_tokens // max(1, len(unit_ids)) + 3),
        timeout,
    )
    wanted = target_tokens
    for _ in range(8):
        prompt = detokenize(base_url, model, source_ids[:wanted], timeout)
        actual = len(tokenize(base_url, model, prompt, timeout))
        if actual == target_tokens:
            return prompt
        wanted += target_tokens - actual
        if wanted <= len(header_ids) or wanted > len(source_ids):
            break
    raise RuntimeError(f"could not construct an exact {target_tokens}-token prompt")


def time_to_first_token(
    base_url: str, model: str, prompt: str, timeout: float
) -> dict:
    parsed = urlparse(server_root(base_url))
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "min_tokens": 1,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    started = time.perf_counter()
    connection.request(
        "POST",
        parsed.path.rstrip("/") + "/v1/completions",
        body=json.dumps(payload),
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error = response.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {error}")
    first_token = None
    usage = None
    while True:
        raw_line = response.readline()
        if not raw_line:
            break
        observed = time.perf_counter()
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or ():
            token_ids = choice.get("token_ids")
            if first_token is None and isinstance(token_ids, list) and token_ids:
                first_token = observed
    connection.close()
    if first_token is None or usage is None:
        raise RuntimeError("stream did not expose both a token ID and usage")
    prompt_tokens = int(usage["prompt_tokens"])
    ttft = first_token - started
    return {
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": ttft,
        "effective_prompt_tokens_per_second": prompt_tokens / ttft,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", required=True, choices=("native-vision", "exl3"))
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument(
        "--prompt-tokens", nargs="+", type=int, default=(8192, 16384, 32768, 65536, 128000)
    )
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.prompt_tokens) < 512 or max(args.prompt_tokens) > 128000:
        parser.error("prompt points must stay within 512..128000 tokens")
    if args.runs < 1 or args.timeout <= 0:
        parser.error("runs and timeout must be positive")

    health(args.base_url)
    warmup = exact_prompt(args.base_url, args.model, 512, "suite-warmup", args.timeout)
    time_to_first_token(args.base_url, args.model, warmup, args.timeout)
    report = {
        "schema": "ds4fv-prefill-depth.v1",
        "method": (
            "C1 exact-length unique prompts; client request to first streamed token; "
            "server tokenization and one-token handoff included; every depth is "
            "warmed independently and no earlier prefix can be reused"
        ),
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "runs_per_point": args.runs,
        "points": [],
    }

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for target in args.prompt_tokens:
        depth_warmup = exact_prompt(
            args.base_url, args.model, target, f"{args.role}-{target}-warmup", args.timeout
        )
        time_to_first_token(args.base_url, args.model, depth_warmup, args.timeout)
        runs = []
        for run_index in range(args.runs):
            prompt = exact_prompt(
                args.base_url,
                args.model,
                target,
                f"{args.role}-{target}-{run_index}-{time.time_ns()}",
                args.timeout,
            )
            result = time_to_first_token(args.base_url, args.model, prompt, args.timeout)
            if result["prompt_tokens"] != target:
                raise RuntimeError(
                    f"server counted {result['prompt_tokens']} tokens, expected {target}"
                )
            runs.append(result)
            print(
                f"{target:,} run {run_index + 1}/{args.runs}: "
                f"{result['effective_prompt_tokens_per_second']:.1f} tok/s, "
                f"TTFT {result['ttft_seconds']:.3f} s",
                flush=True,
            )
        rates = [run["effective_prompt_tokens_per_second"] for run in runs]
        ttfts = [run["ttft_seconds"] for run in runs]
        report["points"].append(
            {
                "prompt_tokens": target,
                "effective_prompt_tokens_per_second": {
                    "median": statistics.median(rates),
                    "min": min(rates),
                    "max": max(rates),
                },
                "ttft_seconds": {
                    "median": statistics.median(ttfts),
                    "min": min(ttfts),
                    "max": max(ttfts),
                },
                "runs": runs,
            }
        )
        persist()
    report["passed"] = True
    persist()


if __name__ == "__main__":
    main()
