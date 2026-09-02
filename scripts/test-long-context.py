#!/usr/bin/env python3
"""Cold six-needle retrieval at the 128K DS4FV release boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import urllib.request
import uuid
from pathlib import Path

from ds4fv_benchmark_common import detokenize, health, server_root, tokenize


FACTS = (
    (0.05, "CINDER-05", "azurite-4831"),
    (0.25, "JUNIPER-25", "topaz-7614"),
    (0.50, "LANTERN-50", "cobalt-2097"),
    (0.75, "MARBLE-75", "saffron-6382"),
    (0.95, "ORBIT-95", "willow-1459"),
    (0.99, "QUARTZ-99", "indigo-8726"),
)


def build_exact_prompt(
    base_url: str, model: str, target_tokens: int, nonce: str, timeout: float
) -> tuple[str, list[dict]]:
    prefix = (
        f"Cold DS4FV long-context qualification {nonce}.\n"
        "Remember every KEY=VALUE audit record. All other prose is distractor.\n"
    )
    suffix = (
        "\nEND OF DOCUMENT. Return all six audit records in document order, exactly "
        "as KEY=VALUE, one per line. Return nothing else.\nANSWER:\n"
    )
    filler = (
        "Ordinary archive prose describes quiet rivers, copper clocks, patient "
        "engineers, and slate valleys; it contains no audit key or value.\n"
    )
    prefix_ids = tokenize(base_url, model, prefix, timeout)
    suffix_ids = tokenize(base_url, model, suffix, timeout)
    filler_ids = tokenize(base_url, model, filler, timeout)
    fact_ids = [
        tokenize(base_url, model, f"\nAUDIT RECORD: {key}={value}\n", timeout)
        for _, key, value in FACTS
    ]
    fixed = len(prefix_ids) + len(suffix_ids) + sum(map(len, fact_ids))
    if target_tokens <= fixed + len(FACTS):
        raise ValueError("target token count is too small")

    result = list(prefix_ids)
    placements = []
    for (fraction, key, value), encoded in zip(FACTS, fact_ids, strict=True):
        desired = round(target_tokens * fraction)
        gap = max(1, desired - len(result))
        result.extend((filler_ids * math.ceil(gap / len(filler_ids)))[:gap])
        placements.append(
            {
                "fraction": fraction,
                "key": key,
                "value": value,
                "token_offset_before_record": len(result),
            }
        )
        result.extend(encoded)

    tail = target_tokens - len(result) - len(suffix_ids)
    if tail < 1:
        raise RuntimeError("record placement left no room for the question")
    result.extend((filler_ids * math.ceil(tail / len(filler_ids)))[:tail])
    result.extend(suffix_ids)
    for _ in range(8):
        prompt = detokenize(base_url, model, result, timeout)
        roundtrip = tokenize(base_url, model, prompt, timeout)
        delta = target_tokens - len(roundtrip)
        if delta == 0:
            break
        tail_start = len(result) - len(suffix_ids) - tail
        tail += delta
        if tail < 1:
            raise RuntimeError("canonicalization exhausted the final filler span")
        replacement = (filler_ids * math.ceil(tail / len(filler_ids)))[:tail]
        result = result[:tail_start] + replacement + result[-len(suffix_ids) :]
    else:
        raise RuntimeError("could not canonicalize the exact prompt-token count")
    if len(roundtrip) != target_tokens:
        raise RuntimeError(
            f"prompt roundtrip has {len(roundtrip)} tokens, expected {target_tokens}"
        )
    for _, key, value in FACTS:
        if prompt.count(f"{key}={value}") != 1:
            raise RuntimeError(f"prompt does not contain exactly one {key} record")
    return prompt, placements


def stream_request(base_url: str, payload: dict, timeout: float) -> tuple[str, dict, float, float]:
    request = urllib.request.Request(
        server_root(base_url) + "/v1/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first_token_at = None
    output_parts = []
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
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
                text = choice.get("text") or ""
                if text:
                    first_token_at = first_token_at or time.perf_counter()
                    output_parts.append(text)
    finished = time.perf_counter()
    if first_token_at is None or usage is None:
        raise RuntimeError("stream did not expose output and usage")
    return "".join(output_parts), usage, first_token_at - started, finished - started


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
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=7200.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 8192 <= args.tokens <= 128000:
        parser.error("long-context tokens must stay within 8192..128000")

    health(args.base_url)
    nonce = f"{args.role}-{uuid.uuid4().hex}"
    cache_salt = uuid.uuid4().hex
    started = time.perf_counter()
    prompt, placements = build_exact_prompt(
        args.base_url, args.model, args.tokens, nonce, args.timeout
    )
    built = time.perf_counter()
    output, usage, ttft_seconds, request_seconds = stream_request(
        args.base_url,
        {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "cache_salt": cache_salt,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        args.timeout,
    )
    checks = [
        {"key": key, "expected": value, "passed": f"{key}={value}" in output}
        for _, key, value in FACTS
    ]
    report = {
        "schema": "ds4fv-cold-multi-needle.v1",
        "passed": all(check["passed"] for check in checks),
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "target_prompt_tokens": args.tokens,
        "nonce": nonce,
        "cache_salt": cache_salt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "placements": placements,
        "checks": checks,
        "usage": usage,
        "build_seconds": round(built - started, 3),
        "ttft_seconds": round(ttft_seconds, 3),
        "stream_after_first_token_seconds": round(request_seconds - ttft_seconds, 3),
        "request_seconds": round(request_seconds, 3),
        "output": output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
