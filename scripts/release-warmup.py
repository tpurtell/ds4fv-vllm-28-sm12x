#!/usr/bin/env python3
"""Exercise real DS4FV serving paths before Docker reports healthy."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor


PROMPT = """You are editing an async Python task runner. Fix the cancellation and
exception-handling bugs in this implementation, preserve result ordering, and add precise type
hints. Return only the complete replacement Python module.

```python
import asyncio

async def run_all(factories, limit=8):
    sem = asyncio.Semaphore(limit)
    results = []
    async def one(factory):
        async with sem:
            results.append(await factory())
    tasks = [asyncio.create_task(one(factory)) for factory in factories]
    try:
        await asyncio.gather(*tasks)
    except Exception:
        for task in tasks:
            task.cancel()
    return results
```
"""


def request_json(
    base_url: str, path: str, payload: dict | None, timeout: float
) -> dict:
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise SystemExit(
            f"{path} returned HTTP {error.code}: {detail[:1000]}"
        ) from error


def request_ok(base_url: str, path: str, timeout: float) -> None:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as response:
        response.read()


def server_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def require_choices(result: dict, expected: int, label: str) -> None:
    choices = result.get("choices") or []
    if len(choices) != expected:
        raise SystemExit(f"{label} returned {len(choices)} choices, expected {expected}")


def timed_request(
    label: str,
    base_url: str,
    path: str,
    payload: dict,
    timeout: float,
    *,
    expected_choices: int = 1,
) -> dict:
    started = time.perf_counter()
    result = request_json(base_url, path, payload, timeout)
    require_choices(result, expected_choices, label)
    print(
        f"DS4FV release startup {label} completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return result


def timed_concurrent_requests(
    label: str,
    base_url: str,
    path: str,
    payloads: list[dict],
    timeout: float,
) -> list[dict]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(payloads)) as executor:
        futures = [
            executor.submit(request_json, base_url, path, payload, timeout)
            for payload in payloads
        ]
        results = [future.result() for future in futures]
    for index, result in enumerate(results):
        require_choices(result, 1, f"{label} request {index + 1}")
    print(
        f"DS4FV release startup {label} completed in "
        f"{time.perf_counter() - started:.2f}s",
        flush=True,
    )
    return results


def exact_token_prefix(
    base_url: str, model: str, tokens: int, timeout: float
) -> list[int]:
    unit = request_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "prompt": "Slate rivers cross quiet valleys while copper clocks mark patient hours.\n",
            "add_special_tokens": False,
        },
        60,
    ).get("tokens")
    if not isinstance(unit, list) or not unit:
        raise SystemExit("startup long-prefill warmup could not tokenize filler")
    return (unit * ((tokens + len(unit) - 1) // len(unit)))[:tokens]


def png_data_url() -> str:
    size = 448
    pixels = bytes((255, 255, 255)) * size * size
    scanlines = b"".join(
        b"\0" + pixels[row * size * 3 : (row + 1) * size * 3]
        for row in range(size)
    )

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=("native-text", "native-vision", "exl3", "exl3-vision"),
    )
    parser.add_argument(
        "--ready-timeout",
        type=float,
        default=float(os.getenv("DS4FV_ENGINE_READY_TIMEOUT_S", "3600")),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("DS4FV_STARTUP_WARMUP_TIMEOUT_S", "1800")),
    )
    parser.add_argument("--passes", type=int, default=4)
    args = parser.parse_args()

    if args.passes < 1:
        raise SystemExit("--passes must be at least one")

    deadline = time.monotonic() + args.ready_timeout
    while True:
        if not server_alive(args.server_pid):
            raise SystemExit("vLLM exited before startup warmup")
        try:
            request_ok(args.base_url, "/health", 3)
            break
        except (OSError, urllib.error.URLError):
            if time.monotonic() >= deadline:
                raise SystemExit("timed out waiting for vLLM's internal health endpoint")
            time.sleep(2)

    models = request_json(args.base_url, "/v1/models", None, 30).get("data") or []
    if not models or not isinstance(models[0].get("id"), str):
        raise SystemExit("vLLM returned no served model for startup warmup")
    model = models[0]["id"]
    nonce = f"{os.getpid()}-{uuid.uuid4().hex}"

    # Mia's post-ready sweep covered the exact scheduled-token buckets used by
    # DeepSeek's speculative-input preparation kernel. Do the same before the
    # release marker. For the qualified K3/K5 profiles, scheduled tokens plus
    # one sampled target token and the draft depth land exactly on each
    # power-of-two specialization from 8 through 256.
    if os.getenv("ENABLE_DSPARK", "1") == "1":
        default_draft_tokens = (
            3 if args.role in ("native-vision", "exl3-vision") else 5
        )
        raw_draft_tokens = os.getenv("DSPARK_TOKENS", "").strip()
        draft_tokens = (
            int(raw_draft_tokens) if raw_draft_tokens else default_draft_tokens
        )
        if draft_tokens < 1:
            raise SystemExit("DSPARK_TOKENS must be positive during startup warmup")
        overhead = 1 + draft_tokens
        for block_size in (8, 16, 32, 64, 128, 256):
            scheduled_tokens = max(1, block_size - overhead)
            ladder_tokens = exact_token_prefix(
                args.base_url, model, scheduled_tokens, args.request_timeout
            )
            timed_request(
                f"DSpark BLOCK {block_size}",
                args.base_url,
                "/v1/completions",
                {
                    "model": model,
                    "prompt": ladder_tokens,
                    "add_special_tokens": False,
                    "max_tokens": 1,
                    "min_tokens": 1,
                    "ignore_eos": True,
                    "temperature": 0,
                    "cache_prompt": False,
                    "cache_salt": f"ds4fv-release-ladder-{block_size}-{nonce}",
                },
                args.request_timeout,
            )

    rendered = request_json(
        args.base_url,
        "/v1/chat/completions/render",
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "chat_template_kwargs": {"thinking": False},
        },
        60,
    ).get("token_ids")
    if not isinstance(rendered, list) or not rendered:
        raise SystemExit("startup rendered-chat warmup did not return token IDs")

    base_completion = {
        "model": model,
        "prompt": rendered,
        "add_special_tokens": False,
        "max_tokens": 64,
        "min_tokens": 64,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 20260902,
        "cache_prompt": False,
        "cache_salt": f"ds4fv-release-rendered-{nonce}",
    }
    timed_request(
        "greedy C1",
        args.base_url,
        "/v1/completions",
        {**base_completion, "n": 1},
        args.request_timeout,
    )
    timed_concurrent_requests(
        "greedy C2",
        args.base_url,
        "/v1/completions",
        [
            {
                **base_completion,
                "cache_salt": f"ds4fv-release-c2-{index}-{nonce}",
            }
            for index in range(2)
        ],
        args.request_timeout,
    )

    # The release clients express concurrency as one OpenAI request with
    # multiple choices (n=2/n=4).  That creates a different request-to-token
    # mapping from several simultaneous n=1 requests and, on DeepSeek's sparse
    # path, can select a distinct Triton pointer specialization.  Exercise both
    # batched-choice shapes before the ready marker.
    for choice_count in (2, 4):
        timed_request(
            f"greedy N{choice_count}",
            args.base_url,
            "/v1/completions",
            {
                **base_completion,
                "n": choice_count,
                "temperature": 0.2,
                "cache_salt": (
                    f"ds4fv-release-n{choice_count}-{nonce}"
                ),
            },
            args.request_timeout,
            expected_choices=choice_count,
        )

    long_tokens = exact_token_prefix(args.base_url, model, 8192, args.request_timeout)
    timed_request(
        "8K prefill",
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": long_tokens,
            "add_special_tokens": False,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": False,
            "cache_salt": f"ds4fv-release-prefill-8192-{nonce}",
        },
        args.request_timeout,
    )

    # Cross the 8192-token scheduler boundary so both a full chunk and its
    # tail shape are materialized before Docker can report healthy.
    chunk_crossing_tokens = exact_token_prefix(
        args.base_url, model, 9500, args.request_timeout
    )
    timed_request(
        "9.5K chunk-crossing prefill",
        args.base_url,
        "/v1/completions",
        {
            "model": model,
            "prompt": chunk_crossing_tokens,
            "add_special_tokens": False,
            "max_tokens": 1,
            "min_tokens": 1,
            "ignore_eos": True,
            "temperature": 0,
            "cache_prompt": False,
            "cache_salt": f"ds4fv-release-prefill-9500-{nonce}",
        },
        args.request_timeout,
    )

    structured_prompt = "Return only JSON with integer key warmup set to 1."
    timed_request(
        "structured output",
        args.base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": structured_prompt}],
            "temperature": 0,
            "max_tokens": 32,
            "chat_template_kwargs": {"thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "warmup",
                    "schema": {
                        "type": "object",
                        "properties": {"warmup": {"type": "integer"}},
                        "required": ["warmup"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        args.request_timeout,
    )

    tool_result = timed_request(
        "tool parser",
        args.base_url,
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": "What's the weather in Berlin right now?"}
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for one city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                            "additionalProperties": False,
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 128,
            "chat_template_kwargs": {"thinking": False},
        },
        args.request_timeout,
    )
    if not (tool_result["choices"][0].get("message", {}).get("tool_calls") or []):
        raise SystemExit("startup tool-parser warmup produced no tool call")

    if args.role in ("native-vision", "exl3-vision"):
        image_url = png_data_url()
        image_counts = (1, 4, 16)
        for image_count in image_counts:
            content = [
                {
                    "type": "text",
                    "text": f"Acknowledge these {image_count} blank images briefly.",
                }
            ]
            content.extend(
                {"type": "image_url", "image_url": {"url": image_url}}
                for _ in range(image_count)
            )
            timed_request(
                f"Vision {image_count}-image",
                args.base_url,
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0,
                    "max_tokens": 32,
                    "chat_template_kwargs": {"thinking": False},
                },
                args.request_timeout,
            )

    for pass_index in range(args.passes):
        timed_concurrent_requests(
            f"C4 pass {pass_index + 1}/{args.passes}",
            args.base_url,
            "/v1/completions",
            [
                {
                    **base_completion,
                    "max_tokens": 256,
                    "min_tokens": 256,
                    "cache_salt": (
                        f"ds4fv-release-c4-{pass_index}-{index}-{nonce}"
                    ),
                }
                for index in range(4)
            ],
            args.request_timeout,
        )
        time.sleep(1)


if __name__ == "__main__":
    main()
