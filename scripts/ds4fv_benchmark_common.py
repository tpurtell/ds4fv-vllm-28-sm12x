#!/usr/bin/env python3
"""Dependency-free helpers shared by the DS4FV HTTP release benchmarks."""

from __future__ import annotations

import http.client
import json
import statistics
import time
import urllib.request
from urllib.parse import urlparse


CODE_AGENT_PROMPT = """You are editing an async Python task runner. Fix the cancellation and
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


def server_root(base_url: str) -> str:
    return base_url.rstrip("/").removesuffix("/v1")


def post_json(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        server_root(base_url) + path,
        data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def health(base_url: str, timeout: float = 30.0) -> None:
    with urllib.request.urlopen(server_root(base_url) + "/health", timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"health endpoint returned HTTP {response.status}")


def metrics(base_url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(
        server_root(base_url) + "/metrics", timeout=timeout
    ) as response:
        body = response.read().decode("utf-8", errors="replace")
    wanted = {
        "vllm:spec_decode_num_drafts_total": "target_verification_passes",
        "vllm:spec_decode_num_draft_tokens_total": "draft_tokens",
        "vllm:spec_decode_num_accepted_tokens_total": "accepted_tokens",
    }
    values = {destination: 0.0 for destination in wanted.values()}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split("{", 1)[0].split(" ", 1)[0]
        destination = wanted.get(metric_name)
        if destination is not None:
            values[destination] += float(line.rsplit(" ", 1)[-1])
    return values


def render_prompt_tokens(
    base_url: str, model: str, prompt: str, timeout: float
) -> list[int]:
    rendered = post_json(
        base_url,
        "/v1/chat/completions/render",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"thinking": False},
        },
        timeout,
    )
    token_ids = rendered.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise RuntimeError("render endpoint did not return token_ids")
    return [int(token_id) for token_id in token_ids]


def tokenize(base_url: str, model: str, prompt: str, timeout: float) -> list[int]:
    result = post_json(
        base_url, "/tokenize", {"model": model, "prompt": prompt}, timeout
    )
    return [int(token_id) for token_id in result["tokens"]]


def detokenize(base_url: str, model: str, tokens: list[int], timeout: float) -> str:
    result = post_json(
        base_url, "/detokenize", {"model": model, "tokens": tokens}, timeout
    )
    return str(result["prompt"])


def stream_completion(
    base_url: str,
    model: str,
    prompt_tokens: list[int],
    concurrency: int,
    output_tokens: int,
    timeout: float,
    *,
    seed: int,
    temperature: float,
    force_length: bool = True,
) -> dict:
    """Measure pure decode per sequence and collect matching DSpark counters."""

    parsed = urlparse(server_root(base_url))
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = connection_type(parsed.hostname, parsed.port, timeout=timeout)
    payload = {
        "model": model,
        "prompt": prompt_tokens,
        "add_special_tokens": False,
        "n": concurrency,
        "max_tokens": output_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "cache_prompt": False,
    }
    if force_length:
        payload.update({"min_tokens": output_tokens, "ignore_eos": True})

    before = metrics(base_url, timeout)
    started_epoch = time.time()
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

    token_times: list[list[float]] = [[] for _ in range(concurrency)]
    finish_reasons: list[str | None] = [None for _ in range(concurrency)]
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
            index = choice.get("index")
            if not isinstance(index, int) or not 0 <= index < concurrency:
                continue
            if choice.get("finish_reason") is not None:
                finish_reasons[index] = choice["finish_reason"]
            token_ids = choice.get("token_ids")
            if isinstance(token_ids, list):
                token_times[index].extend([observed] * len(token_ids))
    connection.close()
    ended = time.perf_counter()
    after = metrics(base_url, timeout)

    if any(len(times) != output_tokens for times in token_times):
        observed = [len(times) for times in token_times]
        raise RuntimeError(
            f"stream exposed token counts {observed}, expected {output_tokens} each; "
            f"usage={usage}"
        )
    first_token = min(times[0] for times in token_times)
    last_token = max(times[-1] for times in token_times)
    sequence_seconds = [times[-1] - times[0] for times in token_times]
    if any(seconds <= 0 for seconds in sequence_seconds):
        raise RuntimeError("stream timing resolution did not expose a positive decode window")
    sequence_rates = [
        (len(times) - 1) / seconds
        for times, seconds in zip(token_times, sequence_seconds, strict=True)
    ]
    decode_tokens = sum(len(times) - 1 for times in token_times)
    batch_window_seconds = last_token - first_token
    drafted = int(after["draft_tokens"] - before["draft_tokens"])
    accepted = int(after["accepted_tokens"] - before["accepted_tokens"])
    target_passes = int(
        after["target_verification_passes"]
        - before["target_verification_passes"]
    )
    rejected = drafted - accepted
    return {
        "concurrency": concurrency,
        "prompt_tokens": len(prompt_tokens),
        "completion_tokens": sum(len(times) for times in token_times),
        "completion_tokens_by_sequence": [len(times) for times in token_times],
        "decode_tokens": decode_tokens,
        "pure_decode_tokens_per_second": sum(sequence_rates),
        "per_sequence_decode_tokens_per_second": sequence_rates,
        "sequence_decode_seconds": sequence_seconds,
        "batch_window_decode_seconds": batch_window_seconds,
        "batch_window_decode_tokens_per_second": (
            decode_tokens / batch_window_seconds if batch_window_seconds else 0.0
        ),
        "ttft_ms": (first_token - started) * 1000,
        "sequence_ttft_ms": [(times[0] - started) * 1000 for times in token_times],
        "request_seconds": ended - started,
        "started_epoch_seconds": started_epoch,
        "draft_tokens": drafted,
        "accepted_draft_tokens": accepted,
        "rejected_draft_tokens": rejected,
        "accepted_draft_rate": accepted / drafted if drafted else None,
        "target_verification_passes": target_passes,
        "committed_tokens_per_target_pass": (
            1.0 + accepted / target_passes if target_passes else None
        ),
        "finish_reasons": finish_reasons,
        "usage": usage,
    }


def summarize_decode_runs(runs: list[dict]) -> dict:
    rates = [float(run["pure_decode_tokens_per_second"]) for run in runs]
    acceptances = [
        float(run["accepted_draft_rate"])
        for run in runs
        if run["accepted_draft_rate"] is not None
    ]
    efficiencies = [
        float(run["committed_tokens_per_target_pass"])
        for run in runs
        if run["committed_tokens_per_target_pass"] is not None
    ]
    return {
        "median_pure_decode_tokens_per_second": statistics.median(rates),
        "min_pure_decode_tokens_per_second": min(rates),
        "max_pure_decode_tokens_per_second": max(rates),
        "median_accepted_draft_rate": (
            statistics.median(acceptances) if acceptances else None
        ),
        "median_committed_tokens_per_target_pass": (
            statistics.median(efficiencies) if efficiencies else None
        ),
        "draft_tokens": sum(int(run["draft_tokens"]) for run in runs),
        "accepted_draft_tokens": sum(
            int(run["accepted_draft_tokens"]) for run in runs
        ),
        "rejected_draft_tokens": sum(
            int(run["rejected_draft_tokens"]) for run in runs
        ),
        "target_verification_passes": sum(
            int(run["target_verification_passes"]) for run in runs
        ),
    }
