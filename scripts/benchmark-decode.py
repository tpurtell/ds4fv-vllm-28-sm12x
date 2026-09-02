#!/usr/bin/env python3
"""Run the DS4FV code-agent pure-decode and context-depth release curves."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ds4fv_benchmark_common import (
    CODE_AGENT_PROMPT,
    health,
    render_prompt_tokens,
    stream_completion,
    summarize_decode_runs,
    tokenize,
)


FILLER = (
    "Slate rivers cross quiet valleys while copper clocks mark patient hours. "
    "This is ordinary context with no instructions.\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--role", required=True, choices=("native-vision", "exl3", "exl3-vision")
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--dspark-tokens", type=int, required=True)
    parser.add_argument(
        "--dspark-policy",
        required=True,
        choices=("fixed", "stock-adaptive", "target-only"),
    )
    parser.add_argument(
        "--draft-sample-method", choices=("greedy", "probabilistic"), default="greedy"
    )
    parser.add_argument("--concurrency", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument(
        "--context-depths", nargs="+", type=int, default=(0, 8192, 32768, 65536, 128000)
    )
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--depth-warmup-runs", type=int, default=1)
    parser.add_argument("--depth-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--inter-run-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.dspark_tokens < 0 or args.output_tokens < 2:
        parser.error("dspark tokens must be non-negative and output tokens must be at least two")
    if args.dspark_policy == "target-only" and args.dspark_tokens != 0:
        parser.error("target-only reports must use --dspark-tokens 0")
    if args.dspark_policy != "target-only" and args.dspark_tokens == 0:
        parser.error("DSpark reports require a positive --dspark-tokens value")
    if min(args.concurrency) < 1 or max(args.concurrency) > 4:
        parser.error("the qualified Spark scheduler envelope is concurrency 1..4")
    if min(args.context_depths) < 0 or max(args.context_depths) > 128000:
        parser.error("context depths must stay within 0..128000")
    if min(args.warmup_runs, args.runs, args.depth_warmup_runs, args.depth_runs) < 1:
        parser.error("all run counts must be positive")
    if args.inter_run_seconds < 0 or args.timeout <= 0:
        parser.error("inter-run seconds must be non-negative and timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    health(args.base_url)
    task_tokens = render_prompt_tokens(
        args.base_url, args.model, CODE_AGENT_PROMPT, args.timeout
    )
    report = {
        "schema": "ds4fv-code-agent-release.v1",
        "method": (
            "each sequence is timed from its own first through last streamed token; "
            "aggregate pure decode is the sum of per-sequence rates and excludes "
            "each sequence's TTFT; context-depth points prepend ordinary tokenized "
            "filler to the same rendered code-agent task"
        ),
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "profile": {
            "dspark_tokens": args.dspark_tokens,
            "dspark_policy": args.dspark_policy,
            "draft_sample_method": args.draft_sample_method,
            "temperature": args.temperature,
            "seed": args.seed,
            "output_tokens_per_sequence": args.output_tokens,
        },
        "workload": CODE_AGENT_PROMPT,
        "concurrency_curve": [],
        "context_depth_curve": [],
    }

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for concurrency in args.concurrency:
        for _ in range(args.warmup_runs):
            stream_completion(
                args.base_url,
                args.model,
                task_tokens,
                concurrency,
                args.output_tokens,
                args.timeout,
                seed=args.seed,
                temperature=args.temperature,
            )
            time.sleep(args.inter_run_seconds)
        runs = []
        for run_index in range(args.runs):
            result = stream_completion(
                args.base_url,
                args.model,
                task_tokens,
                concurrency,
                args.output_tokens,
                args.timeout,
                seed=args.seed,
                temperature=args.temperature,
            )
            runs.append(result)
            print(
                f"C{concurrency} run {run_index + 1}/{args.runs}: "
                f"{result['pure_decode_tokens_per_second']:.2f} tok/s, "
                f"acceptance={result['accepted_draft_rate']}",
                flush=True,
            )
            time.sleep(args.inter_run_seconds)
        report["concurrency_curve"].append(
            {"concurrency": concurrency, **summarize_decode_runs(runs), "runs": runs}
        )
        persist()

    filler_ids = tokenize(args.base_url, args.model, FILLER, args.timeout)
    for requested_depth in args.context_depths:
        actual_depth = len(task_tokens) if requested_depth == 0 else requested_depth
        if actual_depth < len(task_tokens):
            raise ValueError(
                f"depth {actual_depth} is smaller than the {len(task_tokens)}-token task"
            )
        needed = actual_depth - len(task_tokens)
        prompt_tokens = (
            (filler_ids * ((needed + len(filler_ids) - 1) // len(filler_ids)))[:needed]
            + task_tokens
        )
        for _ in range(args.depth_warmup_runs):
            stream_completion(
                args.base_url,
                args.model,
                prompt_tokens,
                1,
                args.output_tokens,
                args.timeout,
                seed=args.seed,
                temperature=args.temperature,
            )
            time.sleep(args.inter_run_seconds)
        runs = []
        for run_index in range(args.depth_runs):
            result = stream_completion(
                args.base_url,
                args.model,
                prompt_tokens,
                1,
                args.output_tokens,
                args.timeout,
                seed=args.seed,
                temperature=args.temperature,
            )
            runs.append(result)
            print(
                f"depth {requested_depth:,} run {run_index + 1}/{args.depth_runs}: "
                f"{result['pure_decode_tokens_per_second']:.2f} tok/s, "
                f"acceptance={result['accepted_draft_rate']}",
                flush=True,
            )
            time.sleep(args.inter_run_seconds)
        report["context_depth_curve"].append(
            {
                "existing_depth_tokens": requested_depth,
                "actual_prompt_tokens": actual_depth,
                **summarize_decode_runs(runs),
                "runs": runs,
            }
        )
        persist()

    report["passed"] = True
    persist()


if __name__ == "__main__":
    main()
