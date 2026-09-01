#!/usr/bin/env python3
"""Post-long-context C4 reliability soak for the frozen DS4FV image."""

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
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--role", required=True, choices=("native-vision", "exl3"))
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--inter-run-seconds", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 4:
        parser.error("concurrency must stay within the qualified 1..4 envelope")
    if args.runs < 1 or args.output_tokens < 2:
        parser.error("runs must be positive and output tokens must be at least two")

    health(args.base_url)
    prompt_tokens = render_prompt_tokens(
        args.base_url, args.model, CODE_AGENT_PROMPT, args.timeout
    )
    report = {
        "schema": "ds4fv-post-long-context-soak.v1",
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "concurrency": args.concurrency,
        "requested_runs": args.runs,
        "output_tokens_per_sequence": args.output_tokens,
        "runs": [],
    }

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    for run_index in range(args.runs):
        try:
            result = stream_completion(
                args.base_url,
                args.model,
                prompt_tokens,
                args.concurrency,
                args.output_tokens,
                args.timeout,
                seed=args.seed,
                temperature=args.temperature,
            )
            health(args.base_url)
        except Exception as error:  # retain the first failure in the release receipt
            report["failure"] = {
                "run": run_index + 1,
                "type": type(error).__name__,
                "message": str(error),
            }
            report["passed"] = False
            persist()
            raise
        report["runs"].append(result)
        persist()
        print(
            f"soak {run_index + 1}/{args.runs}: "
            f"{result['pure_decode_tokens_per_second']:.2f} tok/s",
            flush=True,
        )
        time.sleep(args.inter_run_seconds)
    report["summary"] = summarize_decode_runs(report["runs"])
    report["passed"] = len(report["runs"]) == args.runs
    persist()


if __name__ == "__main__":
    main()
