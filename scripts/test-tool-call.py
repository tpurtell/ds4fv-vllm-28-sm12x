#!/usr/bin/env python3
"""Verify DeepSeek V4's production tool parser with a deterministic call."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from ds4fv_benchmark_common import health, server_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--role", required=True, choices=("native-vision", "exl3", "exl3-vision")
    )
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--recipe-commit", required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    health(args.base_url)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": "What's the weather like in Berlin right now?"}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for one city.",
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
        "max_tokens": 256,
        "chat_template_kwargs": {"thinking": False},
    }
    request = urllib.request.Request(
        server_root(args.base_url) + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.load(response)
    message = body["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    issues = []
    arguments = None
    if len(tool_calls) != 1:
        issues.append(f"observed {len(tool_calls)} tool calls, expected one")
    elif tool_calls[0].get("function", {}).get("name") != "get_weather":
        issues.append("model did not select get_weather")
    else:
        raw_arguments = tool_calls[0]["function"].get("arguments")
        try:
            arguments = json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            issues.append("tool arguments are not valid JSON")
        else:
            if arguments != {"location": "Berlin"}:
                issues.append(f"unexpected tool arguments: {arguments!r}")
    report = {
        "schema": "ds4fv-tool-call.v1",
        "passed": not issues,
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
            "model": args.model,
        },
        "expected": {"name": "get_weather", "arguments": {"location": "Berlin"}},
        "observed_tool_calls": tool_calls,
        "parsed_arguments": arguments,
        "issues": issues,
        "usage": body.get("usage"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
