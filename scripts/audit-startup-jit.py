#!/usr/bin/env python3
"""Record whether a DS4FV container compiled kernels after readiness."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


READY_TEXT = "DS4FV release startup warmup complete; container is ready."
MONITORED_JIT_PATTERN = re.compile(
    r"(?:Triton kernel|TileLang) JIT compilation during inference: ([^.]+)"
)
TILELANG_COMPILE_PATTERN = re.compile(
    r"TileLang begins to compile kernel `([^`]+)`"
)
EMBEDDED_TIMESTAMP_PATTERN = re.compile(
    r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
)
WARNING_TIMESTAMP_PATTERN = re.compile(
    r"WARNING (?P<month>\d{2})-(?P<day>\d{2}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
)


def command(*args: str) -> str:
    return subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout


def docker_timestamp(line: str) -> datetime:
    raw = line.split(maxsplit=1)[0]
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def event_timestamp(line: str, ready_at: datetime) -> datetime:
    """Use worker event time instead of delayed Ray log-delivery time."""

    embedded = list(EMBEDDED_TIMESTAMP_PATTERN.finditer(line))
    if embedded:
        values = {key: int(value) for key, value in embedded[-1].groupdict().items()}
        return datetime(**values, tzinfo=timezone.utc)
    warning = WARNING_TIMESTAMP_PATTERN.search(line)
    if warning:
        values = {key: int(value) for key, value in warning.groupdict().items()}
        return datetime(year=ready_at.year, **values, tzinfo=timezone.utc)
    return docker_timestamp(line)


def jit_event(line: str, ready_at: datetime) -> dict[str, str] | None:
    monitored = MONITORED_JIT_PATTERN.search(line)
    tilelang = TILELANG_COMPILE_PATTERN.search(line)
    match = monitored or tilelang
    if match is None:
        return None
    occurred_at = event_timestamp(line, ready_at)
    return {
        "kernel": match.group(1),
        "source": "jit-monitor" if monitored else "tilelang-compiler",
        "occurred_at": occurred_at.isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inspect = json.loads(command("docker", "inspect", args.container))[0]
    lines = command("docker", "logs", "--timestamps", args.container).splitlines()
    ready_indices = [index for index, line in enumerate(lines) if READY_TEXT in line]
    if len(ready_indices) != 1:
        raise SystemExit(
            f"expected exactly one ready marker, observed {len(ready_indices)}"
        )
    ready_index = ready_indices[0]
    ready_at = docker_timestamp(lines[ready_index])
    after = lines[ready_index + 1 :]
    events = [event for line in lines if (event := jit_event(line, ready_at))]
    pre_ready_jit = [
        event
        for event in events
        if datetime.fromisoformat(event["occurred_at"]) < ready_at
    ]
    post_ready_jit = [
        event
        for event in events
        if datetime.fromisoformat(event["occurred_at"]) >= ready_at
    ]
    report = {
        "schema": "ds4fv-release-startup-jit-audit.v1",
        "container": args.container,
        "image": inspect["Config"]["Image"],
        "image_id": inspect["Image"],
        "container_state": inspect["State"]["Status"],
        "health_status_at_audit": (
            (inspect["State"].get("Health") or {}).get("Status")
            if inspect["State"]["Status"] == "running"
            else None
        ),
        "ready_marker": lines[ready_index],
        "pre_ready_jit_count": len(pre_ready_jit),
        "pre_ready_jit_kernels": sorted(
            {event["kernel"] for event in pre_ready_jit}
        ),
        "post_ready_request_count": sum('\"POST ' in line for line in after),
        "post_ready_health_200_count": sum(
            '\"GET /health HTTP/1.1\" 200 OK' in line for line in after
        ),
        "post_ready_jit_count": len(post_ready_jit),
        "post_ready_jit_kernels": sorted(
            {event["kernel"] for event in post_ready_jit}
        ),
        "post_ready_jit_events": post_ready_jit,
        "passed": not post_ready_jit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
