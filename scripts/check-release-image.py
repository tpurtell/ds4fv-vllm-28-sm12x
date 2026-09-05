#!/usr/bin/env python3
"""Verify a published release image without downloading its layers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", help="tagged or digest-pinned OCI image reference")
    parser.add_argument("--expected-digest")
    parser.add_argument("--expected-revision")
    parser.add_argument("--expected-architecture", default="arm64")
    parser.add_argument("--expected-os", default="linux")
    return parser.parse_args()


def fail(message: str) -> None:
    raise SystemExit(f"release image check failed: {message}")


def main() -> None:
    args = parse_args()
    command = [
        "docker",
        "buildx",
        "imagetools",
        "inspect",
        "--format",
        "{{json .}}",
        args.reference,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        fail(f"{args.reference!r} is not resolvable: {detail}")

    try:
        inspection = json.loads(result.stdout)
        manifest_digest = inspection["manifest"]["digest"]
        image = inspection["image"]
        architecture = image["architecture"]
        operating_system = image["os"]
        labels = image.get("config", {}).get("Labels", {}) or {}
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        fail(f"unexpected Buildx inspection output: {error}")

    revision = labels.get("org.opencontainers.image.revision")
    checks = (
        (args.expected_digest, manifest_digest, "manifest digest"),
        (args.expected_revision, revision, "recipe revision"),
        (args.expected_architecture, architecture, "architecture"),
        (args.expected_os, operating_system, "operating system"),
    )
    for expected, actual, label in checks:
        if expected is not None and actual != expected:
            fail(f"{label} is {actual!r}, expected {expected!r}")

    print(
        json.dumps(
            {
                "reference": args.reference,
                "manifest_digest": manifest_digest,
                "architecture": architecture,
                "os": operating_system,
                "recipe_revision": revision,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
