#!/usr/bin/env python3
"""Docker health gate for the release-ready DS4FV service."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path


role = os.getenv("DS4FV_ROLE", "head")
if role in {"worker", "ray-worker", "ray-head"}:
    raise SystemExit(0)

if not Path("/tmp/ds4fv-release-ready").is_file():
    raise SystemExit("release-ready marker is absent")

port = os.getenv("API_PORT", "8000")
with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
    response.read()
