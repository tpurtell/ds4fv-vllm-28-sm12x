#!/usr/bin/env python3
"""Exercise the release warmup against a CPU-only recording HTTP stub."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUESTS: list[tuple[str, dict]] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args) -> None:
        pass

    def reply(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self.reply({})
        elif self.path == "/v1/models":
            self.reply({"data": [{"id": "mock-deepseek-v4"}]})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        REQUESTS.append((self.path, payload))
        if self.path == "/tokenize":
            self.reply({"tokens": [11, 12, 13]})
        elif self.path == "/v1/chat/completions/render":
            self.reply({"token_ids": [21, 22, 23, 24]})
        elif self.path in {"/v1/completions", "/v1/chat/completions"}:
            count = int(payload.get("n", 1))
            choices = [{"text": "ok"} for _ in range(count)]
            if payload.get("tools"):
                choices[0] = {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location":"Berlin"}',
                                },
                            }
                        ]
                    }
                }
            self.reply({"choices": choices})
        else:
            self.send_error(404)


def main() -> None:
    path = ROOT / "scripts/release-warmup.py"
    spec = importlib.util.spec_from_file_location("ds4fv_release_warmup", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    warmup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(warmup)
    warmup.time.sleep = lambda _seconds: None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    old_argv = sys.argv
    old_dspark_tokens = os.environ.get("DSPARK_TOKENS")
    try:
        os.environ["DSPARK_TOKENS"] = "6"
        sys.argv = [
            str(path),
            "--server-pid",
            str(os.getpid()),
            "--base-url",
            f"http://127.0.0.1:{server.server_port}",
            "--role",
            "native-vision",
            "--passes",
            "1",
        ]
        warmup.main()
    finally:
        sys.argv = old_argv
        if old_dspark_tokens is None:
            os.environ.pop("DSPARK_TOKENS", None)
        else:
            os.environ["DSPARK_TOKENS"] = old_dspark_tokens
        server.shutdown()
        thread.join()
        server.server_close()

    completion_payloads = [payload for path, payload in REQUESTS if path == "/v1/completions"]
    prompt_lengths = {
        len(payload["prompt"])
        for payload in completion_payloads
        if isinstance(payload.get("prompt"), list)
    }
    assert {1, 9, 25, 57, 121, 249, 8192, 9500} <= prompt_lengths
    assert any(payload.get("n") == 2 for payload in completion_payloads)
    assert any(payload.get("n") == 4 for payload in completion_payloads)

    chat_payloads = [
        payload for path, payload in REQUESTS if path == "/v1/chat/completions"
    ]
    assert any("response_format" in payload for payload in chat_payloads)
    assert any("tools" in payload for payload in chat_payloads)
    image_counts = []
    for payload in chat_payloads:
        content = payload["messages"][0]["content"]
        if isinstance(content, list):
            image_counts.append(
                sum(item.get("type") == "image_url" for item in content)
            )
    assert image_counts == [1, 4, 16]
    print("release warmup HTTP smoke passed")


if __name__ == "__main__":
    main()
