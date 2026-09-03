#!/usr/bin/env python3
"""Backport small post-v0.28 DeepSeek agent/tool robustness fixes.

Backports:

* vllm-project/vllm#54838: implicitly close a DSML parameter when the next
  parameter starts.
* vllm-project/vllm#48922: make malformed historical tool arguments
  recoverable instead of permanently failing the conversation.
* vllm-project/vllm#51262: render a valid assistant transition after a
  trailing system message.
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def patch_dsml(root: Path) -> None:
    path = root / "parser/deepseek_v4.py"
    replace_once(
        path,
        '''DSML_INVOKE_NAME_END = '\">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
''',
        '''DSML_INVOKE_NAME_END = '\">'
DSML_INVOKE_END = f"</{_DSML}invoke>"
DSML_PARAM_START = f"<{_DSML}parameter"
DSML_PARAM_CLOSE = f"</{_DSML}parameter>"
''',
        "#54838 DSML parameter start token",
    )
    replace_once(
        path,
        '''_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML}parameter>",
    re.DOTALL,
)
''',
        '''_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    rf"(.*?)"
    rf"(?:</{_ESCAPED_DSML}parameter>|(?=<{_ESCAPED_DSML}parameter\\s+name=))",
    re.DOTALL,
)
''',
        "#54838 implicit DSML parameter close",
    )
    replace_once(
        path,
        '''            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
''',
        '''            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
''',
        "#54838 V4 streaming terminal",
    )

    v32 = root / "parser/deepseek_v32.py"
    replace_once(
        v32,
        '''    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
''',
        '''    DSML_INVOKE_NAME_END,
    DSML_INVOKE_PREFIX,
    DSML_PARAM_CLOSE,
    DSML_PARAM_START,
''',
        "#54838 V3.2 shared terminal import",
    )
    replace_once(
        v32,
        '''            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
''',
        '''            "INVOKE_NAME_END": DSML_INVOKE_NAME_END,
            "INVOKE_END": DSML_INVOKE_END,
            "PARAM_START": DSML_PARAM_START,
            "PARAM_CLOSE": DSML_PARAM_CLOSE,
''',
        "#54838 V3.2 streaming terminal",
    )


def patch_historical_tool_arguments(root: Path) -> None:
    path = root / "entrypoints/chat_utils.py"
    replace_once(
        path,
        '''                # if arguments is None or empty string, set to {}
                if content := function.get("arguments"):
                    if not isinstance(content, (dict, list)):
                        parsed = json.loads(content)
                        function["arguments"] = parsed if parsed is not None else {}
                else:
                    function["arguments"] = {}
''',
        '''                # if arguments is None or empty string, set to {}
                if content := function.get("arguments"):
                    if isinstance(content, dict):
                        parsed = content
                    else:
                        if isinstance(content, str):
                            try:
                                parsed = json.loads(content)
                            except json.JSONDecodeError:
                                # Malformed arguments in conversation history
                                # would otherwise make every later turn fail.
                                logger.warning(
                                    "Tool call %r has arguments that are not valid "
                                    "JSON (%d chars); coercing to an empty object "
                                    "so the conversation can continue.",
                                    function.get("name"),
                                    len(content),
                                )
                                parsed = None
                        else:
                            parsed = content

                        if not isinstance(parsed, dict):
                            if parsed is not None:
                                logger.warning(
                                    "Tool call %r arguments decoded to %s, not a "
                                    "JSON object; coercing to an empty object.",
                                    function.get("name"),
                                    type(parsed).__name__,
                                )
                            parsed = {}

                    function["arguments"] = parsed
                else:
                    function["arguments"] = {}
''',
        "#48922 historical tool argument recovery",
    )


def patch_trailing_system(root: Path) -> None:
    path = root / "tokenizers/deepseek_v4_encoding.py"
    replace_once(
        path,
        '''    elif messages[index].get("role") in ["user", "developer"]:
        # Normal generation: append Assistant + thinking token
''',
        '''    # A trailing system message opens generation, while a system message
    # followed by assistant opens that assistant history turn.
    elif role in ["user", "developer"] or (
        role == "system"
        and (
            index == len(messages) - 1
            or messages[index + 1].get("role") == "assistant"
        )
    ):
        # Normal generation: append Assistant + thinking token
''',
        "#51262 trailing system transition",
    )


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_PACKAGE_ROOT")
    root = Path(sys.argv[1])
    patch_dsml(root)
    patch_historical_tool_arguments(root)
    patch_trailing_system(root)
    print("Applied vLLM post-0.28 DeepSeek agent/tool backports")


if __name__ == "__main__":
    main()
