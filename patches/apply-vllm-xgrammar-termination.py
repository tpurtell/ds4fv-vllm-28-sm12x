#!/usr/bin/env python3
"""Backport vLLM's termination-safe xgrammar speculative token batches."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: {label} expected one anchor, found {count}")
    path.write_text(source.replace(old, new, 1))


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-xgrammar-termination.py VLLM_ROOT")

    root = Path(sys.argv[1]).resolve()
    path = root / "v1/structured_output/backend_xgrammar.py"
    if not path.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")

    replace_once(
        path,
        '''        Returns True if the FSM was advanced successfully.
        Returns False if the FSM failed to advance.
        """
        if self._is_terminated:
            return False
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
        self._is_terminated = self.matcher.is_terminated()
        return True
''',
        '''        Returns True if all grammar-constrained tokens were accepted.
        Tokens after termination are ignored. Returns False if the FSM
        failed to advance.
        """
        if self._is_terminated:
            return True
        for token in tokens:
            if not self.matcher.accept_token(token):
                logger.error(
                    "Failed to advance FSM for request %s "
                    "for tokens %s. Please file an issue.",
                    request_id,
                    token,
                )
                return False
            self.num_processed_tokens += 1
            self._is_terminated = self.matcher.is_terminated()
            if self._is_terminated:
                break
        return True
''',
        "termination-safe accept_tokens",
    )
    replace_once(
        path,
        '''        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
            else:
                break
''',
        '''        if self._is_terminated:
            return []

        accepted_tokens = []
        for token in tokens:
            if self.matcher.accept_token(token):
                accepted_tokens.append(token)
                if self.matcher.is_terminated():
                    break
            else:
                break
''',
        "termination-safe validate_tokens",
    )
    replace_once(
        path,
        '''    def reset(self):
        self.num_processed_tokens = 0
        self.matcher.reset()
''',
        '''    def reset(self):
        self.matcher.reset()
        self.num_processed_tokens = 0
        self._is_terminated = False
''',
        "termination-state reset",
    )


if __name__ == "__main__":
    main()
