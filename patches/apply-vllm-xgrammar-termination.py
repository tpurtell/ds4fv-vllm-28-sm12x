#!/usr/bin/env python3
"""Backport vLLM's speculative xgrammar termination/reasoning fixes.

Backports:

* vllm-project/vllm#52805: stop a token batch when its grammar terminates.
* vllm-project/vllm#53046: validate speculative tokens immediately after a
  reasoning boundary before attempting to advance the grammar FSM.
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply-vllm-xgrammar-termination.py VLLM_ROOT")

    root = Path(sys.argv[1]).resolve()
    path = root / "v1/structured_output/backend_xgrammar.py"
    manager_path = root / "v1/structured_output/__init__.py"
    if not path.is_file():
        raise RuntimeError(f"not a vLLM package root: {root}")
    if not manager_path.is_file():
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
    replace_once(
        manager_path,
        '''                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
''',
        '''                    if advance_grammar and not grammar.is_terminated():
                        if post_reasoning_end_in_window:
                            accepted = bool(grammar.validate_tokens([token]))
                            if accepted:
                                accepted = grammar.accept_tokens(req_id, [token])
                        else:
                            accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
''',
        "post-reasoning speculative token validation",
    )


if __name__ == "__main__":
    main()
