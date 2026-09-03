#!/usr/bin/env python3
"""CPU-only smoke for termination-safe xgrammar speculative token batches."""

from __future__ import annotations

from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar


class FakeMatcher:
    def __init__(self, stop_token: int) -> None:
        self.stop_token = stop_token
        self.tokens: list[int] = []
        self.terminated = False

    def accept_token(self, token: int) -> bool:
        if self.terminated:
            return False
        self.tokens.append(token)
        self.terminated = token == self.stop_token
        return True

    def is_terminated(self) -> bool:
        return self.terminated

    def rollback(self, count: int) -> None:
        del self.tokens[-count:]
        self.terminated = bool(self.tokens and self.tokens[-1] == self.stop_token)

    def reset(self) -> None:
        self.tokens.clear()
        self.terminated = False


def grammar(matcher: FakeMatcher) -> XgrammarGrammar:
    return XgrammarGrammar(vocab_size=128, matcher=matcher, ctx=object())


def main() -> None:
    matcher = FakeMatcher(stop_token=99)
    state = grammar(matcher)

    assert state.accept_tokens("batch", [1, 99, 2])
    assert matcher.tokens == [1, 99]
    assert state.num_processed_tokens == 2
    assert state.is_terminated()

    assert state.accept_tokens("already-stopped", [3])
    assert state.validate_tokens([3]) == []
    assert matcher.tokens == [1, 99]

    state.reset()
    assert not state.is_terminated()
    assert state.num_processed_tokens == 0
    assert matcher.tokens == []

    assert state.validate_tokens([4, 99, 5]) == [4, 99]
    assert matcher.tokens == []
    assert not matcher.is_terminated()
    assert not state.is_terminated()
    print("xgrammar termination smoke passed")


if __name__ == "__main__":
    main()
