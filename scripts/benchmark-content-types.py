#!/usr/bin/env python3
"""Benchmark the seven DS4 content categories plus the Orchid speed arm."""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptArm:
    category: str
    prompt: str
    max_tokens: int
    score_weight: float = 1.0
    constrained: bool = False


STRUCTURED_PROMPT = (
    "Return only a JSON object describing a file edit with keys path, operation, "
    "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
    "lines 41 through 47, and a one-sentence rationale about removing a redundant copy."
)
STRUCTURED_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "operation": {"type": "string"},
        "line_start": {"type": "integer"},
        "line_end": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["path", "operation", "line_start", "line_end", "rationale"],
    "additionalProperties": False,
}

# Structured JSON has normal and response_format arms at half weight each, so
# together they contribute the same score weight as each other semantic category.
ARMS = {
    "code": PromptArm(
        "code",
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and exactly three "
        "concise assert-based examples. Return only one Python code block and keep the "
        "complete response under 450 tokens.",
        512,
    ),
    "math": PromptArm(
        "reasoning",
        "A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the "
        "discounted price. What is the final price? Show the calculation briefly.",
        128,
    ),
    "fable": PromptArm(
        "creative-prose",
        "Write a self-contained fable of 150 to 165 words about two parrots who disagree "
        "about sharing credit. End with one sentence that starts with 'Moral:' and "
        "explicitly includes both words 'share' and 'credit'.",
        256,
    ),
    "hello": PromptArm("short-response", "hi", 32),
    "topic": PromptArm(
        "exposition",
        "Explain virtual memory to a junior programmer in exactly five bullet points. "
        "Use one sentence of at most 25 words per bullet, no heading or closing text, "
        "and include paging, page faults, and the role of the TLB.",
        256,
    ),
    "structured-json-normal": PromptArm(
        "structured-output", STRUCTURED_PROMPT, 128, score_weight=0.5
    ),
    "structured-json-constrained": PromptArm(
        "structured-output",
        STRUCTURED_PROMPT,
        128,
        score_weight=0.5,
        constrained=True,
    ),
    "multilingual": PromptArm(
        "multilingual",
        "請用繁體中文，以正好四個單行條列解釋什麼是寫入時複製（copy-on-write）。"
        "每個條列限一個句子且最多四十個中文字，不要標題、前言、子條列或結語；"
        "其中一個條列必須包含行程 fork 後修改記憶體頁面的例子。",
        384,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--role", choices=("native-vision", "exl3", "exl3-vision")
    )
    parser.add_argument("--image-id")
    parser.add_argument("--recipe-commit")
    parser.add_argument("--dspark-tokens", type=int)
    parser.add_argument(
        "--dspark-policy",
        choices=("fixed", "stock-adaptive", "target-only"),
    )
    parser.add_argument(
        "--draft-sample-method", choices=("greedy", "probabilistic")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--orchid-warmups", type=int, default=1)
    parser.add_argument(
        "--orchid-count",
        type=int,
        default=100,
        help="Minimum repeated words required for a valid Orchid speed sample.",
    )
    parser.add_argument("--orchid-max-tokens", type=int, default=1500)
    parser.add_argument("--skip-orchid", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--require-contracts",
        action="store_true",
        help="Exit nonzero unless every timed semantic sample passes its contract.",
    )
    parser.add_argument(
        "--minimum-contract-passes",
        type=int,
        default=None,
        help=(
            "Exit nonzero unless at least this many timed semantic samples pass, "
            "all structured JSON samples pass, and the Orchid contract passes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.orchid_count < 1 or args.orchid_max_tokens < 1:
        parser.error("repeats, orchid count, and orchid max tokens must be positive")
    if args.orchid_warmups < 0 or args.timeout <= 0:
        parser.error("orchid warmups must be non-negative and timeout must be positive")
    if args.dspark_tokens is not None and args.dspark_tokens < 0:
        parser.error("dspark tokens must be non-negative")
    maximum_contracts = args.repeats * len(ARMS)
    if args.minimum_contract_passes is not None and not (
        0 <= args.minimum_contract_passes <= maximum_contracts
    ):
        parser.error(
            f"minimum contract passes must stay within 0..{maximum_contracts}"
        )
    return args


def stream_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"thinking": False},
    }
    if response_format is not None:
        body["response_format"] = response_format
    request = urllib.request.Request(
        f"{base_url.rstrip('/').removesuffix('/v1')}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    content_parts: list[str] = []
    finish_reason = None
    usage = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            text = (delta.get("content") or "") + (
                delta.get("reasoning") or delta.get("reasoning_content") or ""
            )
            if first is None and text:
                first = time.perf_counter()
            content_parts.append(delta.get("content") or "")
            if choices and choices[0].get("finish_reason"):
                finish_reason = choices[0]["finish_reason"]
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    if usage is None:
        raise RuntimeError("stream ended without a usage record")
    first = first or finished
    completion_tokens = int(usage["completion_tokens"])
    decode_seconds = max(0.001, finished - first)
    return {
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ttft_s": first - started,
        "decode_s": decode_seconds,
        "decode_tok_s": max(0, completion_tokens - 1) / decode_seconds,
        "content": "".join(content_parts),
    }


def structured_passed(content: str) -> bool:
    json_text = content.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n(?P<json>.*)\n```",
        json_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is not None:
        json_text = fenced.group("json")
    try:
        value = json.loads(json_text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(value, dict)
        and set(value) == set(STRUCTURED_SCHEMA["required"])
        and value["path"] == "src/cache.rs"
        and value["operation"] == "replace"
        and value["line_start"] == 41
        and value["line_end"] == 47
        and isinstance(value["rationale"], str)
        and bool(value["rationale"].strip())
    )


def validate_semantic_contract(arm_id: str, content: str) -> tuple[bool, list[str]]:
    """Validate every prompt-visible release contract without executing output."""

    issues: list[str] = []
    stripped = content.strip()
    if not stripped:
        return False, ["response is empty"]
    if arm_id == "code":
        match = re.fullmatch(
            r"\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*",
            content,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            issues.append("response is not exactly one Python code block")
        else:
            try:
                tree = ast.parse(match.group("code"))
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "merge_intervals"
                ]
                if len(functions) != 1:
                    issues.append("merge_intervals function is missing or duplicated")
                else:
                    function = functions[0]
                    if (
                        not function.args.args
                        or function.args.args[0].annotation is None
                        or function.returns is None
                    ):
                        issues.append("merge_intervals lacks requested type hints")
                    if ast.get_docstring(function) is None:
                        issues.append("merge_intervals lacks a docstring")
                if sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) < 3:
                    issues.append("fewer than three assert examples were provided")
    elif arm_id == "math":
        normalized = stripped.replace(",", "")
        if re.search(r"(?<![0-9])(?:\$\s*)?194\.4(?:0)?(?![0-9])", normalized) is None:
            issues.append("response does not contain the correct final price 194.40")
        if not all(value in normalized for value in ("240", "25", "8")):
            issues.append("response does not show the requested calculation inputs")
    elif arm_id == "fable":
        words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
        if not 140 <= len(words) <= 180:
            issues.append(f"fable has {len(words)} words, outside 140..180")
        final_sentence = re.split(r"(?<=[.!?])\s+", stripped)[-1].casefold()
        if not any(
            term in final_sentence
            for term in ("credit", "share", "together", "team", "fair", "both")
        ):
            issues.append("response does not end with a moral about sharing credit")
    elif arm_id == "hello":
        if len(stripped) > 512:
            issues.append("short greeting response is unexpectedly long")
    elif arm_id == "topic":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-5][.)])\s+", line)
        ]
        if len(bullets) != 5:
            issues.append(f"response has {len(bullets)} bullets, expected five")
        lowered = stripped.casefold()
        for term in ("paging", "page fault", "tlb"):
            if term not in lowered:
                issues.append(f"response omits {term}")
    elif arm_id.startswith("structured-json"):
        if not structured_passed(content):
            issues.append("response does not preserve the exact file-edit JSON contract")
    elif arm_id == "multilingual":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-4][.)、])\s*", line)
        ]
        if len(bullets) != 4:
            issues.append(f"response has {len(bullets)} bullets, expected four")
        lowered = stripped.casefold()
        if not ("寫入時複製" in stripped or "copy-on-write" in lowered):
            issues.append("response omits copy-on-write")
        if "fork" not in lowered or "頁" not in stripped:
            issues.append("response omits the requested fork/page example")
    else:
        raise ValueError(f"no semantic contract for arm {arm_id!r}")
    return not issues, issues


def compact_record(
    *, arm_id: str, arm: PromptArm, repeat: int, timed: bool, raw: dict[str, Any]
) -> dict[str, Any]:
    content = raw.pop("content")
    record = {
        "arm": arm_id,
        "category": arm.category,
        "score_weight": arm.score_weight,
        "constrained": arm.constrained,
        "repeat": repeat,
        "timed": timed,
        **raw,
        "content_chars": len(content),
        "content_preview": content[:240].replace("\n", "\\n"),
    }
    if arm.category == "structured-output":
        record["structured_contract_passed"] = structured_passed(content)
    if arm_id != "orchid":
        passed, issues = validate_semantic_contract(arm_id, content)
        if raw["finish_reason"] == "length":
            issues.append("response hit the output token limit")
            passed = False
        record["quality_contract_passed"] = passed
        record["quality_contract_issues"] = issues
    if arm_id == "orchid":
        words = re.findall(r"\b[A-Za-z]+\b", content)
        occurrences = sum(word.lower() == "orchid" for word in words)
        record["observed_orchid_count"] = occurrences
        record["orchid_only"] = re.fullmatch(
            r"\s*orchid(?:\s+orchid)*\s*", content, flags=re.IGNORECASE
        ) is not None
        record["orchid_minimum_reached"] = occurrences >= int(raw["minimum_count"])
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["timed"]:
            by_arm.setdefault(record["arm"], []).append(record)
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm_id, samples in by_arm.items():
        rates = [float(sample["decode_tok_s"]) for sample in samples]
        summary: dict[str, Any] = {
            "category": samples[0]["category"],
            "score_weight": float(samples[0]["score_weight"]),
            "samples": len(samples),
            "median_decode_tok_s": statistics.median(rates),
            "min_decode_tok_s": min(rates),
            "max_decode_tok_s": max(rates),
            "completion_tokens": [int(sample["completion_tokens"]) for sample in samples],
        }
        if arm_id != "orchid":
            summary["contract_passes"] = sum(
                bool(sample["quality_contract_passed"]) for sample in samples
            )
        if arm_id == "orchid":
            summary["valid_repetitions"] = sum(
                bool(sample["orchid_only"] and sample["orchid_minimum_reached"])
                for sample in samples
            )
        arm_summaries[arm_id] = summary

    scored = {key: value for key, value in arm_summaries.items() if key != "orchid"}
    total_weight = sum(float(value["score_weight"]) for value in scored.values())
    weighted_score = sum(
        float(value["score_weight"]) * float(value["median_decode_tok_s"])
        for value in scored.values()
    ) / total_weight
    contract_records = [
        record
        for record in records
        if record["timed"] and record["arm"] != "orchid"
    ]
    orchid_summary = arm_summaries.get("orchid")
    orchid_contract_passed = (
        orchid_summary is None
        or int(orchid_summary["valid_repetitions"])
        == int(orchid_summary["samples"])
    )
    structured_records = [
        record
        for record in contract_records
        if record["arm"]
        in ("structured-json-normal", "structured-json-constrained")
    ]
    return {
        "arms": arm_summaries,
        "weighted_content_score_tok_s": weighted_score,
        "weighted_content_score_total_weight": total_weight,
        "quality_contract_passes": sum(
            bool(record["quality_contract_passed"]) for record in contract_records
        ),
        "quality_contract_total": len(contract_records),
        "quality_contract_passed": all(
            bool(record["quality_contract_passed"]) for record in contract_records
        ),
        "structured_contract_passes": sum(
            bool(record.get("structured_contract_passed"))
            for record in structured_records
        ),
        "structured_contract_total": len(structured_records),
        "structured_contract_passed": bool(structured_records)
        and all(
            bool(record.get("structured_contract_passed"))
            for record in structured_records
        ),
        "orchid_contract_passed": orchid_contract_passed,
    }


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"run-{time.time_ns()}"
    report: dict[str, Any] = {
        "schema": "ds4fv-content-types-v3",
        "run_id": run_id,
        "base_url": args.base_url,
        "model": args.model,
        "provenance": {
            "role": args.role,
            "image_id": args.image_id,
            "recipe_commit": args.recipe_commit,
        },
        "profile": {
            "dspark_tokens": args.dspark_tokens,
            "dspark_policy": args.dspark_policy,
            "draft_sample_method": args.draft_sample_method,
        },
        "repeats": args.repeats,
        "thinking": "off",
        "temperature": 0,
        "arms": {arm_id: asdict(arm) for arm_id, arm in ARMS.items()},
        "records": [],
    }

    def persist() -> None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )

    for repeat in range(1, args.repeats + 1):
        for arm_id, arm in ARMS.items():
            response_format = None
            if arm.constrained:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"name": "file_edit", "schema": STRUCTURED_SCHEMA},
                }
            raw = stream_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=arm.prompt,
                max_tokens=arm.max_tokens,
                timeout=args.timeout,
                response_format=response_format,
            )
            record = compact_record(
                arm_id=arm_id, arm=arm, repeat=repeat, timed=True, raw=raw
            )
            report["records"].append(record)
            persist()
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    if not args.skip_orchid:
        orchid = PromptArm("low-entropy-showcase", "", args.orchid_max_tokens)
        for sample in range(args.orchid_warmups + args.repeats):
            prompt = (
                'Output only the single word "orchid" repeatedly, separated by '
                "single spaces. Continue until the output token limit and do not "
                "add punctuation or any other text."
            )
            raw = stream_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                max_tokens=args.orchid_max_tokens,
                timeout=args.timeout,
            )
            raw["minimum_count"] = args.orchid_count
            record = compact_record(
                arm_id="orchid",
                arm=orchid,
                repeat=sample + 1,
                timed=sample >= args.orchid_warmups,
                raw=raw,
            )
            report["records"].append(record)
            persist()
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    report["summary"] = summarize(report["records"])
    if args.minimum_contract_passes is not None:
        report["summary"]["minimum_contract_passes_required"] = (
            args.minimum_contract_passes
        )
        report["summary"]["release_contract_floor_passed"] = bool(
            report["summary"]["quality_contract_passes"]
            >= args.minimum_contract_passes
            and report["summary"]["structured_contract_passed"]
            and report["summary"]["orchid_contract_passed"]
        )
    persist()
    print(json.dumps({"summary": report["summary"]}, sort_keys=True))
    if args.require_contracts and not (
        report["summary"]["quality_contract_passed"]
        and report["summary"]["orchid_contract_passed"]
    ):
        raise SystemExit(1)
    if args.minimum_contract_passes is not None and not report["summary"][
        "release_contract_floor_passed"
    ]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
