#!/usr/bin/env python3
"""Build a compact per-arm comparison from two content-type reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("schema") != "ds4fv-content-types-v3" or "summary" not in report:
        raise ValueError(f"{path} is not a complete ds4fv content-types v3 report")
    return report


def delta(candidate: float, baseline: float) -> dict[str, float]:
    return {
        "absolute": candidate - baseline,
        "percent": (candidate / baseline - 1.0) * 100.0,
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    for field in ("model", "provenance"):
        if baseline.get(field) != candidate.get(field):
            raise ValueError(f"reports do not have matched {field}")

    baseline_arms = baseline["summary"]["arms"]
    candidate_arms = candidate["summary"]["arms"]
    if baseline_arms.keys() != candidate_arms.keys():
        raise ValueError("reports do not contain the same content arms")

    arms = {}
    for arm in baseline_arms:
        baseline_rate = float(baseline_arms[arm]["median_decode_tok_s"])
        candidate_rate = float(candidate_arms[arm]["median_decode_tok_s"])
        arms[arm] = {
            "category": baseline_arms[arm]["category"],
            "baseline_median_decode_tok_s": baseline_rate,
            "candidate_median_decode_tok_s": candidate_rate,
            "decode_delta": delta(candidate_rate, baseline_rate),
            "baseline_contract_passes": baseline_arms[arm].get("contract_passes"),
            "candidate_contract_passes": candidate_arms[arm].get("contract_passes"),
            "samples_per_profile": baseline_arms[arm]["samples"],
        }

    baseline_score = float(baseline["summary"]["weighted_content_score_tok_s"])
    candidate_score = float(candidate["summary"]["weighted_content_score_tok_s"])
    output = {
        "schema": "ds4fv-content-types-comparison.v1",
        "model": baseline["model"],
        "provenance": baseline["provenance"],
        "baseline": {
            "label": args.baseline_label,
            "artifact": str(args.baseline),
            "profile": baseline.get("profile"),
            "weighted_content_score_tok_s": baseline_score,
            "quality_contract_passes": baseline["summary"][
                "quality_contract_passes"
            ],
            "structured_contract_passes": baseline["summary"][
                "structured_contract_passes"
            ],
        },
        "candidate": {
            "label": args.candidate_label,
            "artifact": str(args.candidate),
            "profile": candidate.get("profile"),
            "weighted_content_score_tok_s": candidate_score,
            "quality_contract_passes": candidate["summary"][
                "quality_contract_passes"
            ],
            "structured_contract_passes": candidate["summary"][
                "structured_contract_passes"
            ],
        },
        "weighted_content_score_delta": delta(candidate_score, baseline_score),
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
