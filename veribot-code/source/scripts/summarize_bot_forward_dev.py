"""Compare frozen Bot-source interventions on the same sampled dev accounts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median


def summarize(path: Path, *, buffer: float) -> dict:
    raw = path.read_bytes()
    report = json.loads(raw)
    if report["split"] != "dev" or tuple(report["sampled_classes"]) != (1,):
        raise ValueError("Expected a Bot-only development sample")
    if report["completed_cases"] != len(report["selected_account_ids"]):
        raise ValueError("The frozen intervention audit is incomplete")
    cases = report["cases"]
    if ([case["account_id"] for case in cases] !=
            list(report["selected_account_ids"]) or
            any(int(case["label"]) != 1 for case in cases)):
        raise ValueError("Sample accounts differ from the fixed Bot IDs")
    threshold = report["threshold_selected_on_dev"]
    rows = []
    for case in cases:
        stage = case["tests"]["stage2"]
        full = case["tests"]["stage2_plus_hgt"]
        rows.append({
            "account_id": case["account_id"],
            "predicted_bot": bool(case["predicted_bot"]),
            "baseline_margin": case["baseline_margin"],
            "stage2_source_count": stage["source_count"],
            "stage2_keep_margin": stage["keep_margin"],
            "stage2_remove_margin": stage["remove_margin"],
            "stage2_buffered_bidirectional": (
                stage["keep_margin"] >= threshold + buffer and
                stage["remove_margin"] < threshold - buffer),
            "full_dependency_source_count": full["source_count"],
            "full_dependency_keep_margin": full["keep_margin"],
            "full_dependency_remove_margin": full["remove_margin"],
            "full_dependency_buffered_bidirectional": (
                full["keep_margin"] >= threshold + buffer and
                full["remove_margin"] < threshold - buffer),
        })
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "source_sha256": report["source_sha256"],
        "sampling_seed": report["sampling_seed"],
        "threshold": threshold,
        "n": len(rows),
        "correct_bot_predictions": sum(row["predicted_bot"] for row in rows),
        "stage2_buffered_bidirectional": sum(
            row["stage2_buffered_bidirectional"] for row in rows),
        "full_dependency_buffered_bidirectional": sum(
            row["full_dependency_buffered_bidirectional"] for row in rows),
        "median_stage2_sources": median(
            row["stage2_source_count"] for row in rows),
        "median_full_dependency_sources": median(
            row["full_dependency_source_count"] for row in rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--buffer", type=float, default=0.05)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.buffer < 0:
        raise ValueError("Buffer must be nonnegative")
    results = [summarize(path, buffer=args.buffer) for path in args.reports]
    ids = [row["account_id"] for row in results[0]["cases"]]
    if any([row["account_id"] for row in result["cases"]] != ids
           for result in results[1:]):
        raise ValueError("Model reports are not paired on identical Bot accounts")
    output = {"split": "dev", "buffer": args.buffer,
              "sampled_account_ids": ids,
              "interpretation": (
                  "Buffered class changes under frozen raw-source interventions "
                  "on the fixed development sample; not semantic ground truth."),
              "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n",
                           encoding="utf-8")
    for result in results:
        print(result["path"], result["n"],
              result["correct_bot_predictions"],
              result["stage2_buffered_bidirectional"],
              result["median_stage2_sources"])


if __name__ == "__main__":
    main()
