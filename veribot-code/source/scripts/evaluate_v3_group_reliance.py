"""Evaluate saved v3 evidence-group reliance on test accounts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.group_reliance import evaluate_group_reliance


def evaluate(directory: Path, report_name: str) -> dict:
    report = json.loads((directory / report_name).read_text(encoding="utf-8"))
    rows = json.loads((directory / "test_predictions.json").read_text(encoding="utf-8"))["predictions"]
    names = list(report["reasoner"]["groups"])
    if not rows or any(set(row["group_contributions"]) != set(names) for row in rows):
        raise ValueError("saved rows do not cover all reasoner groups")
    columns = {name: np.array([row["group_contributions"][name] for row in rows])
               for name in names}
    result = evaluate_group_reliance(
        np.array([row["label"] for row in rows]),
        np.array([row["probability"] for row in rows]),
        columns, report["reasoner"]["intercept"], report["manifest"]["threshold"])
    metric = report["test"]
    if result["n"] != metric["n"] or abs(result["original_acc"] - metric["acc"]) > 1e-12:
        raise ValueError("saved prediction evaluation disagrees with report")
    return {"directory": str(directory), "dataset": report["manifest"]["dataset"],
            "seed": report["manifest"]["seed"], **result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    result = [evaluate(args.model_dir / f"seed{seed}", args.report_name)
              for seed in args.seeds]
    output = args.model_dir / "group_reliance.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "summary": [
        {"seed": row["seed"], "original_acc": row["original_acc"],
         "combined_graph_removal": row["combined_graph_removal"]}
        for row in result]}, indent=2))


if __name__ == "__main__":
    main()
