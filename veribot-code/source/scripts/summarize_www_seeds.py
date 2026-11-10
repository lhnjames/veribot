"""Aggregate every predefined seed without selecting a favorable run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def summarize_run(directory: Path) -> dict:
    report = json.loads((directory / "training_and_test_report.json").read_text())
    hgt = load_jsonl(directory / "hgt/predictions.jsonl")
    rel = load_jsonl(directory / "relnet_reimplementation/predictions.jsonl")
    if len(hgt) != len(rel) or any(
        a["account_id"] != b["account_id"] or a["split"] != b["split"] or
        a["label"] != b["label"] for a, b in zip(hgt, rel)
    ):
        raise RuntimeError(f"unaligned branch predictions: {directory}")
    h_threshold = report["branches"]["hgt"]["threshold"]
    r_threshold = report["branches"]["relnet_reimplementation"]["threshold"]
    test = [(a, b) for a, b in zip(hgt, rel) if a["split"] == "test"]
    rows = []
    for a, b in test:
        fused = (a["bot_probability"] + b["bot_probability"]) / 2
        rows.append({"id": a["account_id"], "label": a["label"],
                     "fused_probability": fused,
                     "fused_prediction": int(fused >= 0.5),
                     "conflict": (a["bot_probability"] >= h_threshold) !=
                                 (b["bot_probability"] >= r_threshold)})
    n = len(rows)
    k = sum(row["conflict"] for row in rows)
    all_errors = sum(row["fused_prediction"] != row["label"] for row in rows)
    conflict_errors = sum(row["fused_prediction"] != row["label"] and row["conflict"]
                          for row in rows)
    uncertain = sorted(rows, key=lambda row: abs(row["fused_probability"] - 0.5))[:k]
    uncertain_errors = sum(row["fused_prediction"] != row["label"] for row in uncertain)
    return {
        "seed": report["seed"], "graph_sha256": report["graph_sha256"],
        "n_test": n, "hgt_test_accuracy": report["branches"]["hgt"]["test_metrics"]["accuracy"],
        "relnet_test_accuracy": report["branches"]["relnet_reimplementation"]["test_metrics"]["accuracy"],
        "fused_test_accuracy": 1 - all_errors / n,
        "conflicts": k, "coverage_after_conflict_deferral": (n - k) / n,
        "selective_risk_conflict_deferral": (all_errors - conflict_errors) / (n - k) if n > k else None,
        "selective_risk_fused_uncertainty_matched": (all_errors - uncertain_errors) / (n - k)
        if n > k else None,
        "conflict_errors": conflict_errors, "all_errors": all_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = {}
    measures = ("hgt_test_accuracy", "relnet_test_accuracy", "fused_test_accuracy",
                "conflicts", "coverage_after_conflict_deferral",
                "selective_risk_conflict_deferral", "selective_risk_fused_uncertainty_matched")
    for dataset in ("fox8", "botsim"):
        runs = [summarize_run(args.runs_root / f"www_{dataset}_qwen_v1_seed{seed}")
                for seed in (42, 43, 44)]
        if len({run["graph_sha256"] for run in runs}) != 1:
            raise RuntimeError(f"different graphs among {dataset} seeds")
        aggregate = {
            measure: {"mean": statistics.mean(run[measure] for run in runs),
                      "sample_sd": statistics.stdev(run[measure] for run in runs)}
            for measure in measures
        }
        datasets[dataset] = {"runs": runs, "aggregate": aggregate}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(datasets, indent=2), encoding="utf-8")
    print(json.dumps(datasets, indent=2))


if __name__ == "__main__":
    main()
