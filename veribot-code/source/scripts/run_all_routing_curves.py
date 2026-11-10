"""Generate routing risk-coverage curves for every predefined WWW run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, required=True)
    args = parser.parse_args()
    evaluator = Path(__file__).with_name("eval_branch_routing.py")
    for dataset in ("fox8", "botsim"):
        for seed in (42, 43, 44):
            run = args.runs_root / f"www_{dataset}_qwen_v1_seed{seed}"
            report = json.loads((run / "training_and_test_report.json").read_text())
            command = [
                sys.executable, str(evaluator),
                "--hgt", str(run / "hgt/predictions.jsonl"),
                "--relnet", str(run / "relnet_reimplementation/predictions.jsonl"),
                "--hgt-threshold", str(report["branches"]["hgt"]["threshold"]),
                "--relnet-threshold", str(report["branches"]["relnet_reimplementation"]["threshold"]),
                "--split", "test", "--output", str(run / "routing_test"),
            ]
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
            result = json.loads((run / "routing_test/routing_report.json").read_text())
            print(f"{dataset} seed={seed} conflicts={result['conflicts']} "
                  f"risk={result['defer_conflicts_selective_risk']:.5f} "
                  f"uncertainty_risk={result['matched_coverage_controls']['fused_uncertainty']['selective_risk']:.5f}",
                  flush=True)


if __name__ == "__main__":
    main()
