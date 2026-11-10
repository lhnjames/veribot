"""Lock a TwiBot-20 reasoner from official dev metrics before test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock(screen_path: Path) -> dict:
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    if screen["dataset"] != "twibot20" or \
            screen["scope"] != "official_dev_only" or \
            screen["test_predictions_accessed"] is not False or \
            [row["seed"] for row in screen["runs"]] != [0, 1, 2]:
        raise ValueError("selection screen is not the fixed three-seed official dev protocol")
    candidates = ("spline_22", "dual_tail_22", "spline_30", "dual_tail_30")
    averages = {
        name: {metric: statistics.mean(row[name]["validation"][metric]
                                       for row in screen["runs"])
               for metric in ("macro_f1", "acc", "auroc")}
        for name in candidates
    }
    selected = max(candidates, key=lambda name: (
        averages[name]["macro_f1"], averages[name]["acc"], name))
    return {
        "dataset": "twibot20", "selection_scope": "official_dev_only",
        "criterion": "maximum three-seed mean dev Macro-F1, tie by mean dev ACC",
        "selected_model": selected,
        "candidate_mean_dev_metrics": averages,
        "screen_sha256": sha256(screen_path),
        "base_facts_sha256": screen["base_facts_sha256"],
        "lexical_facts_sha256": screen["lexical_facts_sha256"],
        "source_archive_sha256": screen["source_archive_sha256"],
        "test_predictions_accessed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = lock(args.screen)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
