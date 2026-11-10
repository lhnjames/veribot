"""Check label-free source-bound entry on fixed test accounts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    original_read_csv = pd.read_csv
    original_read_text = Path.read_text
    original_open = Path.open
    forbidden_reads = []

    def protected(path) -> bool:
        name = Path(path).name.lower()
        return (name in {"test_predictions.json", "fixed_test_report.json",
                         "report.json", "labels.csv", "label.csv"} or
                "test_evidence" in name)

    def no_test_text(path, *args, **kwargs):
        if protected(path):
            forbidden_reads.append(str(path))
            raise AssertionError("Fox8 inference read test artifact")
        return original_read_text(path, *args, **kwargs)

    def no_test_open(path, *args, **kwargs):
        if protected(path):
            forbidden_reads.append(str(path))
            raise AssertionError("Fox8 inference opened test artifact")
        return original_open(path, *args, **kwargs)

    def no_labels(path, *args, **kwargs):
        if Path(path).name.lower() in {"labels.csv", "label.csv"}:
            raise AssertionError("runtime attempted to open class labels")
        return original_read_csv(path, *args, **kwargs)

    results = []
    with patch.object(pd, "read_csv", no_labels):
        for seed in (0, 1, 2):
            saved_path = root / ("output/v3_fox8_neighbor_atomic_evidence_capacity/"
                                 f"seed{seed}/test_predictions.json")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
            chosen = [saved[0], saved[len(saved) // 2], saved[-1]]
            with patch.object(Path, "read_text", no_test_text), \
                    patch.object(Path, "open", no_test_open):
                runtime = Fox8NeighborAtomicRuntime(root, seed)
                observed = [runtime.predict(row["account_id"])
                            for row in chosen]
            max_difference = 0.0
            for row, result in zip(chosen, observed, strict=True):
                max_difference = max(max_difference,
                                     abs(result["probability"] - row["probability"]))
                if (len(result["evidence"]) != 74
                        or abs(sum(term["logit_contribution"]
                                   for term in result["evidence"]) +
                               result["intercept_logit"] - result["logit"]) > 1e-9):
                    raise ValueError("runtime source evidence does not reconstruct score")
            if max_difference > 1e-14 or forbidden_reads:
                raise ValueError("source-bound runtime differs from saved test")
            results.append({"seed": seed, "sample_count": len(chosen),
                            "max_probability_difference": max_difference,
                            "label_file_read_attempts": 0,
                            "test_artifact_read_attempts": len(forbidden_reads)})
            print(results[-1], flush=True)
    path = root / "research/certificates/fox8_neighbor_atomic_source_bound_entry.json"
    path.write_text(json.dumps({
        "scope": "Fox8_frozen_source_bound_label_free_inference_samples",
        "results": results}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
