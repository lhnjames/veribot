"""Count frozen decisions that change when every evidence term is zero-masked."""

from __future__ import annotations

import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    models = {
        "botsim": root / "output/v3_computable/botsim_group_tail_12",
        "fox8": root / "output/v3_fox8_native_spline_l2/group_tail_20_selected",
        "mgtab": root / "output/v3_mgtab_ab_spline_content/direct_selected_20",
        "twibot20": root / "output/v3_twibot20_sparse_text_evidence/selected",
    }
    rows = []
    for dataset, base in models.items():
        for seed in ((None,) if dataset == "twibot20" else (0, 1, 2)):
            folder = base if seed is None else base / f"seed{seed}"
            report = read(folder / "report.json")
            saved = read(folder / "test_predictions.json")
            threshold = float(report["manifest"]["threshold"])
            if dataset == "twibot20":
                intercept = float(report["reasoner"]["intercept"])
                all_mask_probability = 1 / (1 + math.exp(-intercept))
            else:
                all_mask_probability = float(report["all_evidence_mask_probability"])
            baseline = int(all_mask_probability >= threshold)
            predicted = [int(float(row["probability"]) >= threshold)
                         for row in saved["predictions"]]
            changed = sum(value != baseline for value in predicted)
            rows.append({
                "dataset": dataset, "seed": seed,
                "test_n": len(predicted),
                "all_evidence_zero_mask_probability": all_mask_probability,
                "all_evidence_zero_mask_prediction": baseline,
                "decisions_changed_by_all_evidence_mask": changed,
                "fraction_changed_by_all_evidence_mask": changed / len(predicted),
                "positive_predictions": sum(predicted),
                "saved_test_labels_read": False,
            })
    output = {
        "scope": "frozen_test_decisions_only_no_model_selection",
        "interpretation": "A changed class requires some evidence terms under the frozen model; an unchanged class may still receive nonzero evidence contribution.",
        "rows": rows,
    }
    target = root / "research/all_evidence_mask_dependency_2026-11-03.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
