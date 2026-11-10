"""Check the nine best-model entry paths against frozen predictions without test reads."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_best_evidence_models import MODELS, predict


ROOT = Path(__file__).resolve().parents[2]
SAVED = {
    "botsim": "output/v3_computable/botsim_group_tail_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}


def main() -> None:
    original_text = Path.read_text
    original_open = Path.open
    original_csv = pd.read_csv
    original_torch = torch.load
    forbidden = []

    def protected(path) -> bool:
        name = Path(path).name.lower()
        return ("test_predictions" in name or "test_evidence" in name or
                name in {"fixed_test_report.json", "report.json",
                         "labels.csv", "label.csv", "labels_bot.pt",
                         "labels.pt"})

    def checked_text(path, *args, **kwargs):
        if protected(path):
            forbidden.append(str(path))
            raise AssertionError("best-model inference read test or label data")
        return original_text(path, *args, **kwargs)

    def checked_open(path, *args, **kwargs):
        if protected(path):
            forbidden.append(str(path))
            raise AssertionError("best-model inference opened test or label data")
        return original_open(path, *args, **kwargs)

    def checked_csv(path, *args, **kwargs):
        if protected(path):
            forbidden.append(str(path))
            raise AssertionError("best-model inference read label CSV")
        return original_csv(path, *args, **kwargs)

    def checked_torch(path, *args, **kwargs):
        if protected(path):
            forbidden.append(str(path))
            raise AssertionError("best-model inference read label tensor")
        return original_torch(path, *args, **kwargs)

    rows = []
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    for dataset in MODELS:
        for seed in (0, 1, 2):
            saved_path = ROOT / SAVED[dataset] / f"seed{seed}/test_predictions.json"
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            item = saved["predictions"][0]
            before = len(forbidden)
            with patch.object(Path, "read_text", checked_text), \
                    patch.object(Path, "open", checked_open), \
                    patch.object(pd, "read_csv", checked_csv), \
                    patch.object(torch, "load", checked_torch):
                observed = predict(ROOT, dataset, seed, item["account_id"])
            decision = observed["decision"]
            difference = abs(float(decision["probability"])-
                             float(item["probability"]))
            if (difference > 1e-12 or
                    decision["account_id"] != item["account_id"] or
                    "label" in decision or len(forbidden) != before):
                raise ValueError("unified best-model prediction or read guard differs")
            rows.append({
                "dataset": dataset, "seed": seed,
                "account_id": item["account_id"],
                "model_id": observed["model_id"],
                "evidence_term_count": observed["evidence_term_count"],
                "probability_difference": difference,
                "forbidden_reads": len(forbidden)-before,
            })
            print(rows[-1], flush=True)
    output = ROOT / "research/certificates/best_evidence_entry_nine_run_audit.json"
    output.write_text(json.dumps({
        "scope": "one_fixed_test_account_per_dataset_seed_in_unified_best_entry",
        "runs": rows}, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
