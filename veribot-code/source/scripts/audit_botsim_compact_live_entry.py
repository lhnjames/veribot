"""Check sealed BotSim inference on held-out and other accounts without test reads."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from predict_compact_from_verified_sources import predict


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    selected = root / "output/v3_computable/botsim_group_tail_12"
    users = pd.read_csv(root / "data/raw/botsim_24/Users.csv")["user_id"].astype(str).tolist()
    original_read = Path.read_text
    original_open = Path.open
    runs = []
    for seed in (0, 1, 2):
        saved_path = selected / f"seed{seed}/test_predictions.json"
        frozen = json.loads(saved_path.read_text(encoding="utf-8"))["predictions"]
        test_ids = [row["account_id"] for row in frozen]
        non_test_id = next(account for account in users if account not in
                           {row["account_id"] for row in frozen})
        forbidden = {"report.json", "test_predictions.json", "labels.csv", "labels_bot.pt"}
        def checked_read(path, *args, **kwargs):
            if path.name in forbidden:
                raise AssertionError(f"evaluation artifact read during inference: {path}")
            return original_read(path, *args, **kwargs)
        def checked_open(path, *args, **kwargs):
            if path.name in forbidden:
                raise AssertionError(f"evaluation artifact opened during inference: {path}")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "read_text", checked_read), patch.object(Path, "open", checked_open):
            result = predict(root, "botsim", seed, [*test_ids, non_test_id])
        actual_rows, other = result["decisions"][:-1], result["decisions"][-1]
        max_difference = 0.0
        for actual, expected in zip(actual_rows, frozen, strict=True):
            max_difference = max(max_difference, abs(actual["probability"] -
                                                     expected["probability"]))
            if (actual["account_id"] != expected["account_id"]
                    or len(actual["terms"]) != 12
                    or abs(actual["intercept"] + sum(t["logit_contribution"]
                                                        for t in actual["terms"]) -
                           actual["logit"]) > 1e-10
                    or abs(actual["decision_dependency"][
                        "full_logit_reconstruction_difference"]) > 1e-10):
                raise ValueError(f"seed{seed} account evidence differs")
        if (other["account_id"] != non_test_id or max_difference > 1e-10
                or result["classification_labels_accessed"]):
            raise ValueError(f"seed{seed} sealed source-to-decision inference differs")
        runs.append({"seed": seed, "test_accounts_checked": len(test_ids),
                     "other_account_id": non_test_id,
                     "max_saved_probability_difference": max_difference,
                     "concept_count": 12,
                     "classification_labels_accessed": False,
                     "evaluation_artifacts_opened": False})
    manifest_path = root / "output/v3_current_compact_deployment/botsim_manifest.json"
    certificate = {"dataset": "botsim", "scope": "sealed_live_source_to_decision_inference",
                   "manifest_sha256": sha256_file(manifest_path), "runs": runs,
                   "status": "all_fixed_test_predictions_match_without_evaluation_reads"}
    path = root / "research/certificates/botsim_compact_live_entry_audit.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(certificate, indent=2))


if __name__ == "__main__":
    main()
