"""Descriptive fixed-test diagnostic of the frozen MGTAB projection group."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "output/v3_mgtab_150_neutral_anchor/fixed_test_report.json"
LEDGER = ROOT / "research/certificates/three_dataset_signed_group_ledger.json"
LABELS = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB/labels_bot.pt")
OUT = ROOT / "research/certificates/mgtab_neutral_projection_fixed_test_diagnostic.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    fixed = json.loads(REPORT.read_text(encoding="utf-8"))
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    labels = torch.load(LABELS, map_location="cpu",
                        weights_only=True).numpy().astype(bool)
    assert len(labels) == 10199
    runs = []
    for seed in (0, 1, 2):
        record = fixed["runs"][seed]
        source = next(x for x in ledger["runs"]
                      if x["dataset"] == "mgtab" and x["seed"] == seed)
        path = Path(source["per_account_group_matrix_path"])
        assert sha256(path) == source["per_account_group_matrix_sha256"]
        prediction_path = Path(record["predictions_path"])
        assert sha256(prediction_path) == record["predictions_sha256"]
        with np.load(path, allow_pickle=False) as z:
            ids = z["account_ids"].astype(int)
            names = z["source_group_names"].astype(str).tolist()
            logit = z["logit"].astype(float)
            effects = z["group_logit_effect"].astype(float)
            pred = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        with np.load(prediction_path, allow_pickle=False) as z:
            assert np.array_equal(ids, z["account_ids"].astype(int))
            assert np.array_equal(pred, z["predicted_bot"].astype(bool))
        truth = labels[ids]
        clean_correct = pred == truth
        assert int(clean_correct.sum()) == record["correct"]
        j = names.index("published_tweet_projections")
        masked_p = expit(logit - effects[:, j])
        runtime = MGTABNeutralAnchorRuntime(ROOT, seed)
        rows = np.asarray([runtime.index[str(uid)] for uid in ids])
        design = evidence_design(runtime, rows)
        columns = np.asarray(runtime.group_fact_indices[
            "published_tweet_projections"], dtype=int)
        masked_design = design.copy()
        masked_design[:, columns] = 0.0
        masked_design[:, columns + 150] = 0.0
        direct_p = expit(np.asarray(runtime.model.booster_.predict(
            masked_design, raw_score=True), dtype=float))
        direct_error = float(np.max(np.abs(masked_p - direct_p)))
        assert direct_error < 1e-12
        masked_pred = masked_p >= threshold
        masked_correct = masked_pred == truth
        gain = int(np.count_nonzero(clean_correct & ~masked_correct))
        harm = int(np.count_nonzero(~clean_correct & masked_correct))
        assert gain + harm == int(np.count_nonzero(pred != masked_pred))
        runs.append({
            "seed": seed, "accounts": len(ids),
            "full_correct": int(clean_correct.sum()),
            "without_projection_correct": int(masked_correct.sum()),
            "full_acc": float(clean_correct.mean()),
            "without_projection_acc": float(masked_correct.mean()),
            "full_macro_f1": float(f1_score(truth, pred, average="macro")),
            "without_projection_macro_f1": float(f1_score(
                truth, masked_pred, average="macro")),
            "without_projection_auroc": float(roc_auc_score(truth, masked_p)),
            "direct_mask_probability_max_error": direct_error,
            "decisions_flipped": gain + harm,
            "projection_helps_accounts": gain,
            "projection_hurts_accounts": harm,
            "net_extra_correct_from_projection": gain - harm,
            "hypbot_correct_on_same_population": record[
                "hypbot_correct_on_same_test_population"],
            "prediction_file_sha256": record["predictions_sha256"],
            "ledger_matrix_sha256": source["per_account_group_matrix_sha256"]})
    result = {
        "scope": "frozen_model_projection_group_masking_diagnostic_only",
        "test_population_previously_explored_by_other_candidates": True,
        "no_model_or_threshold_selection_from_this_result": True,
        "published_label_file_sha256": sha256(LABELS),
        "runs": runs,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{k: r[k] for k in (
        "seed", "full_correct", "without_projection_correct",
        "projection_helps_accounts", "projection_hurts_accounts",
        "hypbot_correct_on_same_population")} for r in runs], indent=2))


if __name__ == "__main__":
    main()
