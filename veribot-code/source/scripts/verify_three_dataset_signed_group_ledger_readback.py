"""Read back all signed source-group matrices without calling model runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/three_dataset_signed_group_ledger.json"
REPORTS = {
    "botsim": ROOT / "output/v3_botsim_neutral_threshold/fixed_test_report.json",
    "fox8": ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json",
    "mgtab": ROOT / "output/v3_mgtab_150_neutral_anchor/fixed_test_report.json",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    certificate = json.loads(CERT.read_text(encoding="utf-8"))
    assert certificate["total_accounts"] == 5400
    assert len(certificate["runs"]) == 9
    reports = {name: json.loads(path.read_text(encoding="utf-8"))
               for name, path in REPORTS.items()}
    total = 0
    projection_support = projection_counter = projection_decision_flips = 0
    projection_flip_runs = []
    maximum_probability_error = maximum_logit_error = 0.0
    for row in certificate["runs"]:
        dataset, seed = row["dataset"], int(row["seed"])
        path = Path(row["per_account_group_matrix_path"])
        assert digest(path) == row["per_account_group_matrix_sha256"]
        saved = reports[dataset]["runs"][seed]
        pred_path = Path(saved["predictions_path" if dataset == "mgtab"
                               else "prediction_matrix_path"])
        assert digest(pred_path) == row["fixed_prediction_sha256"]
        with np.load(path, allow_pickle=False) as z:
            ids = z["account_ids"].astype(str)
            names = z["source_group_names"].astype(str).tolist()
            logit = z["logit"].astype(float)
            zero = z["zero_evidence_logit"].astype(float)
            effects = z["group_logit_effect"].astype(float)
            support = z["supports_prediction"].astype(bool)
            counter = z["counterevidence"].astype(bool)
            pred = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        with np.load(pred_path, allow_pickle=False) as z:
            saved_ids = z["account_ids"].astype(str)
            saved_p = z["probability"].astype(float)
            saved_pred = z["predicted_bot"].astype(bool)
            saved_threshold = float(z["threshold"])
        assert len(ids) == len(np.unique(ids)) == row["accounts"]
        assert len(names) == len(set(names))
        assert effects.shape == (len(ids), len(names))
        assert np.array_equal(ids, saved_ids)
        assert np.array_equal(pred, saved_pred)
        assert np.isfinite(effects).all() and np.isfinite(logit).all()
        assert abs(threshold - saved_threshold) < 1e-12
        p_error = float(np.max(np.abs(expit(logit) - saved_p)))
        logit_error = float(np.max(np.abs(zero + effects.sum(axis=1) - logit)))
        assert p_error < 1e-12 and logit_error < 1e-9
        assert np.array_equal(saved_p >= threshold, pred)
        assert np.all(np.where(pred, logit > zero, logit < zero))
        expected_support = np.where(pred[:, None], effects > 1e-10,
                                    effects < -1e-10)
        expected_counter = np.where(pred[:, None], effects < -1e-10,
                                    effects > 1e-10)
        assert np.array_equal(support, expected_support)
        assert np.array_equal(counter, expected_counter)
        assert support.any(axis=1).all()
        assert int(support.any(axis=1).sum()) == row["accounts_with_supporting_group"]
        assert int(counter.any(axis=1).sum()) == row["accounts_with_counterevidence_group"]
        assert int(pred.sum()) == row["bot_predictions"]
        assert int((~pred).sum()) == row["human_predictions"]
        for j, name in enumerate(names):
            assert int(support[:, j].sum()) == row["supporting_accounts_by_group"][name]
            assert int(counter[:, j].sum()) == row["counterevidence_accounts_by_group"][name]
        if dataset == "mgtab":
            j = names.index("published_tweet_projections")
            assert np.count_nonzero(np.abs(effects[:, j]) <= 1e-10) == 0
            projection_support += int(support[:, j].sum())
            projection_counter += int(counter[:, j].sum())
            without_projection = expit(logit - effects[:, j]) >= threshold
            flips = int(np.count_nonzero(without_projection != pred))
            projection_decision_flips += flips
            projection_flip_runs.append({"seed": seed, "decision_flips": flips,
                                         "bot_to_human": int(np.count_nonzero(
                                             pred & ~without_projection)),
                                         "human_to_bot": int(np.count_nonzero(
                                             ~pred & without_projection))})
        maximum_probability_error = max(maximum_probability_error, p_error)
        maximum_logit_error = max(maximum_logit_error, logit_error)
        total += len(ids)
    assert total == 5400
    assert projection_support + projection_counter == 3060
    result = {
        "status": "readback_verified",
        "accounts": total,
        "runs": len(certificate["runs"]),
        "maximum_probability_error": maximum_probability_error,
        "maximum_logit_reconstruction_error": maximum_logit_error,
        "mgtab_projection_support": projection_support,
        "mgtab_projection_counterevidence": projection_counter,
        "mgtab_projection_nonzero_accounts": projection_support + projection_counter,
        "mgtab_projection_decision_flips": projection_decision_flips,
        "mgtab_projection_flip_runs": projection_flip_runs,
        "labels_read": False,
        "semantic_ground_truth_claimed": False,
    }
    out = ROOT / "research/certificates/three_dataset_signed_group_ledger_readback.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
