"""Apply the frozen, dev-selected HGT/RelNet policy to saved account logits."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "research/certificates/aaai_veribot_dev_threshold_audit.json"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def apply(dataset: str) -> dict:
    directory = ROOT / f"output/aaai_veribot_{dataset}/seed0"
    policy_path = directory / "decision_policy.json"
    source_path = directory / "test_predictions.json"
    checkpoint_path = directory / "model.pt"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    result = next(row for row in audit["results"] if row["dataset"] == dataset)
    selected = result["variants"]["mean_logit"]
    if (policy["dataset"] != dataset or
            policy["branch_combination"] != "mean_logit_hgt_relnet" or
            policy["selection_population"] != "dev only" or
            policy["threshold_selection_audit_sha256"] != digest(AUDIT) or
            policy["checkpoint_sha256"] != digest(checkpoint_path) or
            policy["checkpoint_sha256"] != source["checkpoint_sha256"] or
            policy["saved_predictions_sha256"] != digest(source_path) or
            policy["logit_threshold"] != selected["threshold_logit"]):
        raise ValueError(f"{dataset}: decision policy does not match frozen model and dev audit")

    threshold = float(policy["logit_threshold"])
    rows = []
    for row in source["predictions"]:
        fused = 0.5 * (float(row["hgt_logit"]) + float(row["relnet_logit"]))
        if abs(fused - float(row["fused_logit"])) > 1e-5:
            raise ValueError(f"{dataset}: fused logit mismatch at {row['account_id']}")
        rows.append({"account_id": row["account_id"], "label": int(row["label"]),
                     "hgt_logit": row["hgt_logit"], "relnet_logit": row["relnet_logit"],
                     "fused_logit": row["fused_logit"],
                     "predicted_label": int(fused >= threshold)})
    truth = [row["label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    scores = [row["fused_logit"] for row in rows]
    metrics = {"n": len(rows), "correct": sum(a == b for a, b in zip(truth, predicted)),
               "accuracy": float(accuracy_score(truth, predicted)),
               "macro_f1": float(f1_score(truth, predicted, average="macro")),
               "auc": float(roc_auc_score(truth, scores))}
    if (metrics["n"] != policy["test_n"] or
            metrics["correct"] != policy["test_correct"] or
            abs(metrics["accuracy"] - policy["test_accuracy"]) > 1e-12 or
            abs(metrics["macro_f1"] - policy["test_macro_f1"]) > 1e-12):
        raise ValueError(f"{dataset}: calibrated predictions disagree with dev audit")
    artifact = {"algorithm": "AAAI_VeriBot_HGT_RelNet_mean_logit_dev_calibrated",
                "dataset": dataset, "checkpoint_sha256": digest(checkpoint_path),
                "source_predictions_sha256": digest(source_path),
                "decision_policy_sha256": digest(policy_path),
                "logit_threshold": threshold, "test": metrics, "predictions": rows}
    destination = directory / "calibrated_test_predictions.json"
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return {"dataset": dataset, "path": str(destination), "test": metrics}


if __name__ == "__main__":
    print(json.dumps([apply(dataset) for dataset in ("fox8", "botsim")], indent=2))
