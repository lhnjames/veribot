"""Audit whether source-bound concept groups change fixed test decisions.

This reads saved per-account cards and predictions; it does not fit models.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/compact_source_group_decision_value.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cards(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def audit(dataset: str, seed: int) -> dict:
    if dataset == "fox8":
        path = ROOT / f"output/v3_fox8_targeted_neighbor_20_concepts/seed{seed}/ab_group_tree_test_cards.jsonl.gz"
    elif dataset == "mgtab":
        path = ROOT / f"output/v3_mgtab_targeted_20_concept_ab_group_tree/seed{seed}/test_evidence_cards.jsonl.gz"
    else:
        path = ROOT / f"output/v3_computable/botsim_group_tail_12/seed{seed}/concept_evidence_cards.jsonl.gz"
    rows = cards(path)
    assert rows and len({r["account_id"] for r in rows}) == len(rows)
    pred_path = None
    if dataset == "botsim":
        pred_path = path.parent / "test_predictions.json"
        saved = json.loads(pred_path.read_text(encoding="utf-8"))["predictions"]
        labels = {r["account_id"]: r["label"] for r in saved}
        assert len(saved) == len(rows) == len(labels)
    groups = set()
    for row in rows:
        if dataset == "fox8":
            groups.update(row["source_group_zero_mask"])
        elif dataset == "mgtab":
            groups.update(row["group_effects"])
        else:
            groups.update(row["group_contributions"])
    counts = {g: {"changed_classes": 0, "correct_lost_after_mask": 0,
                  "errors_fixed_after_mask": 0, "masked_errors": 0} for g in sorted(groups)}
    full_errors = 0
    zero_evidence_probabilities = []
    for row in rows:
        aid = row["account_id"]
        label = int(labels[aid] if dataset == "botsim" else row["label"])
        pred = int(row["prediction"])
        threshold = float(row["threshold"])
        if dataset == "botsim":
            assert math.isclose(float(row["logit"]), float(row["intercept"]) +
                                sum(float(v) for v in row["group_contributions"].values()), abs_tol=1e-10)
            assert math.isclose(float(row["probability"]), sigmoid(float(row["logit"])), abs_tol=1e-12)
        assert pred == int(float(row["probability"]) >= threshold), (dataset, seed, aid)
        full_errors += pred != label
        if dataset == "fox8":
            zero_prob = float(row["all_evidence_mask_probability"])
            assert (int(zero_prob >= threshold) != pred) == bool(row["all_evidence_mask_changes_class"])
        elif dataset == "mgtab":
            zero_prob = float(row["zero_evidence_probability"])
            assert math.isclose(zero_prob, sigmoid(float(row["zero_evidence_logit"])), abs_tol=1e-12)
        else:
            zero_prob = sigmoid(float(row["intercept"]))
        zero_evidence_probabilities.append(zero_prob)
        for group in groups:
            if dataset == "fox8":
                item = row["source_group_zero_mask"][group]
                prob = float(item["probability"])
                flag = bool(item["changes_class"])
            elif dataset == "mgtab":
                item = row["group_effects"][group]
                prob = float(item["probability_without_group"])
                flag = bool(item["changes_class_without_group"])
            else:
                prob = sigmoid(float(row["logit"]) - float(row["group_contributions"][group]))
                flag = None
            masked = int(prob >= threshold)
            changed = masked != pred
            if flag is not None:
                assert changed == flag, (dataset, seed, aid, group)
            rec = counts[group]
            rec["changed_classes"] += changed
            rec["correct_lost_after_mask"] += pred == label and masked != label
            rec["errors_fixed_after_mask"] += pred != label and masked == label
            rec["masked_errors"] += masked != label
    for rec in counts.values():
        rec["net_correct_lost_after_mask"] = rec["correct_lost_after_mask"] - rec["errors_fixed_after_mask"]
        assert rec["masked_errors"] == full_errors + rec["net_correct_lost_after_mask"]
        assert rec["changed_classes"] == rec["correct_lost_after_mask"] + rec["errors_fixed_after_mask"]
    zero_min = min(zero_evidence_probabilities)
    zero_max = max(zero_evidence_probabilities)
    assert zero_max - zero_min < 1e-12, (dataset, seed, zero_min, zero_max)
    return {"dataset": dataset, "seed": seed, "accounts": len(rows),
            "full_errors": full_errors, "card_path": str(path.relative_to(ROOT)).replace('\\', '/'),
            "card_sha256": sha256(path),
            "prediction_sha256": sha256(pred_path) if pred_path else None,
            "all_evidence_zero_mask_probability": zero_min,
            "all_evidence_zero_mask_probability_range": zero_max - zero_min,
            "group_decision_value": counts}


def main() -> None:
    runs = [audit(dataset, seed) for dataset in ("botsim", "fox8", "mgtab") for seed in range(3)]
    report = {"scope": "saved_fixed_test_cards_exact_source_group_zero_mask_audit",
              "interpretation": "Input-group decision sensitivity of the saved fitted model, not independent semantic truth or a causal claim about the data-generating process.",
              "runs": runs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for run in runs:
        print(run["dataset"], run["seed"], "n", run["accounts"], "errors", run["full_errors"])
        for group, rec in run["group_decision_value"].items():
            if rec["changed_classes"]:
                print(" ", group, rec)
    print(OUT)


if __name__ == "__main__":
    main()
