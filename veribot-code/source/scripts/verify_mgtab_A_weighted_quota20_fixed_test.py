"""Independently recount locked quota-20 test predictions and source budgets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from gate0_shallow_tree import evaluate, mgtab_split, sha256_file


ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "output/v3_mgtab_A_weighted_quota20_facts"
OUT = ROOT / "research/certificates/mgtab_A_weighted_quota20_fixed_test_readback.json"


def main() -> None:
    report_path = FOLDER / "fixed_test_report.json"
    lock_path = FOLDER / "selection_lock.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = json.loads((ROOT / "data/raw/mgtab_source.json").read_text(
        encoding="utf-8"))
    labels = torch.load(Path(source["path"]) / "labels_bot.pt",
                        map_location="cpu", weights_only=True).numpy().astype(int)
    if (len(labels) != 10199 or report["selection_lock_sha256"] !=
            sha256_file(lock_path) or len(report["runs"]) != 3):
        raise ValueError("test report, lock or population differs")
    checked = []
    for seed, run in enumerate(report["runs"]):
        split = mgtab_split(labels, seed)["test"]
        artifact = lock["runs"][seed]
        predictions_path = Path(run["test_predictions_path"])
        if (run["seed"] != seed or artifact["seed"] != seed or
                sha256_file(predictions_path) != run["test_predictions_sha256"] or
                len(split) != run["test_accounts"]):
            raise ValueError("fixed test account or prediction artifact differs")
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
        rows = data["predictions"]
        with np.load(artifact["selector_path"], allow_pickle=False) as selector:
            allowed_names = set(selector["names"].astype(str).tolist())
            quota = tuple(selector["quota_profile_relation_content"].astype(int))
        if quota != (4, 10, 6) or len(allowed_names) != 150:
            raise ValueError("locked source vocabulary or quota differs")
        if (len(rows) != len(split) or data["seed"] != seed or
                data["selector_sha256"] != artifact["selector_sha256"] or
                data["model_sha256"] != artifact["model_sha256"] or
                data["threshold"] != artifact["threshold"]):
            raise ValueError("frozen model identity differs")
        probabilities = np.asarray([row["probability"] for row in rows],
                                   dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("nonfinite predicted probability")
        by_group = Counter()
        for source_index, row in zip(split, rows):
            if (row["account_id"] != str(int(source_index)) or
                    row["label"] != int(labels[source_index]) or
                    row["prediction"] != int(row["probability"] >=
                                             data["threshold"]) or
                    len(row["selected_fact_names"]) != 20 or
                    len(set(row["selected_fact_names"])) != 20 or
                    not set(row["selected_fact_names"]) <= allowed_names):
                raise ValueError("account, label or twenty-fact card differs")
            account_groups = Counter()
            for name in row["selected_fact_names"]:
                group = ("published_tweet_projection" if name.startswith(
                    "published_tweet_projection") else
                    "published_profile" if name.startswith(
                        "published_profile_coordinate") else
                    "published_relation")
                by_group[group] += 1
                account_groups[group] += 1
            if tuple(account_groups[name] for name in (
                    "published_profile", "published_relation",
                    "published_tweet_projection")) != quota:
                raise ValueError("per-account source quota differs")
        metric = evaluate(labels[split], probabilities, data["threshold"])
        if any(abs(metric[key] - run["test_metrics"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("independent metric recount differs")
        hypbot_path = ROOT / f"output/hypbot_mgtab_paired/seed{seed}/test_predictions.json"
        hypbot = json.loads(hypbot_path.read_text(encoding="utf-8"))
        if (sha256_file(hypbot_path) != run["hypbot_predictions_sha256"] or
                len(hypbot["predictions"]) != len(split)):
            raise ValueError("paired HypBot prediction artifact differs")
        baseline_correct = sum(int((row["probability"] >= hypbot["manifest"][
            "threshold"]) == row["label"]) for row in hypbot["predictions"])
        if (baseline_correct != run["hypbot_correct_same_accounts"] or
                run["strict_numeric_goal_passes"] != bool(
                    metric["acc"] > .912 and
                    metric["acc"] > baseline_correct / len(split))):
            raise ValueError("paired baseline or strict gate differs")
        checked.append({
            "seed": seed, "accounts": len(split),
            "correct": int(np.sum(
                (probabilities >= data["threshold"]) == labels[split])),
            "hypbot_correct": baseline_correct,
            "selected_source_group_slot_counts": dict(by_group),
            "selected_facts_per_account": 20,
            "metrics": metric,
            "strict_numeric_goal_passes": run["strict_numeric_goal_passes"],
        })
        print({"seed": seed, "correct": checked[-1]["correct"],
               "hypbot_correct": baseline_correct,
               "selected_groups": dict(by_group)}, flush=True)
    OUT.write_text(json.dumps({
        "status": "locked_quota20_test_predictions_independently_recounted",
        "test_report_sha256": sha256_file(report_path),
        "all_three_strict_numeric_goal_pass": all(
            row["strict_numeric_goal_passes"] for row in checked),
        "runs": checked,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
