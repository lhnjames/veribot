"""Pair the locked 20-concept Fox8 model with official-config HypBot accounts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "output/v3_fox8_targeted_twenty_ab_group"
HYPBOT = ROOT / "output/hypbot_fox8_paired"
OUT = ROOT / "research/certificates/fox8_targeted_twenty_numeric_goal.json"


def run(seed: int, lock: dict) -> dict:
    model_path = MODEL / f"seed{seed}/test_predictions.json"
    hypbot_path = HYPBOT / f"seed{seed}/test_predictions.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    hypbot = json.loads(hypbot_path.read_text(encoding="utf-8"))
    ours = model["predictions"]
    theirs = hypbot["predictions"]
    if (len(ours) != 343 or len(theirs) != 343 or
            lock["runs"][seed]["split_sha256"] !=
            hypbot["manifest"]["split_sha256"] or
            model["threshold"] != lock["runs"][seed]["threshold"] or
            [(row["account_id"], row["label"]) for row in ours] !=
            [(row["account_id"], row["label"]) for row in theirs]):
        raise ValueError("paired Fox8 test population or threshold differs")
    labels = np.asarray([row["label"] for row in ours], dtype=int)
    own_prediction = np.asarray([row["prediction"] for row in ours], dtype=int)
    other_prediction = np.asarray([
        int(row["probability"] >= hypbot["manifest"]["threshold"])
        for row in theirs], dtype=int)
    own_correct = own_prediction == labels
    other_correct = other_prediction == labels
    own_acc = float(np.mean(own_correct))
    other_acc = float(np.mean(other_correct))
    if not own_acc > .994 or not own_acc > other_acc:
        raise ValueError("20-concept model misses strict numeric goal")
    return {
        "seed": seed,
        "test_accounts": len(labels),
        "split_sha256": lock["runs"][seed]["split_sha256"],
        "candidate_acc": own_acc,
        "hypbot_official_config_acc": other_acc,
        "candidate_correct": int(np.count_nonzero(own_correct)),
        "hypbot_correct": int(np.count_nonzero(other_correct)),
        "candidate_only_correct": int(np.count_nonzero(
            own_correct & ~other_correct)),
        "hypbot_only_correct": int(np.count_nonzero(
            other_correct & ~own_correct)),
        "candidate_predictions_sha256": sha256_file(model_path),
        "hypbot_predictions_sha256": sha256_file(hypbot_path),
        "candidate_model_sha256": lock["runs"][seed]["model_sha256"],
        "hypbot_checkpoint_sha256": hypbot["manifest"]["model_sha256"],
    }


def main() -> None:
    lock_path = MODEL / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["status"] != "selected_on_official_dev_before_targeted_twenty_test":
        raise ValueError("20-concept validation lock differs")
    runs = [run(seed, lock) for seed in (0, 1, 2)]
    OUT.write_text(json.dumps({
        "status": "twenty_concept_ab_group_tree_passes_strict_fox8_numeric_goal",
        "selection_lock_sha256": sha256_file(lock_path),
        "prior_test_population_exploration": True,
        "comparison": "same official account splits and labels; official-config HypBot uses its published kNN graph setup",
        "all_seeds_acc_strictly_above_0_994": True,
        "all_seeds_strictly_above_official_hypbot": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print({"candidate_acc": [row["candidate_acc"] for row in runs],
           "hypbot_acc": [row["hypbot_official_config_acc"]
                          for row in runs],
           "paired_net_correct": [row["candidate_only_correct"] -
                                  row["hypbot_only_correct"] for row in runs]})
    print(OUT)


if __name__ == "__main__":
    main()
