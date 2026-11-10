"""Recompute paired ACC gates for the three frozen evidence classifiers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


CONFIG = {
    "botsim": {
        "candidate": "output/v3_computable/botsim_group_tail_12",
        "hypbot": "output/hypbot_botsim_paired",
        "test_accounts": 437,
        "required_acc_strictly_above": 0.994,
        "candidate_artifacts": {},
    },
    "fox8": {
        "candidate": "output/v3_fox8_neighbor_atomic_evidence_capacity",
        "hypbot": "output/hypbot_fox8_paired",
        "hypbot_native": "output/hypbot_fox8_native_graph_paired",
        "test_accounts": 343,
        "required_acc_strictly_above": 0.994,
        "candidate_artifacts": {
            "model_sha256": "model.joblib",
            "reference_sha256": "label_blind_reference.json.gz",
            "evidence_sha256": "test_evidence.npz",
            "cards_sha256": "test_evidence_cards.jsonl",
        },
    },
    "mgtab": {
        "candidate": "output/v3_mgtab_ab_group_additive_tree",
        "hypbot": "output/hypbot_mgtab_paired",
        "test_accounts": 1020,
        "required_acc_strictly_above": 0.912,
        "candidate_artifacts": {
            "model_sha256": "model.joblib",
            "reference_sha256": "label_blind_reference.json.gz",
            "evidence_sha256": "test_evidence.npz",
            "cards_sha256": "test_evidence_cards.jsonl",
            "ab_contract_sha256": "ab_direction_contract.npz",
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_predictions(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("predictions"), list):
        raise ValueError(f"prediction list absent: {path}")
    return payload


def compare(candidate: dict, baseline: dict, n: int) -> dict:
    cm, bm = candidate["manifest"], baseline["manifest"]
    cr, br = candidate["predictions"], baseline["predictions"]
    if (cm["split_sha256"] != bm["split_sha256"]
            or len(cr) != n or len(br) != n
            or len({row["account_id"] for row in cr}) != n
            or any(c["account_id"] != b["account_id"]
                   or c["label"] != b["label"]
                   for c, b in zip(cr, br, strict=True))):
        raise ValueError("paired split, IDs or labels differ")
    labels = np.asarray([row["label"] for row in cr], dtype=int)
    cp = np.asarray([row["probability"] for row in cr], dtype=float)
    bp = np.asarray([row["probability"] for row in br], dtype=float)
    if (not set(labels) <= {0, 1} or
            not np.isfinite(cp).all() or not np.isfinite(bp).all() or
            np.any((cp < 0) | (cp > 1)) or
            np.any((bp < 0) | (bp > 1))):
        raise ValueError("paired labels or probabilities invalid")
    c = (cp >= float(cm["threshold"])) == labels
    b = (bp >= float(bm["threshold"])) == labels
    return {
        "test_accounts": n,
        "candidate_correct": int(c.sum()),
        "hypbot_correct": int(b.sum()),
        "candidate_acc": float(c.mean()),
        "hypbot_acc": float(b.mean()),
        "candidate_only_correct": int(np.count_nonzero(c & ~b)),
        "hypbot_only_correct": int(np.count_nonzero(~c & b)),
        "candidate_strictly_above_hypbot": bool(c.sum() > b.sum()),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    datasets = []
    for dataset, spec in CONFIG.items():
        candidate_dir = root / spec["candidate"]
        lock_path = candidate_dir / "selection_lock.json"
        lock_sha = sha256_file(lock_path)
        runs = []
        for seed in (0, 1, 2):
            cdir = candidate_dir / f"seed{seed}"
            cpath = cdir / "test_predictions.json"
            bpath = root / spec["hypbot"] / f"seed{seed}/test_predictions.json"
            candidate = load_predictions(cpath)
            baseline = load_predictions(bpath)
            manifest = candidate["manifest"]
            if (manifest["dataset"] != dataset or manifest["seed"] != seed
                    or manifest["selection_lock_sha256"] != lock_sha
                    or baseline["manifest"]["dataset"] != dataset
                    or baseline["manifest"]["seed"] != seed):
                raise ValueError("frozen candidate or baseline manifest differs")
            for key, filename in spec["candidate_artifacts"].items():
                if manifest[key] != sha256_file(cdir / filename):
                    raise ValueError(f"candidate {dataset} {seed} {filename} differs")
            if dataset == "botsim":
                report = json.loads((cdir / "report.json")
                                    .read_text(encoding="utf-8"))
                if (report["manifest"] != manifest or
                        report["cards_sha256"] !=
                        sha256_file(cdir / "concept_evidence_cards.jsonl.gz")):
                    raise ValueError("BotSim frozen report or evidence card differs")
            if (baseline["manifest"]["model_sha256"] !=
                    sha256_file(bpath.parent / "model.pt")):
                raise ValueError("HypBot checkpoint differs")
            result = compare(candidate, baseline, spec["test_accounts"])
            result["candidate_above_user_threshold"] = bool(
                result["candidate_acc"] > spec["required_acc_strictly_above"])
            result.update({
                "seed": seed,
                "split_sha256": manifest["split_sha256"],
                "candidate_predictions_sha256": sha256_file(cpath),
                "hypbot_predictions_sha256": sha256_file(bpath),
                "hypbot_checkpoint_sha256": baseline["manifest"]["model_sha256"],
            })
            if dataset == "fox8":
                native_path = (root / spec["hypbot_native"] /
                               f"seed{seed}/test_predictions.json")
                native = load_predictions(native_path)
                if (native["manifest"]["model_sha256"] !=
                        sha256_file(native_path.parent / "model.pt")):
                    raise ValueError("native-graph HypBot checkpoint differs")
                result["same_native_graph_hypbot"] = compare(
                    candidate, native, spec["test_accounts"])
            runs.append(result)
        datasets.append({
            "dataset": dataset,
            "candidate_model": spec["candidate"],
            "hypbot_model": spec["hypbot"],
            "required_acc_strictly_above": spec["required_acc_strictly_above"],
            "candidate_mean_acc": float(np.mean([
                item["candidate_acc"] for item in runs])),
            "hypbot_mean_acc": float(np.mean([
                item["hypbot_acc"] for item in runs])),
            "candidate_above_official_hypbot_each_seed": all(
                item["candidate_strictly_above_hypbot"] for item in runs),
            "candidate_above_user_threshold_each_seed": all(
                item["candidate_above_user_threshold"] for item in runs),
            "runs": runs,
        })
        print(dataset, datasets[-1]["candidate_mean_acc"],
              datasets[-1]["hypbot_mean_acc"], flush=True)
    output = {
        "scope": "three_dataset_frozen_evidence_model_paired_numeric_goal",
        "all_three_datasets_above_official_hypbot_each_seed": all(
            item["candidate_above_official_hypbot_each_seed"]
            for item in datasets),
        "all_three_datasets_above_user_acc_threshold_each_seed": all(
            item["candidate_above_user_threshold_each_seed"]
            for item in datasets),
        "fox8_native_graph_comparison_is_separate": True,
        "semantic_evidence_validity_evaluated_here": False,
        "prior_test_population_exploration": True,
        "datasets": datasets,
    }
    path = root / "research/certificates/three_dataset_current_numeric_goal.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
