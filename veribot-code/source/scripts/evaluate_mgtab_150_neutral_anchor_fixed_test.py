"""One fixed-test evaluation of the dev-locked neutral-anchor candidate."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_ab_monotone_evidence_tree_dev import evidence_matrix
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference

from screen_mgtab_150_neutral_anchor_dev import ROOT, FROZEN, SOURCE


DEST = ROOT / "output/v3_mgtab_150_neutral_anchor"
BASE_CERT = ROOT / "research/certificates/three_dataset_observed_means_numeric_goal.json"


def main() -> None:
    lock_path = DEST / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    dev_path = DEST / "dev_only.json"
    if (lock["dev_result_sha256"] != sha256_file(dev_path) or
            lock["test_predictions_or_metrics_used_for_selection"] is not False):
        raise ValueError("Selection was not bound to dev-only record")
    baseline = json.loads(BASE_CERT.read_text(encoding="utf-8"))
    mgtab = next(x for x in baseline["datasets"] if x["dataset"] == "mgtab")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=SOURCE)
    results = []
    for seed in (0, 1, 2):
        run = lock["runs"][seed]
        old = json.loads((SOURCE / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if (run["seed"] != seed or
                run["split_sha256"] != old["manifest"]["split_sha256"] or
                run["split_sha256"] != mgtab["runs"][seed]["split_sha256"] or
                lock["published_source_features_sha256"] != sha256_file(
                    args.data_dir / "features.pt")):
            raise ValueError("Test population or source data differs")
        model_path = Path(run["model_path"])
        reference_path = Path(run["reference_path"])
        if sha256_file(model_path) != run["model_sha256"]:
            raise ValueError("Locked model weights changed")
        model = joblib.load(model_path)
        reference = LabelBlindReference.load(reference_path)
        if reference.version != run["reference_version"]:
            raise ValueError("Locked train-only reference changed")
        test = split["test"]
        design = evidence_matrix(transform_with_support(
            reference, facts[test], split["supports"][test]))
        zero = float(model.predict_proba(
            np.zeros((1, design.shape[1])))[:, 1][0])
        if abs(zero - run["zero_evidence_threshold"]) > 1e-12:
            raise ValueError("Neutral decision threshold changed")
        probabilities = model.predict_proba(design)[:, 1]
        prediction = probabilities >= zero
        if np.any(np.abs(probabilities - zero) <= 1e-10):
            raise ValueError("Real account lies on unsupported neutral boundary")
        metrics = evaluate(labels[test], probabilities, zero)
        correct = int(np.count_nonzero(prediction == labels[test]))
        hypbot_correct = int(mgtab["runs"][seed]["hypbot_correct"])
        output_path = DEST / f"seed{seed}/fixed_test_predictions.npz"
        np.savez_compressed(output_path, account_ids=np.asarray(test),
                            probability=probabilities,
                            predicted_bot=prediction,
                            threshold=np.asarray(zero))
        results.append({
            "seed": seed, "accounts": len(test),
            "correct": correct, "metrics": metrics,
            "hypbot_correct_on_same_test_population": hypbot_correct,
            "strictly_above_hypbot": correct > hypbot_correct,
            "strictly_above_91_2_percent_acc": metrics["acc"] > 0.912,
            "score_direction_supported_by_nonzero_input": int(len(test)),
            "predicted_bot": int(prediction.sum()),
            "predicted_human": int((~prediction).sum()),
            "neutral_probability": zero,
            "predictions_path": str(output_path),
            "predictions_sha256": sha256_file(output_path),
            "model_sha256": run["model_sha256"],
            "split_sha256": run["split_sha256"],
        })
        print({"seed": seed, "correct": correct,
               "hypbot_correct": hypbot_correct,
               "acc": metrics["acc"]}, flush=True)
    (DEST / "fixed_test_report.json").write_text(json.dumps({
        "scope": "dev_locked_MGTAB_150_fact_neutral_anchor_fixed_test",
        "selection_lock_sha256": sha256_file(lock_path),
        "prior_numeric_certificate_sha256": sha256_file(BASE_CERT),
        "test_population_previously_explored_by_other_candidates": True,
        "all_seeds_above_hypbot": all(x["strictly_above_hypbot"]
                                      for x in results),
        "all_seeds_above_91_2_percent": all(
            x["strictly_above_91_2_percent_acc"] for x in results),
        "runs": results}, indent=2) + "\n", encoding="utf-8")
    print(DEST / "fixed_test_report.json")


if __name__ == "__main__":
    main()
