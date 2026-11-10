"""Export a source-bound decision card for every fixed-test account."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file
from predict_best_evidence_models import MODELS
from predict_best_verified_evidence import render_selected_decision
from predict_botsim_observed_means_from_sources import predict as predict_botsim
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
SAVED = {
    "botsim": "output/v3_botsim_observed_means_12",
    "fox8": "output/v3_fox8_neighbor_atomic_evidence_capacity",
    "mgtab": "output/v3_mgtab_ab_group_additive_tree",
}
OUT = ROOT / "output/v3_best_verified_evidence_full_test"
CERTIFICATE = ROOT / "research/certificates/best_verified_evidence_full_test.json"


def _source_files_exist(card: dict) -> int:
    present = 0
    for term in card["evidence_terms"]:
        query = term["source_query"]
        if query is None:
            if term["source_fact_observed"]:
                raise ValueError("observed term has no source query")
            continue
        for source in query["source_ids"]:
            file = Path(str(source).split("#", 1)[0])
            if not file.is_file():
                raise FileNotFoundError(file)
            present += 1
    return present


def export_run(dataset: str, seed: int) -> dict:
    recorded_path = ROOT / SAVED[dataset] / f"seed{seed}/test_predictions.json"
    recorded = json.loads(recorded_path.read_text(encoding="utf-8"))["predictions"]
    ids = [str(row["account_id"]) for row in recorded]
    if len(ids) != len(set(ids)):
        raise ValueError("test account IDs repeat")
    if dataset == "botsim":
        sourced = predict_botsim(ROOT, dataset, seed, ids)
        decisions = sourced["decisions"]
        if [str(row["account_id"]) for row in decisions] != ids:
            raise ValueError("BotSim source batch changed test account order")
        source_binding = sourced["source_binding"]
        runtime = None
    else:
        runtime = (Fox8NeighborAtomicRuntime(ROOT, seed) if dataset == "fox8"
                   else MGTABFullAtomicRuntime(ROOT, seed,
                                               variant="ab_group_additive"))
        if not set(ids) <= set(runtime.index):
            raise ValueError("test account absent from source population")
        decisions = None
        source_binding = {"inference_manifest": str(
            ROOT / SAVED[dataset] / "inference_manifest.json")}
    output = OUT / f"{dataset}_seed{seed}.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": dataset, "seed": seed, "accounts": len(ids),
        "evidence_terms": 0, "source_file_references": 0,
        "published_averaged_tweet_projection_terms": 0,
        "all_evidence_changes_class": 0,
        "max_probability_difference": 0.0,
        "max_logit_reconstruction_difference": 0.0,
        "class_mismatches": 0,
        "saved_predictions_sha256": sha256_file(recorded_path),
    }
    with output.open("wb") as sink, gzip.GzipFile(
            filename="", mode="wb", fileobj=sink, mtime=0) as zipped:
        for index, (uid, saved) in enumerate(zip(ids, recorded, strict=True)):
            decision = decisions[index] if decisions is not None else runtime.predict(uid)
            if runtime is not None:
                decision["threshold"] = runtime.threshold
            raw = {
                "dataset": dataset, "seed": seed, "account_id": uid,
                "model_id": MODELS[dataset][0],
                "decision": decision,
                "evidence_term_count": MODELS[dataset][1],
                "source_binding": source_binding,
                "source_scope": (
                    "published tensors and typed relation edges; tweet content is an average tensor"
                    if dataset == "mgtab" else
                    "published account, sampled posts and native relation records"),
            }
            card = render_selected_decision(ROOT, dataset, seed, uid, raw)
            difference = abs(float(saved["probability"]) - card["probability"])
            summary["max_probability_difference"] = max(
                summary["max_probability_difference"], difference)
            summary["max_logit_reconstruction_difference"] = max(
                summary["max_logit_reconstruction_difference"],
                abs(card["reconstruction_difference"]))
            summary["class_mismatches"] += int(
                int(float(saved["probability"]) >= card["threshold"])
                != card["prediction"])
            if difference > 1e-12 or summary["class_mismatches"]:
                raise ValueError("full source card differs from frozen test result")
            summary["evidence_terms"] += len(card["evidence_terms"])
            summary["source_file_references"] += _source_files_exist(card)
            summary["published_averaged_tweet_projection_terms"] += sum(
                item["source_level"] ==
                "published_averaged_tweet_tensor_train_only_projection"
                for item in card["evidence_terms"])
            summary["all_evidence_changes_class"] += int(
                card["exact_frozen_model_dependency"][
                    "all_evidence_changes_class"])
            zipped.write((json.dumps(card, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False)
                          + "\n").encode("utf-8"))
    if (summary["evidence_terms"] != len(ids) * MODELS[dataset][1]
            or not math.isfinite(summary["max_probability_difference"])):
        raise ValueError("full evidence-card population differs")
    summary["cards_file"] = str(output)
    summary["cards_sha256"] = sha256_file(output)
    summary["cards_bytes"] = output.stat().st_size
    print({key: summary[key] for key in ("dataset", "seed", "accounts",
          "evidence_terms", "max_probability_difference")}, flush=True)
    return summary


def main() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(MODELS))
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else list(MODELS)
    runs = [export_run(dataset, seed) for dataset in datasets
            for seed in (0, 1, 2)]
    if args.dataset:
        if not CERTIFICATE.is_file():
            return
        prior = json.loads(CERTIFICATE.read_text(encoding="utf-8"))
        old = {(run["dataset"], run["seed"]): run for run in prior["runs"]}
        old.update({(run["dataset"], run["seed"]): run for run in runs})
        if len(old) != 9:
            raise ValueError("partial refresh cannot reconstruct nine runs")
        runs = [old[dataset, seed] for dataset in MODELS for seed in (0, 1, 2)]
    CERTIFICATE.parent.mkdir(parents=True, exist_ok=True)
    CERTIFICATE.write_text(json.dumps({
        "status": "all_selected_fixed_test_accounts_have_source_bound_evidence_cards",
        "scope": "nine_fixed_test_runs_5400_account_decisions",
        "exporter_sha256": sha256_file(Path(__file__)),
        "card_entry_sha256": sha256_file(ROOT /
            "source/scripts/predict_best_verified_evidence.py"),
        "total_accounts": sum(row["accounts"] for row in runs),
        "total_evidence_terms": sum(row["evidence_terms"] for row in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(CERTIFICATE)


if __name__ == "__main__":
    main()
