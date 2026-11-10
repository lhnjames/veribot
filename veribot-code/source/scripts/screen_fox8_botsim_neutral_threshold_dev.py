"""Dev-only check whether existing Fox8/BotSim models can use zero evidence as boundary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from train_fox8_targeted_twenty_ab_group import OUT as FOX8_DIR, source_and_concepts
from audit_botsim_literal_upvote_zero_dev import DATA, NATIVE, USERS
from verifiable_bot.botsim_literal_zero_runtime import BotSimObservedMeanRuntime
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_botsim_neutral_threshold/dev_only.json"
MAX_MEAN_F1_LOSS = 0.005
MAX_SEED_ACC_LOSS = 0.01


def fox8(seed: int) -> dict:
    report = json.loads((FOX8_DIR / f"seed{seed}/dev_report.json")
                        .read_text(encoding="utf-8"))
    model_path = FOX8_DIR / f"seed{seed}/model.joblib"
    reference = LabelBlindReference.load(
        FOX8_DIR / f"seed{seed}/reference.json.gz")
    source = source_and_concepts(seed, reference)
    model = joblib.load(model_path)
    dev = source["split"]["val"]
    x = source["matrix"][dev]
    y = source["labels"][dev]
    p = model.predict_proba(x)[:, 1]
    neutral = float(model.predict_proba(np.zeros((1, x.shape[1])))[:, 1][0])
    old = evaluate(y, p, float(report["threshold"]))
    new = evaluate(y, p, neutral)
    if any(abs(old[key] - report["validation"][key]) > 1e-12
           for key in ("acc", "macro_f1", "auroc")):
        raise ValueError("Fox8 original dev metrics do not replay")
    return {"dataset": "fox8", "seed": seed, "model_sha256":
            sha256_file(model_path), "source_split_sha256":
            report["split_sha256"], "original_threshold":
            float(report["threshold"]), "zero_evidence_probability": neutral,
            "original_validation": old, "neutral_validation": new}


def botsim(seed: int) -> dict:
    runtime = BotSimObservedMeanRuntime(ROOT, seed)
    _, labels, ids, _, split, source = load(
        "botsim", DATA, USERS, seed, NATIVE)
    if ids != runtime.ids:
        raise ValueError("BotSim source population order differs")
    dev = split["val"]
    atomic = runtime.reference.transform(runtime.facts[dev])
    concepts, _ = collapse_group_tails(atomic, runtime.atomic_groups)
    p = runtime.reasoner.predict_proba(concepts)
    neutral = float(expit(runtime.reasoner.intercept))
    old = evaluate(labels[dev], p, runtime.threshold)
    new = evaluate(labels[dev], p, neutral)
    expected = json.loads((ROOT /
        "output/v3_botsim_observed_means_12/dev_only.json")
                          .read_text(encoding="utf-8"))["runs"][seed]
    if (source["split_sha256"] != expected["split_sha256"] or
            any(abs(old[key] - expected["validation"][key]) > 1e-12
                for key in ("acc", "macro_f1", "auroc"))):
        raise ValueError("BotSim original dev metrics do not replay")
    return {"dataset": "botsim", "seed": seed,
            "reasoner_sha256": runtime.source_binding["reasoner_sha256"],
            "source_split_sha256": source["split_sha256"],
            "original_threshold": runtime.threshold,
            "zero_evidence_probability": neutral,
            "original_validation": old, "neutral_validation": new}


def main() -> None:
    rows = {dataset: [function(seed) for seed in (0, 1, 2)]
            for dataset, function in (("botsim", botsim), ("fox8", fox8))}
    gates = {}
    for dataset, runs in rows.items():
        mean_loss = float(np.mean([
            x["original_validation"]["macro_f1"] -
            x["neutral_validation"]["macro_f1"] for x in runs]))
        max_acc_loss = float(max(
            x["original_validation"]["acc"] -
            x["neutral_validation"]["acc"] for x in runs))
        gates[dataset] = {
            "mean_macro_f1_loss": mean_loss,
            "max_seed_acc_loss": max_acc_loss,
            "passes_predeclared_gate":
                mean_loss <= MAX_MEAN_F1_LOSS and
                max_acc_loss <= MAX_SEED_ACC_LOSS}
        print(dataset, [(x["seed"],
                          round(x["original_validation"]["acc"], 5),
                          round(x["neutral_validation"]["acc"], 5),
                          round(x["zero_evidence_probability"], 5))
                         for x in runs], gates[dataset], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "existing_Fox8_20_concept_and_BotSim_12_concept_neutral_boundary_dev_only",
        "test_predictions_or_metrics_computed": False,
        "threshold_is_model_probability_at_all_zero_encoded_evidence": True,
        "predeclared_gate": {
            "mean_macro_f1_loss_at_most": MAX_MEAN_F1_LOSS,
            "each_seed_acc_loss_at_most": MAX_SEED_ACC_LOSS},
        "runs": rows, "gates": gates}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
