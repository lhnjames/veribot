"""Dev-only reference-mixture sensitivity of the frozen high-ACC MGTAB model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_primary_reference_mixture/dev_only.json"
FRACTIONS = (0.0, .25, .5, .75)
REFERENCE_SIZE = 1500
REPLICATES = 3
CAP = 3.0


def encode(runtime: MGTABFullAtomicRuntime,
           reference: LabelBlindReference, rows: np.ndarray) -> np.ndarray:
    batch = reference.transform(runtime.facts[rows])
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[rows] / (
        runtime.supports[rows] + 5.0)
    x = np.column_stack((np.log1p(batch.upper) * reliability,
                         np.log1p(batch.lower) * reliability))
    if x.shape != (len(rows), 300) or not np.isfinite(x).all():
        raise ValueError("MGTAB evidence encoding differs")
    return x


def predict(runtime: MGTABFullAtomicRuntime, x: np.ndarray) -> np.ndarray:
    full = np.asarray(runtime.model.booster_.predict(
        x, raw_score=True), dtype=float)
    named_x = x.copy()
    named_x[:, 86:150] = 0.0
    named_x[:, 236:300] = 0.0
    named = np.asarray(runtime.model.booster_.predict(
        named_x, raw_score=True), dtype=float)
    return expit(named + np.clip(full - named, -CAP, CAP))


def main() -> None:
    cap_lock_path = (ROOT /
        "output/v3_mgtab_explicit_bounded_projection_residual/selection_lock.json")
    cap_lock = json.loads(cap_lock_path.read_text(encoding="utf-8"))
    if cap_lock["selected_cap"] != CAP:
        raise ValueError("Current primary residual bound differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] !=
                runtime.lock["split_sha256_by_seed"][str(seed)]
                or cap_lock["models"][seed]["model_sha256"] !=
                sha256_file(runtime.run_dir / "model.joblib")):
            raise ValueError("MGTAB source, split, or frozen model differs")
        train, dev = split["train"], split["val"]
        baseline_x = encode(runtime, runtime.reference, dev)
        baseline_p = predict(runtime, baseline_x)
        baseline_pred = baseline_p >= runtime.threshold
        human = train[labels[train] == 0]
        bot = train[labels[train] == 1]
        rows = []
        for fraction in FRACTIONS:
            bot_n = round(REFERENCE_SIZE * fraction)
            human_n = REFERENCE_SIZE - bot_n
            if bot_n > len(bot) or human_n > len(human):
                raise ValueError("Reference mixture exceeds available train labels")
            for replicate in range(REPLICATES):
                rng = np.random.default_rng(20261103 + 1000*seed + 100*replicate)
                chosen = np.r_[rng.choice(human, human_n, replace=False),
                               rng.choice(bot, bot_n, replace=False)]
                reference = LabelBlindReference(min_context_n=30).fit(
                    runtime.facts[chosen], runtime.names)
                x = encode(runtime, reference, dev)
                p = predict(runtime, x)
                pred = p >= runtime.threshold
                row = {
                    "reference_bot_fraction": fraction,
                    "replicate": replicate,
                    "reference_version": reference.version,
                    "validation": evaluate(labels[dev], p, runtime.threshold),
                    "class_changes_vs_all_train_reference": int(np.count_nonzero(
                        pred != baseline_pred)),
                    "mean_absolute_probability_change": float(np.mean(np.abs(
                        p - baseline_p))),
                    "mean_absolute_encoded_evidence_change": float(np.mean(
                        np.abs(x - baseline_x))),
                    "only_new_reference_correct": int(np.count_nonzero(
                        (pred == labels[dev]) & (baseline_pred != labels[dev]))),
                    "only_original_reference_correct": int(np.count_nonzero(
                        (pred != labels[dev]) & (baseline_pred == labels[dev]))),
                }
                rows.append(row)
                print({"seed": seed, "bot_fraction": fraction,
                       "replicate": replicate,
                       "dev_acc": row["validation"]["acc"],
                       "class_changes": row["class_changes_vs_all_train_reference"]},
                      flush=True)
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "original_reference_version": runtime.reference.version,
            "original_train_bot_fraction": float(np.mean(labels[train] == 1)),
            "original_all_train_validation": evaluate(
                labels[dev], baseline_p, runtime.threshold),
            "original_validation_threshold": runtime.threshold,
            "diagnostics": rows,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "frozen_high_acc_mgtab_reference_mixture_dev_only",
        "model_weights_frozen": True,
        "decision_thresholds_frozen": True,
        "diagnostic_reference_selection_uses_train_labels": True,
        "deployed_reference_uses_labels": False,
        "test_predictions_or_metrics_accessed": False,
        "reference_size": REFERENCE_SIZE,
        "replicates_per_fraction": REPLICATES,
        "reference_bot_fractions": FRACTIONS,
        "cap_lock_sha256": sha256_file(cap_lock_path),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
