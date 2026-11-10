"""Dev-only retraining sensitivity to the MGTAB reference population mixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from audit_mgtab_primary_reference_mixture_dev import CAP, ROOT, encode
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


OUT = ROOT / "output/v3_mgtab_primary_reference_mixture/retrained_dev_only.json"
FRACTIONS = (0.0, .25, .5, .75)
REFERENCE_SIZE = 1500


def probability(model: LGBMClassifier, x: np.ndarray) -> np.ndarray:
    full = np.asarray(model.booster_.predict(x, raw_score=True), dtype=float)
    named_x = x.copy()
    named_x[:, 86:150] = 0.0
    named_x[:, 236:300] = 0.0
    named = np.asarray(model.booster_.predict(named_x, raw_score=True), dtype=float)
    return expit(named + np.clip(full - named, -CAP, CAP))


def fit_same_capacity(runtime: MGTABFullAtomicRuntime,
                      x_a: np.ndarray, y_a: np.ndarray,
                      x_b: np.ndarray, y_b: np.ndarray) -> LGBMClassifier:
    signs = directions(x_a, y_a)
    groups = [members + [j + 150 for j in members]
              for members in runtime.group_fact_indices.values()]
    params = runtime.model.get_params().copy()
    params.update(n_jobs=4, monotone_constraints=signs.tolist(),
                  interaction_constraints=groups)
    model = LGBMClassifier(**params)
    model.fit(x_b, y_b)
    single, _ = all_trees_one_group(model, groups)
    if not single:
        raise ValueError("Retrained model crossed source groups")
    return model


def main() -> None:
    prior_path = ROOT / "output/v3_mgtab_primary_reference_mixture/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior["scope"] != "frozen_high_acc_mgtab_reference_mixture_dev_only"
            or prior["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("Reference diagnostic differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] !=
                prior["runs"][seed]["split_sha256"]):
            raise ValueError("MGTAB source or split differs")
        contract_path = runtime.run_dir / "ab_direction_contract.npz"
        with np.load(contract_path, allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
        train, dev = split["train"], split["val"]
        if len(np.intersect1d(a, b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(train)):
            raise ValueError("A/B partition differs from official train")
        x_original_a = encode(runtime, runtime.reference, a)
        x_original_b = encode(runtime, runtime.reference, b)
        x_original_dev = encode(runtime, runtime.reference, dev)
        replay = fit_same_capacity(runtime, x_original_a, labels[a],
                                   x_original_b, labels[b])
        saved_p = probability(runtime.model, x_original_dev)
        replay_p = probability(replay, x_original_dev)
        max_replay = float(np.max(np.abs(saved_p - replay_p)))
        if max_replay > 1e-10:
            raise ValueError(f"Original model retraining did not replay: {max_replay}")
        human = train[labels[train] == 0]
        bot = train[labels[train] == 1]
        rows = []
        for fraction in FRACTIONS:
            bot_n = round(REFERENCE_SIZE*fraction)
            human_n = REFERENCE_SIZE - bot_n
            rng = np.random.default_rng(20261103 + 1000*seed)
            selected = np.r_[rng.choice(human, human_n, replace=False),
                             rng.choice(bot, bot_n, replace=False)]
            reference = LabelBlindReference(min_context_n=30).fit(
                runtime.facts[selected], runtime.names)
            x_a = encode(runtime, reference, a)
            x_b = encode(runtime, reference, b)
            x_dev = encode(runtime, reference, dev)
            model = fit_same_capacity(runtime, x_a, labels[a], x_b, labels[b])
            p = probability(model, x_dev)
            threshold = best_validation_threshold(p, labels[dev])
            row = {
                "reference_bot_fraction": fraction,
                "reference_version": reference.version,
                "fixed_original_threshold_validation": evaluate(
                    labels[dev], p, runtime.threshold),
                "reselected_dev_threshold_validation": evaluate(
                    labels[dev], p, threshold),
                "reselected_dev_threshold": threshold,
                "class_changes_vs_original_at_original_threshold": int(
                    np.count_nonzero((p >= runtime.threshold) !=
                                     (saved_p >= runtime.threshold))),
            }
            rows.append(row)
            print({"seed": seed, "bot_fraction": fraction,
                   "fixed_threshold_acc": row["fixed_original_threshold_validation"]["acc"],
                   "reselected_threshold_acc": row["reselected_dev_threshold_validation"]["acc"]},
                  flush=True)
        runs.append({
            "seed": seed,
            "split_sha256": source["split_sha256"],
            "original_weight_retrain_max_probability_difference": max_replay,
            "original_reference_version": runtime.reference.version,
            "original_all_train_validation": evaluate(
                labels[dev], saved_p, runtime.threshold),
            "diagnostics": rows,
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_same_capacity_ab_group_tree_retrained_reference_mixture_dev_only",
        "source_diagnostic_sha256": sha256_file(prior_path),
        "reference_selection_uses_train_labels_only_to_construct_diagnostic_mixtures": True,
        "A_direction_and_B_fit_remain_disjoint": True,
        "test_predictions_or_metrics_accessed": False,
        "reference_size": REFERENCE_SIZE,
        "fractions": FRACTIONS,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
