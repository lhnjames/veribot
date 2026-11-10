"""Locate which frozen MGTAB evidence group carries reference-mixture drift."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load
from audit_mgtab_primary_reference_mixture_dev import (
    CAP, ROOT, encode, predict)
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


OUT = ROOT / "output/v3_mgtab_primary_reference_mixture/group_sensitivity_dev.json"
FRACTIONS = (0.0, .75)
GROUPS = {
    "published_profile": np.r_[0:20, 150:170],
    "published_native_relations": np.r_[20:86, 170:236],
    "published_tweet_projection": np.r_[86:150, 236:300],
}


def main() -> None:
    prior = json.loads((ROOT / "output/v3_mgtab_primary_reference_mixture/dev_only.json")
                       .read_text(encoding="utf-8"))
    if (prior["scope"] != "frozen_high_acc_mgtab_reference_mixture_dev_only"
            or prior["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("Prior reference diagnostic differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] !=
                prior["runs"][seed]["split_sha256"]):
            raise ValueError("MGTAB frozen population differs")
        train, dev = split["train"], split["val"]
        original_x = encode(runtime, runtime.reference, dev)
        original_p = predict(runtime, original_x)
        original_class = original_p >= runtime.threshold
        human = train[labels[train] == 0]
        bot = train[labels[train] == 1]
        scenarios = []
        for fraction in FRACTIONS:
            bot_n = round(1500*fraction)
            human_n = 1500 - bot_n
            rng = np.random.default_rng(20261103 + 1000*seed)
            selected = np.r_[rng.choice(human, human_n, replace=False),
                             rng.choice(bot, bot_n, replace=False)]
            reference = LabelBlindReference(min_context_n=30).fit(
                runtime.facts[selected], runtime.names)
            x_new = encode(runtime, reference, dev)
            report = {}
            for group, columns in GROUPS.items():
                mixed = original_x.copy()
                mixed[:, columns] = x_new[:, columns]
                p = predict(runtime, mixed)
                report[group] = {
                    "validation": evaluate(labels[dev], p, runtime.threshold),
                    "class_changes_vs_original": int(np.count_nonzero(
                        (p >= runtime.threshold) != original_class)),
                    "mean_absolute_probability_change": float(np.mean(
                        np.abs(p - original_p))),
                    "mean_absolute_encoded_group_change": float(np.mean(
                        np.abs(x_new[:, columns] - original_x[:, columns]))),
                }
            all_p = predict(runtime, x_new)
            matching = next(r for r in prior["runs"][seed]["diagnostics"]
                            if r["reference_bot_fraction"] == fraction
                            and r["replicate"] == 0)
            if (abs(evaluate(labels[dev], all_p, runtime.threshold)["acc"] -
                    matching["validation"]["acc"]) > 1e-12
                    or reference.version != matching["reference_version"]):
                raise ValueError("Group sensitivity did not replay full mixture")
            scenarios.append({
                "reference_bot_fraction": fraction,
                "full_replacement_validation": matching["validation"],
                "single_group_replacements": report,
            })
            print({"seed": seed, "bot_fraction": fraction,
                   "full_acc": matching["validation"]["acc"],
                   "single_group_acc": {k: v["validation"]["acc"]
                                        for k, v in report.items()}}, flush=True)
        runs.append({"seed": seed, "scenarios": scenarios})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "dev_only_single_source_group_reference_replacement_diagnostic",
        "model_weights_and_threshold_frozen": True,
        "reference_selection_uses_train_labels_only_for_diagnostic": True,
        "mixed_group_interventions_are_not_natural_accounts": True,
        "test_predictions_or_metrics_accessed": False,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
