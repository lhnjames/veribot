"""Development-only audit of named evidence versus a 16-coefficient residual."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from screen_mgtab_twenty_named_ratio_struct_neighbor_dev import fixed_groups
from screen_mgtab_twenty_named_bounded_linear_residual_dev import fit_named, rows
from screen_mgtab_supervised_bounded_linear_residual_dev import bounded, fit
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "output/v3_mgtab_twenty_named_ratio_struct_neighbor/dev_only.json"
OUT = ROOT / "output/gate1_mgtab_low_capacity_residual_dominance/dev_only.json"
RATIOS = (0.0, .25, .5, 1.0, 2.0, float("inf"))


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {str(q): float(np.quantile(values, q))
            for q in (0, .25, .5, .75, .9, .95, .99, 1)}


def main() -> None:
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if prior["group_scheme"] != "profile_ratio_singleton_structural_neighbor_six_other_events_one":
        raise ValueError("prior named evidence scheme differs")
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, y, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != \
                prior["runs"][seed]["split_sha256"]:
            raise ValueError("official account population or split differs")
        contract = runtime.run_dir / "ab_direction_contract.npz"
        with np.load(contract, allow_pickle=False) as saved:
            a = saved["a_indices"].astype(int)
            b = saved["b_indices"].astype(int)
        if len(np.intersect1d(a, b)) or not np.array_equal(
                np.sort(np.r_[a, b]), np.sort(split["train"])):
            raise ValueError("A/B partition differs")
        groups = fixed_groups(runtime)
        x_a, content_a = rows(runtime, a, groups)
        x_b, _ = rows(runtime, b, groups)
        dev = split["val"]
        x_dev, content_dev = rows(runtime, dev, groups)
        named_model = fit_named(runtime, x_a, y[a], x_b, y[b])
        named_a = np.asarray(named_model.booster_.predict(x_a, raw_score=True))
        named_dev = np.asarray(named_model.booster_.predict(x_dev, raw_score=True))
        zero_logit = float(named_model.booster_.predict(
            np.zeros((1, x_dev.shape[1])), raw_score=True)[0])
        weights, optimization = fit(content_a, named_a, y[a], 0.0)
        if len(weights) != 16:
            raise ValueError("low-capacity residual coefficient count differs")
        residual = bounded(content_dev, weights)
        original_choice = next(item for item in prior["runs"][seed]["candidates"]
                               if item["residual_fit_half"] ==
                               "A_out_of_sample_named_score" and item["l2"] == 0.0)
        threshold = float(original_choice["threshold"])
        full_metric = evaluate(y[dev], expit(named_dev + residual), threshold)
        if (abs(full_metric["macro_f1"] - original_choice["validation"]["macro_f1"])
                > 1e-9 or full_metric["errors"] !=
                original_choice["validation"]["errors"]):
            raise ValueError("prior development result does not replay")
        named_effect = named_dev - zero_logit
        share = np.abs(residual) / np.maximum(
            np.abs(named_effect) + np.abs(residual), 1e-12)
        named_class = expit(named_dev) >= threshold
        full_class = expit(named_dev + residual) >= threshold
        budgets = []
        for ratio in RATIOS:
            constrained = (residual if np.isinf(ratio) else
                           np.sign(residual) * np.minimum(
                               np.abs(residual), ratio * np.abs(named_effect)))
            prediction = expit(named_dev + constrained)
            budgets.append({
                "residual_to_named_effect_cap": "unbounded" if np.isinf(ratio) else ratio,
                "development": evaluate(y[dev], prediction, threshold),
                "class_changes_from_named_only": int(np.count_nonzero(
                    (prediction >= threshold) != named_class)),
                "accounts_residual_clipped": int(np.count_nonzero(
                    np.abs(constrained - residual) > 1e-12)),
            })
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "named_model_schema": "ten broad named groups, two tail concepts each",
            "residual_input_columns": 16, "residual_trainable_coefficients": 16,
            "residual_form": "3*tanh(linear_projection_tail_scores/3)",
            "residual_optimization": optimization,
            "zero_evidence_named_logit": zero_logit,
            "threshold_from_prior_dev_candidate": threshold,
            "named_effect_absolute_quantiles": _quantiles(np.abs(named_effect)),
            "residual_absolute_quantiles": _quantiles(np.abs(residual)),
            "residual_absolute_effect_share_quantiles": _quantiles(share),
            "accounts_residual_share_above_half": int(np.count_nonzero(share > .5)),
            "accounts_residual_changes_class": int(np.count_nonzero(named_class != full_class)),
            "budgets": budgets,
        })
        print({"seed": seed,
               "share_gt_half": runs[-1]["accounts_residual_share_above_half"],
               "class_changes": runs[-1]["accounts_residual_changes_class"],
               "f1_unbounded": round(full_metric["macro_f1"], 5)}, flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "MGTAB 20 grouped named concepts plus 16-coefficient bounded residual, dev-only dominance audit",
        "prior_dev_report_sha256": sha256_file(PRIOR),
        "test_predictions_or_metrics_accessed": False,
        "fixed_thresholds_reused_from_prior_development_report": True,
        "budget_ratios_are_diagnostic_not_model_selection": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
