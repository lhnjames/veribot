"""Lock a projection-residual cap on validation, then audit fixed test."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_mgtab_explicit_bounded_projection_residual"
SOURCE = ROOT / "output/v3_mgtab_ab_group_additive_tree"
CAP_GRID = (0.0, 1.0, 2.0, 3.0, 4.0)


def scores(runtime: MGTABFullAtomicRuntime, rows: np.ndarray
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = evidence_design(runtime, rows)
    full = np.asarray(runtime.model.booster_.predict(x, raw_score=True),
                      dtype=float)
    named_x = x.copy()
    named_x[:, 86:150] = 0.0
    named_x[:, 236:300] = 0.0
    named = np.asarray(runtime.model.booster_.predict(
        named_x, raw_score=True), dtype=float)
    return full, named, full - named


def metrics(y: np.ndarray, logit: np.ndarray, threshold: float) -> dict:
    probability = expit(logit)
    prediction = probability >= threshold
    return {
        "n": len(y),
        "acc": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "auroc": float(roc_auc_score(y, probability)),
        "errors": int(np.count_nonzero(prediction != y)),
    }


def bounded_group_share(runtime: MGTABFullAtomicRuntime,
                        rows: np.ndarray, named: np.ndarray,
                        residual: np.ndarray, cap: float) -> np.ndarray:
    x = evidence_design(runtime, rows)
    x[:, 86:150] = 0.0
    x[:, 236:300] = 0.0
    named_absolute = np.zeros(len(rows), dtype=float)
    for name, members in runtime.group_fact_indices.items():
        if name == "published_tweet_projections":
            continue
        masked = x.copy()
        masked[:, members] = 0.0
        masked[:, np.asarray(members) + 150] = 0.0
        without = np.asarray(runtime.model.booster_.predict(
            masked, raw_score=True), dtype=float)
        named_absolute += np.abs(named - without)
    bounded_absolute = np.abs(np.clip(residual, -cap, cap))
    denominator = named_absolute + bounded_absolute
    return np.divide(bounded_absolute, denominator,
                     out=np.zeros_like(denominator), where=denominator > 0)


def run_seed(seed: int, stage: str, cap: float | None = None) -> dict:
    runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
    _, labels, ids, _, split, source = load(
        "mgtab", runtime.data_dir, None, seed)
    rows = split["val" if stage == "dev" else "test"]
    full, named, residual = scores(runtime, rows)
    if len(rows) != (2040 if stage == "dev" else 1020):
        raise ValueError("official split size differs")
    original_class = expit(full) >= runtime.threshold
    result = {
        "seed": seed,
        "split_sha256": source["split_sha256"],
        "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
        "reference_version": runtime.reference.version,
        "threshold": runtime.threshold,
        "residual_absolute_quantiles": {
            str(q): float(np.quantile(np.abs(residual), q))
            for q in (0, .25, .5, .75, .9, .95, .99, 1)
        },
        "named_only": metrics(labels[rows], named, runtime.threshold),
        "original": metrics(labels[rows], full, runtime.threshold),
        "named_only_class_changes_from_original": int(np.count_nonzero(
            (expit(named) >= runtime.threshold) != original_class)),
    }
    if stage == "dev":
        result["cap_candidates"] = []
        for candidate in CAP_GRID:
            bounded = named + np.clip(residual, -candidate, candidate)
            result["cap_candidates"].append({
                "cap": candidate,
                "metrics": metrics(labels[rows], bounded, runtime.threshold),
                "class_changes_from_original": int(np.count_nonzero(
                    (expit(bounded) >= runtime.threshold) != original_class)),
                "accounts_clipped": int(np.count_nonzero(
                    np.abs(residual) > candidate)),
            })
    else:
        if cap is None:
            raise ValueError("test stage needs validation-locked cap")
        saved_path = SOURCE / f"seed{seed}/test_predictions.json"
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        frozen = saved["predictions"]
        if (len(frozen) != len(rows) or
                source["split_sha256"] != saved["manifest"]["split_sha256"] or
                [entry["account_id"] for entry in frozen] !=
                [ids[int(i)] for i in rows] or
                [entry["label"] for entry in frozen] != labels[rows].tolist()):
            raise ValueError("fixed test population differs")
        replay_error = float(np.max(np.abs(expit(full) - np.asarray(
            [entry["probability"] for entry in frozen]))))
        if replay_error > 1e-12:
            raise ValueError("original frozen test predictions do not replay")
        bounded = named + np.clip(residual, -cap, cap)
        share = bounded_group_share(runtime, rows, named, residual, cap)
        named_class = expit(named) >= runtime.threshold
        bounded_class = expit(bounded) >= runtime.threshold
        high_share = share > .5
        result.update({
            "cap": cap,
            "saved_predictions_sha256": sha256_file(saved_path),
            "max_original_probability_replay_error": replay_error,
            "bounded": metrics(labels[rows], bounded, runtime.threshold),
            "bounded_class_changes_from_original": int(np.count_nonzero(
                (expit(bounded) >= runtime.threshold) != original_class)),
            "accounts_with_residual_clipped": int(np.count_nonzero(
                np.abs(residual) > cap)),
            "accounts_where_projection_changes_original_class": int(
                np.count_nonzero((expit(named) >= runtime.threshold) !=
                                 original_class)),
            "residual_absolute_effect_share_median": float(np.median(share)),
            "residual_absolute_effect_share_p90": float(np.quantile(share, .9)),
            "residual_share_above_50_percent_accounts": int(high_share.sum()),
            "residual_share_above_50_percent_class_changes": int(
                np.count_nonzero((named_class != bounded_class) & high_share)),
        })
        OUT.mkdir(parents=True, exist_ok=True)
        artifact = OUT / f"seed{seed}_test_effects.npz"
        np.savez_compressed(
            artifact, ids=np.asarray([ids[int(i)] for i in rows]),
            labels=labels[rows], full_logit=full, named_logit=named,
            projection_residual=residual, bounded_logit=bounded,
            residual_absolute_effect_share=share,
            residual_changes_class=(named_class != bounded_class),
            threshold=np.asarray(runtime.threshold),
            cap=np.asarray(cap))
        result["effect_matrix_path"] = str(artifact.relative_to(ROOT)).replace(
            "\\", "/")
        result["effect_matrix_sha256"] = sha256_file(artifact)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "test"), required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "dev":
        runs = [run_seed(seed, "dev") for seed in (0, 1, 2)]
        selected = next((cap for cap in CAP_GRID if all(
            next(row for row in run["cap_candidates"] if row["cap"] == cap)[
                "class_changes_from_original"] == 0 for run in runs)), None)
        if selected is None:
            raise ValueError("no shared cap preserves original validation decisions")
        report = {
            "status": "validation_only_bounded_projection_screen",
            "test_predictions_or_metrics_computed": False,
            "cap_grid": CAP_GRID,
            "selection_rule": "smallest shared cap preserving every original validation class",
            "selected_cap": selected,
            "runs": runs,
        }
        dev_path = OUT / "dev_only.json"
        dev_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        lock = {
            "status": "fixed_before_bounded_residual_test_evaluation",
            "selected_cap": selected,
            "dev_report_sha256": sha256_file(dev_path),
            "models": [{key: run[key] for key in (
                "seed", "split_sha256", "model_sha256", "reference_version",
                "threshold")} for run in runs],
        }
        (OUT / "selection_lock.json").write_text(
            json.dumps(lock, indent=2) + "\n", encoding="utf-8")
        print({"stage": "dev", "selected_cap": selected,
               "changes": [next(row for row in run["cap_candidates"]
                                if row["cap"] == selected)[
                                    "class_changes_from_original"]
                           for run in runs]})
    else:
        dev_path = OUT / "dev_only.json"
        lock = json.loads((OUT / "selection_lock.json").read_text(
            encoding="utf-8"))
        if (lock["status"] != "fixed_before_bounded_residual_test_evaluation"
                or lock["dev_report_sha256"] != sha256_file(dev_path)):
            raise ValueError("validation cap lock differs")
        cap = float(lock["selected_cap"])
        runs = [run_seed(seed, "test", cap) for seed in (0, 1, 2)]
        for run, expected in zip(runs, lock["models"], strict=True):
            if any(run[key] != expected[key] for key in expected):
                raise ValueError("frozen model or split differs from validation lock")
        report = {
            "status": "validation_locked_explicit_bounded_projection_residual_test",
            "selection_lock_sha256": sha256_file(OUT / "selection_lock.json"),
            "selected_cap": cap,
            "interpretation": "The projection group remains a 64-coordinate high-capacity numeric residual; output capping alone does not satisfy the plan's low-capacity residual condition.",
            "runs": runs,
        }
        path = OUT / "fixed_test_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print({"stage": "test", "cap": cap,
               "acc": [run["bounded"]["acc"] for run in runs],
               "changes": [run["bounded_class_changes_from_original"]
                           for run in runs]})
        print(path)


if __name__ == "__main__":
    main()
