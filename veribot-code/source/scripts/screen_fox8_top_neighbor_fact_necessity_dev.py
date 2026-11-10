"""Train/dev ablations of the strongest frozen Fox8 neighbor source facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_fox8_neighbor_atomic_evidence_capacity/top_neighbor_fact_necessity_dev.json"
FACTS = (
    "native_interaction_neighbor_mean_unique_lang_ratio",
    "native_interaction_neighbor_mean_statuses_count",
)
CONFIGS = {
    "current_control": (),
    "without_neighbor_language_ratio": FACTS[:1],
    "without_neighbor_status_count": FACTS[1:],
    "without_both_top_neighbor_facts": FACTS,
    "without_all_native_neighbor_facts": "all_native_neighbor",
}


def matrix(runtime: Fox8NeighborAtomicRuntime) -> np.ndarray:
    batch = runtime.reference.transform(runtime.facts)
    reliability = batch.reliability.copy()
    reliability[:, 43:] *= runtime.supports / (runtime.supports + 5.0)
    result = np.column_stack((np.log1p(batch.upper) * reliability,
                              np.log1p(batch.lower) * reliability))
    if result.shape != (2279, 148) or not np.isfinite(result).all():
        raise ValueError("Fox8 encoded evidence population differs")
    return result


def fit_model(values: np.ndarray, labels: np.ndarray) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary", n_estimators=200, num_leaves=15,
        min_child_samples=30, reg_lambda=10.0, learning_rate=.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1)
    model.fit(values, labels)
    return model


def main() -> None:
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None, structural_facts=None,
        neighbor_facts=None,
        fox8_native_facts=ROOT / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=ROOT / "output/v3_fox8_native_spline_l2/selected")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        report_path = args.selected_model_dir / f"seed{seed}/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _, labels, _, split = selected_facts(args, seed, report)
        train, dev = split["train"], split["val"]
        values = matrix(runtime)
        frozen_probability = runtime.model.predict_proba(values[dev])[:, 1]
        frozen_metric = evaluate(labels[dev], frozen_probability, runtime.threshold)
        if frozen_metric["errors"] != 0:
            raise ValueError("frozen Fox8 development control differs")
        outcomes = {}
        for variant, excluded in CONFIGS.items():
            masked = values.copy()
            names = (runtime.reference.names[43:] if excluded == "all_native_neighbor"
                     else excluded)
            for name in names:
                j = runtime.reference.names.index(name)
                masked[:, j] = 0.0
                masked[:, j + 74] = 0.0
            model = fit_model(masked[train], labels[train])
            probability = model.predict_proba(masked[dev])[:, 1]
            if variant == "current_control" and np.max(np.abs(
                    probability - frozen_probability)) > 1e-12:
                raise ValueError("retrained control does not reproduce frozen model")
            outcomes[variant] = {
                "excluded_source_facts": list(names),
                "validation": evaluate(labels[dev], probability, .5),
                "deviation_from_frozen_probability_max": float(np.max(
                    np.abs(probability - frozen_probability))),
                "class_changes_from_frozen": int(np.count_nonzero(
                    (probability >= .5) != (frozen_probability >= .5))),
            }
        runs.append({"seed": seed,
                     "split_sha256": report["manifest"]["split_sha256"],
                     "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
                     "frozen_validation": frozen_metric,
                     "variants": outcomes})
        print({"seed": seed, "dev_errors": {
            key: item["validation"]["errors"] for key, item in outcomes.items()}},
            flush=True)
    OUT.write_text(json.dumps({
        "scope": "fox8_train_dev_retrain_after_top_neighbor_fact_removal",
        "test_predictions_or_metrics_used": False,
        "reference_retains_frozen_train_only_unlabeled_statistics": True,
        "source_model_lock_sha256": sha256_file(ROOT /
            "output/v3_fox8_neighbor_atomic_evidence_capacity/selection_lock.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
