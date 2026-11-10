"""Same-label-budget MGTAB named-evidence capacity control on official dev splits."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_train_selected_concepts import rank_atomic
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v1.json"
OUTPUT = ROOT / "research/certificates/mgtab_rq1_same_budget_source_capacity.json"
ARTIFACTS = ROOT / "output/mgtab_rq1_same_budget_source_capacity"


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def fit_and_eval(train: np.ndarray, dev: np.ndarray, y_train: np.ndarray,
                 y_dev: np.ndarray, train_b: np.ndarray,
                 selected: list[int], model_path: Path) -> tuple[dict, np.ndarray]:
    width = train.shape[1] // 2
    columns = np.r_[selected, np.asarray(selected) + width]
    model = LGBMClassifier(
        objective="binary", n_estimators=200, num_leaves=15,
        min_child_samples=50, reg_lambda=10.0, learning_rate=0.05,
        n_jobs=4, random_state=0, deterministic=True,
        force_col_wise=True, verbosity=-1)
    model.fit(train[train_b][:, columns], y_train[train_b])
    probability = np.asarray(model.booster_.predict(dev[:, columns]), dtype=float)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    return {
        "fact_count": len(selected),
        "fixed_threshold": 0.5,
        "dev": evaluate(y_dev, probability, 0.5),
        "model_feature_gain_nonzero": int(np.count_nonzero(
            model.booster_.feature_importance(importance_type="gain"))),
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": sha256_file(model_path),
    }, probability


def run() -> dict:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    schema_names = [x["fact"] for x in schema["schema"]["mgtab"]["concepts"]]
    if len(schema_names) != 18 or len(set(schema_names)) != 18:
        raise ValueError("predeclared MGTAB evidence schema changed")
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=ROOT / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=ROOT / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=ROOT / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for seed in (0, 1, 2):
        old_path = args.selected_model_dir / f"seed{seed}/report.json"
        old = json.loads(old_path.read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if len(names) != 150 or names[:86] != list(dict.fromkeys(names[:86])):
            raise ValueError("published source fact vocabulary differs")
        ref_path = args.selected_model_dir / f"seed{seed}/reference.json.gz"
        reference = LabelBlindReference.load(ref_path)
        if (reference.version != old["manifest"]["reference_version"] or
                tuple(reference.names) != tuple(names)):
            raise ValueError("frozen train-only label-blind reference differs")
        train = split["train"]
        dev = split["val"]
        train_a, train_b = train_test_split(
            np.arange(len(train)), test_size=0.5, stratify=y[train],
            random_state=seed)
        evidence_train = transform_with_support(
            reference, x[train], split["supports"][train])
        evidence_dev = transform_with_support(
            reference, x[dev], split["supports"][dev])
        encoded_train = matrix(evidence_train)
        encoded_dev = matrix(evidence_dev)
        name_index = {name: i for i, name in enumerate(names)}
        schema_indices = [name_index[name] for name in schema_names]
        if any(i >= 86 for i in schema_indices):
            raise ValueError("opaque content projection entered named evidence schema")
        selected_20 = rank_atomic(evidence_train, y[train], train_a,
                                  list(range(86)), 20)
        selected_relation_20 = rank_atomic(evidence_train, y[train], train_a,
                                           list(range(20, 86)), 20)
        candidates = {
            "predeclared_schema18": schema_indices,
            "A_selected_named20": selected_20,
            "A_selected_relation20": selected_relation_20,
            "all_named86": list(range(86)),
            "all_named_relation66": list(range(20, 86)),
            "all_named_and_content150": list(range(150)),
        }
        results = {}
        probabilities = {}
        for key, selected in candidates.items():
            model_path = ARTIFACTS / f"seed{seed}/{key}.txt"
            report, probability = fit_and_eval(
                encoded_train, encoded_dev, y[train], y[dev], train_b,
                selected, model_path)
            report["selected_fact_names"] = ([names[j] for j in selected]
                                             if len(selected) <= 20 else None)
            results[key] = report
            probabilities[key] = probability
        prediction_path = ARTIFACTS / f"seed{seed}/dev_probabilities.npz"
        np.savez_compressed(prediction_path, dev_indices=dev,
                            train_A_indices=train_a, train_B_indices=train_b,
                            **probabilities)
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "source_report_sha256": sha256_file(old_path),
            "reference_sha256": sha256_file(ref_path),
            "train_accounts": len(train),
            "train_A_selection_accounts": len(train_a),
            "train_B_fit_accounts": len(train_b),
            "dev_accounts": len(dev),
            "prediction_path": str(prediction_path.relative_to(ROOT)),
            "prediction_sha256": sha256_file(prediction_path),
            "results": results,
        })
        print(json.dumps({"seed": seed,
                          "macro_f1": {key: round(value["dev"]["macro_f1"], 5)
                                       for key, value in results.items()}},
                         ensure_ascii=False), flush=True)
    result = {
        "status": "mgtab_rq1_same_budget_named_source_capacity_dev_diagnostic",
        "scope": "same official train/dev splits, label-blind train reference, B-only model fit, fixed LightGBM config and threshold 0.5; only the two selected-20 candidates additionally use A labels for fact selection",
        "equal_total_supervised_label_access_across_candidates": False,
        "source_limit": "first 86 facts are published normalized profile and typed edge/neighbor calculations; last 64 are projections of published averaged-tweet tensor and have no per-tweet semantic source",
        "fixed_test_predictions_or_metrics_computed": False,
        "dev_population_previously_explored": True,
        "independent_behavior_semantic_truth": False,
        "schema_sha256": sha256_file(SCHEMA),
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
