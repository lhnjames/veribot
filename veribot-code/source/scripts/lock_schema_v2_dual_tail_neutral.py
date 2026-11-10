"""Freeze the dev-selected schema-v2 dual-tail candidate before test scoring."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _model, _source_groups, CONFIG
from screen_schema_v2_dual_tail_neutral_dev import _context, _design, _signs
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
PARENT_LOCK = ROOT / "output/evidence_schema_v2_group_additive_candidates/selection_lock.json"
DEV = ROOT / "output/gate1_schema_v2_dual_tail_neutral_dev/dev_only.json"
OUT = ROOT / "output/schema_v2_dual_tail_neutral_candidate"
RATIO = 0.02


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    parent = json.loads(PARENT_LOCK.read_text(encoding="utf-8"))
    dev_report = json.loads(DEV.read_text(encoding="utf-8"))
    if (parent["schema_sha256"] != sha256_file(SCHEMA)
            or dev_report["selection_lock_sha256"] != sha256_file(PARENT_LOCK)
            or dev_report["test_predictions_or_metrics_accessed"] is not False
            or dev_report["ratios_prelisted"] != [0.0, 0.02, 0.1, 0.5]):
        raise ValueError("Dev protocol or parent lock differs")
    runs = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [item["fact"] for item in spec["concepts"]]
        _, single_groups = _source_groups(names, spec)
        groups = [[2*j+k for j in members for k in (0, 1)]
                  for members in single_groups]
        dataset_runs = []
        for seed in (0, 1, 2):
            frozen = parent["runs"][dataset][seed]
            screen = dev_report["runs"][dataset][seed]
            chosen = next(item for item in screen["ratios"]
                          if item["anchor_ratio"] == RATIO)
            if not chosen["screen_gate_no_more_than_one_percent_acc_loss_and_no_ties"]:
                raise ValueError("Chosen dev candidate did not pass screen")
            raw, labels, ids, fact_names, split, source = _source_facts(dataset, seed)
            if (source["split_sha256"] != frozen["split_sha256"]
                    or source["files_sha256"] != frozen["source_files_sha256"]
                    or names != frozen["selected_fact_names"]
                    or source["split_sha256"] != screen["split_sha256"]):
                raise ValueError("Source or schema differs from development screen")
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            train, val = split["train"], split["val"]
            a, b = train_test_split(train, test_size=.5,
                                    stratify=labels[train],
                                    random_state=1000+seed)
            reference_path = ROOT / frozen["reference_path"]
            if sha256_file(reference_path) != frozen["reference_sha256"]:
                raise ValueError("Frozen label-blind reference differs")
            reference = LabelBlindReference.load(reference_path)
            context = _context(facts, names, frozen)
            batches = [reference.transform(facts[subset],
                       None if context is None else context[subset])
                       for subset in (a, b, val)]
            e_a, e_b, e_val = batches
            folder = OUT / dataset / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            model_paths = []
            signs_by_fold = []
            logits = []
            zeros = []
            for fold, (direction_batch, direction_y, fit_batch, fit_y) in enumerate((
                    (e_a, labels[a], e_b, labels[b]),
                    (e_b, labels[b], e_a, labels[a]))):
                signs = _signs(direction_batch, direction_y)
                model = _model(seed, groups, signs)
                fit_x, fit_labels, weights = neutral_training_rows(
                    _design(fit_batch), fit_y, RATIO)
                model.fit(fit_x, fit_labels, sample_weight=weights)
                if not all_trees_one_group(model, groups)[0]:
                    raise ValueError("Frozen tree crosses source groups")
                path = folder / f"fold{fold}.joblib"
                joblib.dump(model, path)
                model_paths.append(path)
                signs_by_fold.append(signs)
                loaded = joblib.load(path)
                x = _design(e_val)
                logits.append(np.asarray(loaded.booster_.predict(
                    x, raw_score=True)))
                zeros.append(float(loaded.booster_.predict(
                    np.zeros((1, x.shape[1])), raw_score=True)[0]))
            signs_path = folder / "signs.json"
            signs_path.write_text(json.dumps(signs_by_fold, indent=2) + "\n",
                                  encoding="utf-8")
            score = np.mean(logits, axis=0) - np.mean(zeros)
            probability = expit(score)
            threshold = float(np.nextafter(.5, 1.0))
            if not np.array_equal(probability >= threshold, score > 0):
                raise ValueError("Neutral decision boundary differs")
            metric = evaluate(labels[val], probability, threshold)
            prior = chosen["neutral_dev_metric"]
            if (metric["errors"] != prior["errors"]
                    or abs(metric["acc"]-prior["acc"]) > 1e-12
                    or abs(metric["macro_f1"]-prior["macro_f1"]) > 1e-12
                    or abs(np.mean(zeros)-chosen["zero_input_logit"]) > 1e-12
                    or int(np.count_nonzero(score == 0)) != chosen["zero_score_ties"]):
                raise ValueError("Readback fails dev-screen replay")
            dataset_runs.append({
                "seed": seed,
                "split_sha256": source["split_sha256"],
                "source_files_sha256": source["files_sha256"],
                "selected_fact_names": names,
                "source_groups": {name: members for name, members in
                                  zip(_source_groups(names, spec)[0], groups, strict=True)},
                "reference_path": frozen["reference_path"],
                "reference_sha256": frozen["reference_sha256"],
                "reference_version": frozen["reference_version"],
                "reference_context_rule": frozen["reference_context_rule"],
                "reference_context_fact": frozen["reference_context_fact"],
                "train_only_context_quantiles": frozen["train_only_context_quantiles"],
                "model_paths": [str(path.relative_to(ROOT)) for path in model_paths],
                "model_sha256": [sha256_file(path) for path in model_paths],
                "signs_path": str(signs_path.relative_to(ROOT)),
                "signs_sha256": sha256_file(signs_path),
                "zero_input_logit_by_fold": zeros,
                "decision_probability_threshold": threshold,
                "dev_metric_replayed": metric,
                "dev_zero_score_ties": int(np.count_nonzero(score == 0)),
            })
            print(dataset, seed, metric["errors"], zeros, flush=True)
        runs[dataset] = dataset_runs
    report = {
        "status": "dual_tail_zero_anchor_candidate_locked_before_this_test_score",
        "schema_sha256": sha256_file(SCHEMA),
        "parent_lock_sha256": sha256_file(PARENT_LOCK),
        "dev_report_sha256": sha256_file(DEV),
        "selected_anchor_ratio": RATIO,
        "selection_basis": "same_16_facts_dev_accuracy_within_one_percent_of_parent_and_no_dev_ties",
        "model_configuration": CONFIG,
        "test_predictions_or_metrics_accessed": False,
        "zero_vector_is_observed_account": False,
        "runs": runs,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_lock.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT / "selection_lock.json")


if __name__ == "__main__":
    main()
