"""Freeze Fox8/BotSim evidence-only crossfit candidates before test evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _contexts, _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _model, _source_groups, CONFIG
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "research/evidence_schema_v2.json"
DEV_PATH = ROOT / "output/gate1_group_additive_schema_v2/dev_only.json"
OUT = ROOT / "output/evidence_schema_v2_group_additive_candidates"


def _design(batch, direction_manifest: dict, names: list[str]) -> np.ndarray:
    direction = direction_manifest["directions"]
    tails = np.column_stack([
        batch.upper[:, j] if direction[name]["tail"] == "upper" else
        batch.lower[:, j] for j, name in enumerate(names)])
    return np.log1p(tails) * batch.reliability


def main() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    prior = json.loads(DEV_PATH.read_text(encoding="utf-8"))
    if (schema["version"] != "candidate-v2" or
            prior["schema_sha256"] != sha256_file(SCHEMA_PATH) or
            prior["fixed_model_configuration"] != CONFIG or
            prior["test_predictions_or_metrics_accessed"] is not False):
        raise ValueError("candidate development protocol differs")
    runs = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [item["fact"] for item in spec["concepts"]]
        groups, indices = _source_groups(names, spec)
        dataset_runs = []
        for seed in (0, 1, 2):
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            train, dev = split["train"], split["val"]
            a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                    random_state=1000 + seed)
            context_spec = spec["reference_context"]
            if context_spec["rule"] == "pooled_unlabeled_training_reference":
                context = None
                cuts = None
            elif context_spec["rule"] == \
                    "unlabeled_training_tertiles_with_global_fallback":
                j = names.index(context_spec["fact"])
                context = np.full(len(y), "", dtype="<U16")
                context[train], cuts = _contexts(facts[train, j], facts[train, j])
                context[dev], _ = _contexts(facts[train, j], facts[dev, j])
            else:
                raise ValueError("unexpected context policy")
            reference = LabelBlindReference(min_context_n=30).fit(
                facts[train], names, None if context is None else context[train])
            e_a = reference.transform(facts[a], None if context is None else context[a])
            e_b = reference.transform(facts[b], None if context is None else context[b])
            e_dev = reference.transform(facts[dev], None if context is None else context[dev])

            models = []
            direction_manifests = []
            for direction_indices, fit_indices, direction_batch, fit_batch in (
                (a, b, e_a, e_b), (b, a, e_b, e_a)):
                direction = SignConstrainedReasoner(groups, l2=.01).fit(
                    direction_batch, y[direction_indices], fit_batch, y[fit_indices])
                direction_manifest = direction.manifest()
                model = _model(seed, indices, direction.sign.astype(int).tolist())
                model.fit(_design(fit_batch, direction_manifest, names), y[fit_indices])
                valid, _ = all_trees_one_group(model, indices)
                if not valid:
                    raise ValueError("frozen candidate crosses evidence groups")
                direction_manifests.append(direction_manifest)
                models.append(model)
            logit = sum(np.asarray(model.booster_.predict(
                _design(e_dev, direction_manifest, names), raw_score=True))
                for model, direction_manifest in zip(models, direction_manifests,
                                                      strict=True)) / 2
            probability = expit(logit)
            threshold = best_validation_threshold(probability, y[dev])
            metric = evaluate(y[dev], probability, threshold)
            old = prior["runs"][dataset][seed]
            if (old["split_sha256"] != source["split_sha256"] or
                abs(old["crossfit_group_gam_dev_calibrated"]["threshold"] - threshold)
                    > 1e-12 or
                old["crossfit_group_gam_dev_calibrated"]["validation"]["errors"] !=
                    metric["errors"]):
                raise ValueError("locked candidate does not replay development result")

            folder = OUT / dataset / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            reference_path = folder / "reference.json.gz"
            reference.save(reference_path)
            model_paths = []
            for index, model in enumerate(models):
                path = folder / f"fold{index}.joblib"
                joblib.dump(model, path)
                model_paths.append(path)
            direction_path = folder / "directions.json"
            direction_path.write_text(json.dumps(direction_manifests,
                indent=2) + "\n", encoding="utf-8")

            loaded_ref = LabelBlindReference.load(reference_path)
            loaded_models = [joblib.load(path) for path in model_paths]
            loaded_directions = json.loads(direction_path.read_text(encoding="utf-8"))
            loaded_e = loaded_ref.transform(
                facts[dev], None if context is None else context[dev])
            loaded_logit = sum(np.asarray(model.booster_.predict(
                _design(loaded_e, manifest, names), raw_score=True))
                for model, manifest in zip(loaded_models, loaded_directions,
                                           strict=True)) / 2
            readback_error = float(np.max(np.abs(loaded_logit - logit)))
            if readback_error > 1e-12:
                raise ValueError("saved model/reference readback differs")
            dataset_runs.append({
                "seed": seed, "split_sha256": source["split_sha256"],
                "source_files_sha256": source["files_sha256"],
                "selected_fact_names": names,
                "reference_context_rule": context_spec["rule"],
                "reference_context_fact": context_spec["fact"],
                "train_only_context_quantiles": cuts,
                "reference_version": reference.version,
                "classification_threshold_selected_on_dev": threshold,
                "dev_metric_on_same_rows_used_to_calibrate": metric,
                "reference_path": str(reference_path.relative_to(ROOT)),
                "reference_sha256": sha256_file(reference_path),
                "model_paths": [str(path.relative_to(ROOT)) for path in model_paths],
                "model_sha256": [sha256_file(path) for path in model_paths],
                "direction_path": str(direction_path.relative_to(ROOT)),
                "direction_sha256": sha256_file(direction_path),
                "max_dev_logit_readback_error": readback_error,
            })
            print({"dataset": dataset, "seed": seed,
                   "dev_correct": len(dev) - metric["errors"],
                   "threshold": threshold,
                   "readback_error": readback_error}, flush=True)
        runs[dataset] = dataset_runs
    lock = {"status": "development_locked_candidate_not_independently_tested",
            "schema_sha256": sha256_file(SCHEMA_PATH),
            "development_report_sha256": sha256_file(DEV_PATH),
            "model_configuration": CONFIG,
            "test_predictions_or_metrics_accessed": False,
            "dev_threshold_selection_and_dev_report_share_rows": True,
            "source_level_semantic_truth_independently_validated": False,
            "runs": runs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection_lock.json").write_text(json.dumps(lock, indent=2) + "\n",
                                               encoding="utf-8")
    print(OUT / "selection_lock.json")


if __name__ == "__main__":
    main()
