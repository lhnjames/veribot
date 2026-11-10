"""Train/dev-only group-additive monotone reasoner over named evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _contexts, _missing_means, _source_facts
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
OUT = ROOT / "output/gate1_group_additive_schema_v2/dev_only.json"
CONFIG = {"n_estimators": 250, "learning_rate": .05, "num_leaves": 7,
          "max_depth": 3, "min_child_samples": 20, "reg_lambda": 1.0}


def _model(seed: int, groups: list[list[int]],
           directions: list[int] | None) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_jobs=4, random_state=seed,
        deterministic=True, force_col_wise=True, verbosity=-1,
        monotone_constraints=directions,
        monotone_constraints_method="advanced",
        interaction_constraints=groups,
        **CONFIG)


def _source_groups(names: list[str], spec: dict) -> tuple[dict[str, list[str]], list[list[int]]]:
    by_group = {}
    for row in spec["concepts"]:
        by_group.setdefault(row["group"], []).append(row["fact"])
    index = {name: j for j, name in enumerate(names)}
    groups = [[index[name] for name in members] for members in by_group.values()]
    if sorted(j for members in groups for j in members) != list(range(len(names))):
        raise ValueError("evidence source groups do not partition concepts")
    return by_group, groups


def _run(dataset: str, spec: dict) -> list[dict]:
    names = [row["fact"] for row in spec["concepts"]]
    by_group, group_indices = _source_groups(names, spec)
    runs = []
    for seed in (0, 1, 2):
        raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
        raw = _missing_means(dataset, raw, fact_names, ids)
        selected = raw[:, [fact_names.index(name) for name in names]]
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=1000 + seed)
        if spec["reference_context"]["rule"] == \
                "pooled_unlabeled_training_reference":
            context = None
        elif spec["reference_context"]["rule"] == \
                "unlabeled_training_tertiles_with_global_fallback":
            j = names.index(spec["reference_context"]["fact"])
            context = np.full(len(y), "", dtype="<U16")
            context[train], _ = _contexts(selected[train, j], selected[train, j])
            context[dev], _ = _contexts(selected[train, j], selected[dev, j])
        else:
            raise ValueError("reference context rule differs")
        reference = LabelBlindReference(min_context_n=30).fit(
            selected[train], names, None if context is None else context[train])
        evidence_a = reference.transform(selected[a], None if context is None else context[a])
        evidence_b = reference.transform(selected[b], None if context is None else context[b])
        evidence_dev = reference.transform(selected[dev], None if context is None else context[dev])
        direction = SignConstrainedReasoner(by_group, l2=.01).fit(
            evidence_a, y[a], evidence_b, y[b])

        def design(batch):
            values = np.column_stack([
                batch.upper[:, j] if tail == "upper" else batch.lower[:, j]
                for j, tail in enumerate(direction.tail)])
            return np.log1p(values) * batch.reliability

        model = _model(seed, group_indices, direction.sign.astype(int).tolist())
        model.fit(design(evidence_b), y[b])
        all_one_group, tree_counts = all_trees_one_group(model, group_indices)
        if not all_one_group:
            raise ValueError("evidence tree crosses groups")
        x = design(evidence_dev)
        logits = np.asarray(model.booster_.predict(x, raw_score=True))
        p = expit(logits)
        zero_logits = np.asarray(model.booster_.predict(np.zeros_like(x), raw_score=True))
        group_effects = []
        group_changes = {}
        for group, members in zip(by_group, group_indices, strict=True):
            masked = x.copy()
            masked[:, members] = 0.0
            without = np.asarray(model.booster_.predict(masked, raw_score=True))
            group_effects.append(logits - without)
            group_changes[group] = int(np.count_nonzero((expit(without) >= .5) !=
                                                         (p >= .5)))
        additive_error = float(np.max(np.abs(
            logits - zero_logits - np.sum(group_effects, axis=0))))
        if additive_error > 1e-9:
            raise ValueError("group-masked logits do not exactly add")

        reverse_direction = SignConstrainedReasoner(by_group, l2=.01).fit(
            evidence_b, y[b], evidence_a, y[a])

        def reverse_design(batch):
            values = np.column_stack([
                batch.upper[:, j] if tail == "upper" else batch.lower[:, j]
                for j, tail in enumerate(reverse_direction.tail)])
            return np.log1p(values) * batch.reliability

        reverse_model = _model(
            seed, group_indices, reverse_direction.sign.astype(int).tolist())
        reverse_model.fit(reverse_design(evidence_a), y[a])
        reverse_one_group, reverse_tree_counts = all_trees_one_group(
            reverse_model, group_indices)
        if not reverse_one_group:
            raise ValueError("reverse evidence tree crosses groups")
        reverse_x = reverse_design(evidence_dev)
        reverse_logits = np.asarray(reverse_model.booster_.predict(
            reverse_x, raw_score=True))
        crossfit_logits = (logits + reverse_logits) / 2
        crossfit_p = expit(crossfit_logits)
        reverse_zero = np.asarray(reverse_model.booster_.predict(
            np.zeros_like(reverse_x), raw_score=True))
        crossfit_group_effects = []
        for members in group_indices:
            reverse_masked = reverse_x.copy()
            reverse_masked[:, members] = 0.0
            reverse_without = np.asarray(reverse_model.booster_.predict(
                reverse_masked, raw_score=True))
            crossfit_group_effects.append((reverse_logits - reverse_without) / 2)
        crossfit_error = float(np.max(np.abs(
            crossfit_logits - (zero_logits + reverse_zero) / 2 -
            (np.sum(group_effects, axis=0) / 2 +
             np.sum(crossfit_group_effects, axis=0)))))
        if crossfit_error > 1e-9:
            raise ValueError("crossfit group contributions do not add")
        raw_model = _model(seed, group_indices, None)
        raw_model.fit(selected[train], y[train])
        raw_one_group, _ = all_trees_one_group(raw_model, group_indices)
        if not raw_one_group:
            raise ValueError("raw-fact GAM tree crosses groups")
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": reference.version,
            "A_direction_accounts": len(a), "B_fit_accounts": len(b),
            "dev_accounts": len(dev),
            "evidence_group_gam": evaluate(y[dev], p, .5),
            "evidence_group_gam_dev_calibrated": {
                "threshold": best_validation_threshold(p, y[dev]),
                "validation": evaluate(y[dev], p,
                    best_validation_threshold(p, y[dev]))},
            "crossfit_group_gam_at_half": evaluate(y[dev], crossfit_p, .5),
            "crossfit_group_gam_dev_calibrated": {
                "threshold": best_validation_threshold(crossfit_p, y[dev]),
                "validation": evaluate(y[dev], crossfit_p,
                    best_validation_threshold(crossfit_p, y[dev]))},
            "same_facts_raw_group_gam_full_train": evaluate(y[dev],
                raw_model.predict_proba(selected[dev])[:, 1], .5),
            "all_trees_single_group": True,
            "tree_counts_by_group": dict(zip(by_group, [tree_counts.get(k, 0)
                                           for k in range(len(by_group))], strict=True)),
            "max_exact_group_additivity_error": additive_error,
            "max_crossfit_group_additivity_error": crossfit_error,
            "reverse_tree_counts_by_group": dict(zip(by_group,
                [reverse_tree_counts.get(k, 0) for k in range(len(by_group))],
                strict=True)),
            "group_class_changes_when_zeroed": group_changes,
            "direction_by_fact": direction.manifest()["directions"],
            "reverse_direction_by_fact": reverse_direction.manifest()["directions"],
        })
        print({"dataset": dataset, "seed": seed,
               "evidence_acc": runs[-1]["evidence_group_gam"]["acc"],
               "evidence_macro_f1": runs[-1]["evidence_group_gam"]["macro_f1"],
               "crossfit_errors_at_half": runs[-1]["crossfit_group_gam_at_half"]["errors"],
               "crossfit_errors_dev_calibrated": runs[-1][
                   "crossfit_group_gam_dev_calibrated"]["validation"]["errors"],
               "raw_group_gam_macro_f1": runs[-1][
                   "same_facts_raw_group_gam_full_train"]["macro_f1"]}, flush=True)
    return runs


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if schema["version"] != "candidate-v2":
        raise ValueError("schema version differs")
    result = {"scope": "Fox8/BotSim schema-v2 group-additive monotone evidence reasoner, official train/dev only",
              "schema_sha256": sha256_file(SCHEMA),
              "fixed_model_configuration": CONFIG,
              "test_predictions_or_metrics_accessed": False,
              "evidence_A_direction_B_fit_disjoint": True,
              "crossfit_reverses_A_B_and_averages_two_additive_logits": True,
              "calibrated_threshold_and_reported_validation_reuse_same_dev_rows": True,
              "raw_fact_group_controls_use_full_train": True,
              "runs": {dataset: _run(dataset, schema["schema"][dataset])
                       for dataset in ("fox8", "botsim")}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
