"""Development screen of a measured training exemplar as score origin.

Keeps the 16-fact dual-tail schema, label-blind reference, A/B directions,
group-restricted LightGBM, and prelisted anchor weights. Only the anchor
location changes from the synthetic zero vector to an observed train medoid.
No fixed-test predictions or labels enter this screen.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from gate1_fox8_botsim_schema_dev import _missing_means, _source_facts
from gate1_group_additive_schema_v2_dev import _model, _source_groups
from lock_group_additive_schema_v2_candidates import OUT as LOCK_DIR
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_schema_v2_dual_tail_neutral_dev import _context, _design, _signs
from audit_schema_v2_dual_tail_baseline_sensitivity_dev import _medoid
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = LOCK_DIR / "selection_lock.json"
PRIOR = ROOT / "output/gate1_schema_v2_dual_tail_neutral_dev/dev_only.json"
OUT = ROOT / "output/gate1_schema_v2_observed_anchor_dev/dev_only.json"
RATIOS = (0.0, 0.02, 0.1, 0.5)


def _observed_anchor_rows(x: np.ndarray, y: np.ndarray,
                          ratio: float, anchor: np.ndarray):
    if ratio == 0:
        return x, y, None
    anchors = np.repeat(anchor[None, :], 100, axis=0)
    labels = np.r_[np.zeros(50, dtype=y.dtype), np.ones(50, dtype=y.dtype)]
    weights = np.r_[np.ones(len(x)), np.full(100, ratio * len(x) / 100)]
    return np.vstack((x, anchors)), np.r_[y, labels], weights


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA) or \
            prior["selection_lock_sha256"] != sha256_file(LOCK):
        raise ValueError("frozen vocabulary or prior screen differs")
    results = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [row["fact"] for row in spec["concepts"]]
        group_names, atomic_groups = _source_groups(names, spec)
        groups = [[2*j+k for j in indices for k in (0, 1)]
                  for indices in atomic_groups]
        rows = []
        for seed in (0, 1, 2):
            frozen = lock["runs"][dataset][seed]
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != frozen["split_sha256"] or \
                    source["files_sha256"] != frozen["source_files_sha256"] or \
                    names != frozen["selected_fact_names"]:
                raise ValueError("source facts differ")
            raw = _missing_means(dataset, raw, fact_names, ids)
            facts = raw[:, [fact_names.index(name) for name in names]]
            train, dev = split["train"], split["val"]
            a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                    random_state=1000 + seed)
            reference_path = ROOT / frozen["reference_path"]
            if sha256_file(reference_path) != frozen["reference_sha256"]:
                raise ValueError("reference differs")
            reference = LabelBlindReference.load(reference_path)
            context = _context(facts, names, frozen)
            e_train = reference.transform(facts[train],
                                          None if context is None else context[train])
            e_a = reference.transform(facts[a], None if context is None else context[a])
            e_b = reference.transform(facts[b], None if context is None else context[b])
            e_dev = reference.transform(facts[dev],
                                        None if context is None else context[dev])
            x_train = _design(e_train)
            medoid_position = _medoid(x_train)
            anchor = x_train[medoid_position]
            ratio_rows = []
            for ratio in RATIOS:
                full_logits = []
                anchor_logits = []
                group_effects = []
                for direction_batch, direction_labels, fit_batch, fit_labels in (
                        (e_a, y[a], e_b, y[b]), (e_b, y[b], e_a, y[a])):
                    signs = _signs(direction_batch, direction_labels)
                    model = _model(seed, groups, signs)
                    fit_x, fit_y, weights = _observed_anchor_rows(
                        _design(fit_batch), fit_labels, ratio, anchor)
                    model.fit(fit_x, fit_y, sample_weight=weights)
                    if not all_trees_one_group(model, groups)[0]:
                        raise ValueError("source group constraint differs")
                    x_dev = _design(e_dev)
                    full = np.asarray(model.booster_.predict(x_dev, raw_score=True))
                    anchor_score = np.asarray(model.booster_.predict(
                        np.repeat(anchor[None, :], len(dev), axis=0), raw_score=True))
                    effects = []
                    for group in groups:
                        masked = x_dev.copy()
                        masked[:, group] = anchor[group]
                        without = np.asarray(model.booster_.predict(
                            masked, raw_score=True))
                        effects.append(full - without)
                    full_logits.append(full)
                    anchor_logits.append(anchor_score)
                    group_effects.append(np.column_stack(effects))
                score = np.mean(full_logits, axis=0) - np.mean(anchor_logits, axis=0)
                effects = np.mean(group_effects, axis=0)
                error = float(np.max(np.abs(score - effects.sum(axis=1))))
                if error > 1e-9:
                    raise ValueError("observed-anchor group effects do not add")
                p = expit(score)
                threshold = np.nextafter(.5, 1.0)
                if not np.array_equal(p >= threshold, score > 0):
                    raise ValueError("probability rounding changes score rule")
                metric = evaluate(y[dev], p, threshold)
                null_row = next(row for row in prior["runs"][dataset][seed]["ratios"]
                                if row["anchor_ratio"] == ratio)
                ratio_rows.append({
                    "anchor_ratio": ratio,
                    "observed_anchor_dev_metric": metric,
                    "synthetic_zero_anchor_dev_errors": null_row["neutral_dev_metric"]["errors"],
                    "zero_score_ties": int(np.count_nonzero(score == 0)),
                    "bot_predictions": int(np.count_nonzero(score > 0)),
                    "human_predictions": int(np.count_nonzero(score < 0)),
                    "max_group_additivity_error": error,
                    "observed_anchor_logit": float(np.mean(anchor_logits, axis=0)[0]),
                    "gate_within_one_percent_of_zero_anchor_and_no_ties":
                        metric["acc"] >= null_row["neutral_dev_metric"]["acc"] - .01 and
                        not np.any(score == 0),
                })
                print({"dataset": dataset, "seed": seed, "ratio": ratio,
                       "errors": metric["errors"],
                       "zero_anchor_errors": null_row["neutral_dev_metric"]["errors"],
                       "ties": ratio_rows[-1]["zero_score_ties"]}, flush=True)
            rows.append({"seed": seed, "split_sha256": frozen["split_sha256"],
                         "observed_anchor_training_account": ids[int(train[medoid_position])],
                         "observed_anchor_selected_without_labels": True,
                         "source_groups": list(group_names),
                         "ratios": ratio_rows})
        results[dataset] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "same_schema_dual_tail_measured_train_medoid_boundary_dev_only",
        "ratios_prelisted": RATIOS,
        "selection_lock_sha256": sha256_file(LOCK),
        "prior_zero_anchor_report_sha256": sha256_file(PRIOR),
        "test_predictions_or_metrics_accessed": False,
        "runs": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
