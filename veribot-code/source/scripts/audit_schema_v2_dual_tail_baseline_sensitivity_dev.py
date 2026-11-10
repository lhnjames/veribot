"""Dev-only audit: signed group reasons under null versus observed medoid.

Refits the prelisted 0.02 dual-tail candidate, checks saved dev metrics, and
compares exact group effects against two fixed, label-blind reference inputs.
It does not inspect fixed-test rows or use dev labels to choose a baseline.
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
from screen_mgtab_positive_human_evidence_dev import neutral_training_rows
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_schema_v2_dual_tail_neutral_dev import _context, _design, _signs
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "research/evidence_schema_v2.json"
LOCK = LOCK_DIR / "selection_lock.json"
DEV = ROOT / "output/gate1_schema_v2_dual_tail_neutral_dev/dev_only.json"
OUT = ROOT / "output/gate1_schema_v2_dual_tail_baseline_sensitivity_dev/report.json"
RATIO = 0.02


def _medoid(x: np.ndarray) -> int:
    median = np.median(x, axis=0)
    q1, q3 = np.quantile(x, [.25, .75], axis=0)
    scale = np.maximum(q3 - q1, .05)
    distance = np.mean(((x - median) / scale) ** 2, axis=1)
    return int(np.argmin(distance))


def _effects(model, x: np.ndarray, baseline: np.ndarray,
             groups: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full = np.asarray(model.booster_.predict(x, raw_score=True))
    base = np.asarray(model.booster_.predict(
        np.repeat(baseline[None, :], len(x), axis=0), raw_score=True))
    effects = []
    for group in groups:
        masked = x.copy()
        masked[:, group] = baseline[group]
        without = np.asarray(model.booster_.predict(masked, raw_score=True))
        effects.append(full - without)
    return full, base, np.column_stack(effects)


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    prior = json.loads(DEV.read_text(encoding="utf-8"))
    if lock["schema_sha256"] != sha256_file(SCHEMA) or \
            prior["selection_lock_sha256"] != sha256_file(LOCK):
        raise ValueError("schema or dev screen differs")
    reports = {}
    for dataset in ("fox8", "botsim"):
        spec = schema["schema"][dataset]
        names = [row["fact"] for row in spec["concepts"]]
        group_names, atomic_groups = _source_groups(names, spec)
        groups = [[2*j+k for j in members for k in (0, 1)]
                  for members in atomic_groups]
        rows = []
        for seed in (0, 1, 2):
            frozen = lock["runs"][dataset][seed]
            raw, y, ids, fact_names, split, source = _source_facts(dataset, seed)
            if source["split_sha256"] != frozen["split_sha256"] or \
                    source["files_sha256"] != frozen["source_files_sha256"]:
                raise ValueError("population differs")
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
            evidence = {key: reference.transform(
                facts[idx], None if context is None else context[idx])
                for key, idx in (("train", train), ("a", a), ("b", b),
                                 ("dev", dev))}
            x_train = _design(evidence["train"])
            medoid_index = _medoid(x_train)
            medoid = x_train[medoid_index]
            null = np.zeros(x_train.shape[1])
            fold_null = []
            fold_medoid = []
            for direction_key, fit_key, direction_ids, fit_ids in (
                    ("a", "b", a, b), ("b", "a", b, a)):
                signs = _signs(evidence[direction_key], y[direction_ids])
                model = _model(seed, groups, signs)
                fit_x, fit_y, weights = neutral_training_rows(
                    _design(evidence[fit_key]), y[fit_ids], RATIO)
                model.fit(fit_x, fit_y, sample_weight=weights)
                if not all_trees_one_group(model, groups)[0]:
                    raise ValueError("group constraint differs")
                x_dev = _design(evidence["dev"])
                fold_null.append(_effects(model, x_dev, null, groups))
                fold_medoid.append(_effects(model, x_dev, medoid, groups))
            full = np.mean([fold[0] for fold in fold_null], axis=0)
            zero = np.mean([fold[1] for fold in fold_null], axis=0)
            medoid_score = np.mean([fold[1] for fold in fold_medoid], axis=0)
            null_effects = np.mean([fold[2] for fold in fold_null], axis=0)
            medoid_effects = np.mean([fold[2] for fold in fold_medoid], axis=0)
            null_error = float(np.max(np.abs(full-zero-null_effects.sum(axis=1))))
            medoid_error = float(np.max(np.abs(
                full-medoid_score-medoid_effects.sum(axis=1))))
            if max(null_error, medoid_error) > 1e-9:
                raise ValueError("baseline-specific group decomposition differs")
            probability = expit(full-zero)
            metric = evaluate(y[dev], probability, np.nextafter(.5, 1.0))
            prior_row = next(row for row in prior["runs"][dataset][seed]["ratios"]
                             if row["anchor_ratio"] == RATIO)
            if metric["errors"] != prior_row["neutral_dev_metric"]["errors"]:
                raise ValueError("0.02 dev candidate does not replay")
            active = (np.abs(null_effects) > 1e-9) & \
                     (np.abs(medoid_effects) > 1e-9)
            sign_flip = active & (np.sign(null_effects) != np.sign(medoid_effects))
            output_path = OUT.parent / dataset / f"seed{seed}.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_path, ids=np.asarray([ids[int(i)] for i in dev]),
                                groups=np.asarray(list(group_names)), full_logit=full,
                                null_logit=zero, medoid_logit=medoid_score,
                                null_group_effects=null_effects,
                                medoid_group_effects=medoid_effects,
                                medoid_training_account=np.asarray(ids[int(train[medoid_index])]))
            rows.append({
                "seed": seed, "dev_accounts": len(dev),
                "medoid_training_account": ids[int(train[medoid_index])],
                "medoid_selected_without_labels": True,
                "dev_errors_at_predefined_null_boundary": metric["errors"],
                "max_null_additivity_error": null_error,
                "max_medoid_additivity_error": medoid_error,
                "both_baselines_nonzero_group_account_pairs": int(np.count_nonzero(active)),
                "opposite_group_sign_account_pairs": int(np.count_nonzero(sign_flip)),
                "opposite_group_sign_by_group": {
                    name: int(np.count_nonzero(sign_flip[:, j]))
                    for j, name in enumerate(group_names)},
                "account_matrix": str(output_path.relative_to(ROOT)),
                "account_matrix_sha256": sha256_file(output_path),
            })
            print({"dataset": dataset, "seed": seed,
                   "sign_flips": rows[-1]["opposite_group_sign_account_pairs"],
                   "both_active": rows[-1]["both_baselines_nonzero_group_account_pairs"]},
                  flush=True)
        reports[dataset] = rows
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "dev_only_signed_group_reason_baseline_sensitivity",
        "fixed_training_anchor_ratio": RATIO,
        "selection_lock_sha256": sha256_file(LOCK),
        "test_predictions_or_metrics_accessed": False,
        "runs": reports,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
