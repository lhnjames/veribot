"""Compare compact evidence to additive and interaction baselines on the same 20 raw facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import metric, sha256, threshold_on_dev


def raw_additive_design(raw: np.ndarray, train: np.ndarray, names: list[str]):
    columns = []
    metadata = {}
    for j, name in enumerate(names):
        observed = raw[train, j]
        observed = observed[np.isfinite(observed)]
        if not len(observed):
            raise ValueError(f"All training values missing: {name}")
        low, high = np.quantile(observed, [0.01, 0.99])
        missing = ~np.isfinite(raw[:, j])
        replacement = float(np.median(observed))
        if high > low:
            z = (np.clip(np.where(missing, replacement, raw[:, j]), low, high) - low) / (high - low)
            columns.append(z)
            knots = sorted({float((q - low) / (high - low))
                            for q in np.quantile(observed, [0.25, 0.5, 0.75])
                            if low < q < high})
            for knot in knots:
                columns.append(np.maximum(z - knot, 0))
        else:
            knots = []
            columns.append(np.zeros(len(raw)))
        columns.append(missing.astype(float))
        metadata[name] = {"train_p01": float(low), "train_p99": float(high),
                          "train_median": replacement, "hinge_knots_scaled": knots,
                          "basis_columns": 2 + len(knots)}
    return np.column_stack(columns), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if sha256(args.facts) != compact["base_facts_sha256"]:
        raise ValueError("Compact evidence and raw fact source differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        names = saved["names"].astype(str).tolist()
    train, dev, test = (np.flatnonzero(split == part) for part in ("train", "dev", "test"))
    retained = compact["retained_facts"]
    if len(retained) != 20 or [row["account_id"] for row in compact["rows"]] != ids[test].tolist():
        raise ValueError("Compact 20-fact test population differs")
    raw = values[:, [names.index(name) for name in retained]]
    design, basis = raw_additive_design(raw, train, retained)
    additive_dev_runs = []
    fitted = {}
    for penalty in ("l1", "l2"):
        model = LogisticRegression(C=1.0, penalty=penalty, solver="liblinear",
                                   max_iter=2000, random_state=0)
        model.fit(design[train], labels[train])
        dev_p = model.predict_proba(design[dev])[:, 1]
        threshold = threshold_on_dev(dev_p, labels[dev])
        additive_dev_runs.append({"penalty": penalty,
                                  "dev": metric(labels[dev], dev_p, threshold),
                                  "threshold": threshold,
                                  "nonzero_basis_terms": int(np.count_nonzero(model.coef_))})
        fitted[penalty] = (model, threshold)
    chosen = max(additive_dev_runs,
                 key=lambda row: (row["dev"]["macro_f1"], row["penalty"] == "l1"))
    additive_model, additive_threshold = fitted[chosen["penalty"]]
    additive_p = additive_model.predict_proba(design[test])[:, 1]
    additive_class = additive_p >= additive_threshold
    interaction = HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, max_depth=3,
        learning_rate=0.05, l2_regularization=1.0, random_state=0)
    interaction.fit(raw[train], labels[train])
    interaction_dev_p = interaction.predict_proba(raw[dev])[:, 1]
    interaction_threshold = threshold_on_dev(interaction_dev_p, labels[dev])
    interaction_p = interaction.predict_proba(raw[test])[:, 1]
    interaction_class = interaction_p >= interaction_threshold
    compact_class = np.asarray([row["compact_class"] for row in compact["rows"]], dtype=int)
    if compact["test"]["errors"] != int(np.count_nonzero(compact_class != labels[test])):
        raise ValueError("Compact certificate class errors differ")
    result = {
        "scope": "post hoc official train/dev/test diagnostic; test population previously explored",
        "source_archive_sha256": compact["source_archive_sha256"],
        "base_facts_sha256": compact["base_facts_sha256"],
        "compact_certificate_sha256": sha256(args.compact),
        "same_20_raw_facts": retained,
        "additive_basis": "train-only p01/p99 clipping, linear value plus train-only p25/p50/p75 hinges and missing indicator per raw fact",
        "additive_basis_count": design.shape[1],
        "additive_basis_by_fact": basis,
        "additive_dev_runs": additive_dev_runs,
        "additive_selected_by_dev_macro_f1": chosen["penalty"],
        "additive_test": metric(labels[test], additive_p, additive_threshold),
        "additive_vs_compact_paired": {
            "compact_only_correct": int(np.count_nonzero((compact_class == labels[test])
                                                             & (additive_class != labels[test]))),
            "additive_only_correct": int(np.count_nonzero((compact_class != labels[test])
                                                              & (additive_class == labels[test]))),
        },
        "interaction_protocol": "HistGradientBoosting 200 iter, 15 leaves, depth 3, learning rate .05, L2 1; same 20 raw facts, dev threshold",
        "interaction_dev": metric(labels[dev], interaction_dev_p, interaction_threshold),
        "interaction_test": metric(labels[test], interaction_p, interaction_threshold),
        "interaction_vs_compact_paired": {
            "compact_only_correct": int(np.count_nonzero((compact_class == labels[test])
                                                             & (interaction_class != labels[test]))),
            "interaction_only_correct": int(np.count_nonzero((compact_class != labels[test])
                                                                 & (interaction_class == labels[test]))),
        },
        "rows": [{"account_id": str(ids[i]), "label_for_evaluation": int(labels[i]),
                  "compact_class": int(compact_class[k]),
                  "raw_additive_probability": float(additive_p[k]),
                  "raw_additive_class": int(additive_class[k]),
                  "raw_interaction_probability": float(interaction_p[k]),
                  "raw_interaction_class": int(interaction_class[k])}
                 for k, i in enumerate(test)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"
                      and key != "additive_basis_by_fact"}, indent=2))


if __name__ == "__main__":
    main()
