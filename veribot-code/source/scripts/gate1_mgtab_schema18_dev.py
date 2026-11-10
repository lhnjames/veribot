"""Development-only capacity gate for the predeclared MGTAB evidence schema.

All model fitting and reference estimation use official training rows.  This
script never predicts or evaluates official test rows.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.evidence_schema_v1 import SCHEMA, validate_schema
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir
from verifiable_bot.v3_evidence_reasoner import (
    LabelBlindReference, SignConstrainedReasoner)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/gate1_mgtab_schema18/dev_only.json"


def _contexts(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    observed = np.asarray(train, dtype=float)
    if not np.isfinite(observed).all():
        raise ValueError("MGTAB activity context has missing training values")
    cuts = np.quantile(observed, [1 / 3, 2 / 3]).astype(float)
    labels = np.asarray([f"activity_{k}" for k in
                         np.searchsorted(cuts, values, side="right")])
    return labels, cuts.tolist()


def _lgbm(*, stump: bool, seed: int,
          monotone: list[int] | None = None) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=200, learning_rate=.05,
        num_leaves=2 if stump else 15, max_depth=1 if stump else -1,
        min_child_samples=30, reg_lambda=1.0, n_jobs=4,
        random_state=seed, deterministic=True, force_col_wise=True,
        monotone_constraints=monotone,
        verbosity=-1)


def main() -> None:
    if validate_schema()["mgtab"] != 18:
        raise ValueError("predeclared MGTAB vocabulary changed")
    schema_path = ROOT / "research/evidence_schema_v1.json"
    schema_digest = sha256_file(schema_path)
    specification = SCHEMA["mgtab"]
    names18 = [row["fact"] for row in specification["concepts"]]
    groups = {}
    for row in specification["concepts"]:
        groups.setdefault(row["group"], []).append(row["fact"])

    data_dir = resolve_mgtab_source_dir(ROOT)
    cache_path = ROOT / "data/processed/mgtab_v3/structural_facts.npz"
    cache_manifest = json.loads(cache_path.with_suffix(".json").read_text(
        encoding="utf-8"))
    if sha256_file(cache_path) != cache_manifest["facts_sha256"]:
        raise ValueError("MGTAB structural cache differs from source manifest")
    with np.load(cache_path, allow_pickle=False) as cache:
        structure = cache["values"].astype(float)
        structure_names = cache["names"].astype(str).tolist()
        structure_ids = cache["ids"].astype(str).tolist()

    runs = []
    for seed in (0, 1, 2):
        base, y, ids, names34, split, manifest = load(
            "mgtab", data_dir, None, seed)
        if structure_ids != ids or structure.shape != (len(ids), len(structure_names)):
            raise ValueError("structural rows do not match published MGTAB accounts")
        all_facts = np.column_stack((base, structure))
        all_names = names34 + structure_names
        if len(set(all_names)) != len(all_names) or set(names18) - set(all_names):
            raise ValueError("schema facts not present in published source extractors")
        selected = all_facts[:, [all_names.index(name) for name in names18]]
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=1000 + seed)
        context_col = names18.index(specification["reference_context"]["fact"])
        train_context, cuts = _contexts(selected[train, context_col],
                                        selected[train, context_col])
        all_context = np.full(len(y), "", dtype="<U16")
        all_context[train] = train_context
        all_context[dev], _ = _contexts(selected[train, context_col],
                                        selected[dev, context_col])
        if not np.array_equal(train_context, all_context[train]):
            raise ValueError("training reference contexts differ")
        reference = LabelBlindReference(min_context_n=30).fit(
            selected[train], names18, train_context)
        evidence_a = reference.transform(selected[a], all_context[a])
        evidence_b = reference.transform(selected[b], all_context[b])
        evidence_dev = reference.transform(selected[dev], all_context[dev])
        reasoner = SignConstrainedReasoner(groups, l2=.01).fit(
            evidence_a, y[a], evidence_b, y[b])
        p_evidence = reasoner.predict_proba(evidence_dev)

        def selected_tail(batch):
            values = np.column_stack([
                batch.upper[:, j] if tail == "upper" else batch.lower[:, j]
                for j, tail in enumerate(reasoner.tail)])
            return np.log1p(values) * batch.reliability

        evidence_gam = _lgbm(stump=True, seed=seed,
                             monotone=reasoner.sign.astype(int).tolist())
        evidence_gam.fit(selected_tail(evidence_b), y[b])
        p_evidence_gam = evidence_gam.predict_proba(selected_tail(evidence_dev))[:, 1]
        dual_b = np.column_stack((np.log1p(evidence_b.upper) * evidence_b.reliability,
                                  np.log1p(evidence_b.lower) * evidence_b.reliability))
        dual_dev = np.column_stack((np.log1p(evidence_dev.upper) * evidence_dev.reliability,
                                    np.log1p(evidence_dev.lower) * evidence_dev.reliability))
        dual_gam = _lgbm(stump=True, seed=seed)
        dual_gam.fit(dual_b, y[b])
        p_dual_gam = dual_gam.predict_proba(dual_dev)[:, 1]

        controls = {}
        published = torch.load(data_dir / "features.pt", map_location="cpu",
                               weights_only=True).numpy()
        matrices = {
            "same_18_raw_fact_gam_stumps": (selected, True),
            "same_18_raw_fact_blackbox": (selected, False),
            "all_published_788_plus_28_structure_blackbox":
                (np.column_stack((published, structure)), False),
        }
        for label, (matrix, stump) in matrices.items():
            model = _lgbm(stump=stump, seed=seed)
            model.fit(matrix[train], y[train])
            controls[label] = evaluate(y[dev],
                model.predict_proba(matrix[dev])[:, 1], .5)
        runs.append({
            "seed": seed, "split_sha256": manifest["split_sha256"],
            "training_accounts": len(train), "direction_accounts": len(a),
            "coefficient_accounts": len(b), "dev_accounts": len(dev),
            "reference_version": reference.version,
            "activity_context_train_quantiles": cuts,
            "computed_evidence": evaluate(y[dev], p_evidence, .5),
            "computed_evidence_monotone_gam_stumps": evaluate(
                y[dev], p_evidence_gam, .5),
            "computed_evidence_dual_tail_gam_capacity_upper_bound": evaluate(
                y[dev], p_dual_gam, .5),
            "controls": controls,
            "directions": reasoner.manifest()["directions"],
        })
        print({"seed": seed,
               "evidence_macro_f1": runs[-1]["computed_evidence"]["macro_f1"],
               "raw_gam_macro_f1": controls["same_18_raw_fact_gam_stumps"]["macro_f1"],
               "full_blackbox_macro_f1": controls[
                   "all_published_788_plus_28_structure_blackbox"]["macro_f1"]},
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "scope": "MGTAB predeclared 18 named computed evidence capacity gate",
        "schema_sha256": schema_digest,
        "source_features_sha256": sha256_file(data_dir / "features.pt"),
        "structural_cache_sha256": sha256_file(cache_path),
        "reference_fit_uses_train_values_without_labels": True,
        "direction_A_and_coefficient_B_disjoint": True,
        "controls_fit_on_all_training_labels": True,
        "evidence_gam_fits_only_B_after_A_directions": True,
        "classification_threshold": .5,
        "test_predictions_or_metrics_accessed": False,
        "semantic_truth_independently_annotated": False,
        "prior_fixed_test_population_explored_elsewhere": True,
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
