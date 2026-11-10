"""Independently replay Fox8 facts, twenty concepts, and frozen decisions."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_fox8_neighbor_atomic_evidence_capacity_dev import evidence_matrix
from screen_fox8_native_neighbor_profile_concepts_dev import weighted_atomic
from screen_mgtab_ab_monotone_evidence_tree_dev import directions
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    lock_path = folder / "selection_lock_ab_group_tree.json"
    report_path = folder / "fixed_test_ab_group_tree.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["selection_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("Selection lock changed")
    audit_path = root / "research/certificates/fox8_neighbor_volume_full_raw_source.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cache_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/fox8_23")
    feature_path = data_dir / "features.csv"
    if (report["full_raw_source_audit_sha256"] != sha256_file(audit_path)
            or audit["fact_cache_sha256"] != sha256_file(cache_path)
            or audit["published_feature_sha256"] != sha256_file(feature_path)
            or audit["event_index_sha256"] != sha256_file(event_path)):
        raise ValueError("Raw source audit or feature graph changed")
    features = pd.read_csv(feature_path)
    ids = features["user_id"].astype(str).tolist()
    with np.load(cache_path, allow_pickle=False) as cache:
        if ids != cache["ids"].astype(str).tolist():
            raise ValueError("Native-neighbor account order differs")
        names_new = cache["names"].astype(str).tolist()
        values_new = cache["values"].astype(float)
        support_new = cache["supports"].astype(float)
    neighbor_name = "native_interaction_neighbor_mean_statuses_count"
    idx = names_new.index(neighbor_name)
    own_status = features["statuses_count"].to_numpy(dtype=float)
    index = {account: j for j, account in enumerate(ids)}
    neighbors = [set() for _ in ids]
    events = json.loads(event_path.read_text(encoding="utf-8"))["events"]
    for event in events:
        a = index.get(str(event["source_id"]))
        b = index.get(str(event["target_id"]))
        if a is not None and b is not None and a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    means = np.asarray([np.mean(own_status[sorted(row)]) if row else np.nan
                        for row in neighbors], dtype=np.float32)
    if (not np.array_equal(np.isfinite(means), np.isfinite(values_new[:, idx]))
            or not np.array_equal(means[np.isfinite(means)],
                                  values_new[np.isfinite(means), idx])
            or not np.array_equal(np.asarray([len(row) for row in neighbors]),
                                  support_new[:, idx])):
        raise ValueError("Decisive native-neighbor concept differs from original graph")
    old_groups = json.loads((root / "research/fox8_20_concept_groups.json")
                            .read_text(encoding="utf-8"))
    removed = set(old_groups["presentation_flags"])
    groups = {name: members for name, members in old_groups.items()
              if name != "presentation_flags"}
    groups["native_neighbor_volume"] = [neighbor_name]
    group_names = list(groups)
    group_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
    args = SimpleNamespace(
        dataset="fox8", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    rows = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        saved_path = run_dir / "ab_group_tree_test_evidence.npz"
        model_path = run_dir / "ab_group_tree.joblib"
        ref_path = run_dir / "label_blind_reference.json.gz"
        prediction_path = run_dir / "ab_group_tree_test_predictions.json"
        result = report["runs"][seed]
        if any(result[key] != sha256_file(path) for key, path in (
                ("model_sha256", model_path), ("reference_sha256", ref_path),
                ("evidence_sha256", saved_path),
                ("predictions_sha256", prediction_path))):
            raise ValueError("Frozen model or prediction fingerprint changed")
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        test = split["test"]
        full = np.column_stack((facts, values_new))
        all_names = names + names_new
        reference = LabelBlindReference.load(ref_path)
        fresh = LabelBlindReference(min_context_n=30).fit(full[split["train"]], all_names)
        if (reference.version != fresh.version or
                reference.version != lock["reference_version_by_seed"][str(seed)]):
            raise ValueError("Train-only label-blind reference differs")
        kept = [j for j, name in enumerate(all_names)
                if (j < 43 and name not in removed) or name == neighbor_name]
        atomic = columns(weighted_atomic(reference, full[test], support_new[test]), kept)
        concept, winners = collapse_group_tails(atomic, groups)
        x = evidence_matrix(concept)
        model = joblib.load(model_path)
        one_group, trees = all_trees_one_group(model, group_cols)
        if (not one_group or {str(k): v for k, v in trees.items()} !=
                result["tree_count_by_group"]):
            raise ValueError("Source-group tree structure differs")
        train_atomic = columns(weighted_atomic(
            reference, full[split["train"]], support_new[split["train"]]), kept)
        train_concept, _ = collapse_group_tails(train_atomic, groups)
        x_train = evidence_matrix(train_concept)
        a, _ = train_test_split(np.arange(len(split["train"])), test_size=0.5,
                                stratify=labels[split["train"]], random_state=seed)
        if not np.array_equal(directions(x_train[a], labels[split["train"]][a]),
                              np.asarray(model.get_params()["monotone_constraints"])):
            raise ValueError("Model monotone directions differ from A half")
        p = model.predict_proba(x)[:, 1]
        logit = np.asarray(model.booster_.predict(x, raw_score=True))
        shap = np.asarray(model.booster_.predict(x, pred_contrib=True))
        contributions = np.column_stack([shap[:, cols].sum(axis=1)
                                         for cols in group_cols])
        neighbor_index = group_names.index("native_neighbor_volume")
        masked = x.copy()
        masked[:, group_cols[neighbor_index]] = 0
        neighbor_p = model.predict_proba(masked)[:, 1]
        all_p = model.predict_proba(np.zeros_like(x))[:, 1]
        with np.load(saved_path, allow_pickle=False) as saved:
            if (saved["ids"].astype(str).tolist() != [ids[int(i)] for i in test]
                    or not np.array_equal(saved["label"], labels[test])
                    or saved["concept_names"].astype(str).tolist() != list(concept.names)
                    or saved["retained_fact_names"].astype(str).tolist() !=
                    [all_names[j] for j in kept]
                    or not np.allclose(saved["retained_raw_values"],
                                       full[test][:, kept], equal_nan=True, rtol=0, atol=0)
                    or not np.array_equal(saved["concept_scores"], concept.upper)
                    or not np.array_equal(saved["model_input"], x)
                    or not np.array_equal(saved["winning_fact_index"],
                        np.column_stack([winners[name] for name in concept.names]))
                    or not np.array_equal(saved["direction_signs"],
                        np.asarray(model.get_params()["monotone_constraints"]))
                    or np.max(np.abs(saved["probability"] - p)) > 1e-14
                    or np.max(np.abs(saved["logit"] - logit)) > 1e-13
                    or np.max(np.abs(saved["shap"] - shap)) > 1e-13
                    or np.max(np.abs(saved["group_contribution"] - contributions)) > 1e-13
                    or np.max(np.abs(saved["neighbor_mask_probability"] - neighbor_p)) > 1e-14
                    or np.max(np.abs(saved["all_mask_probability"] - all_p)) > 1e-14):
                raise ValueError("Source-to-concept-to-decision replay differs")
        if (np.max(np.abs(expit(logit) - p)) > 1e-12 or
                np.max(np.abs(contributions.sum(axis=1) + shap[:, -1] - logit)) > 1e-9):
            raise ValueError("Actual decision does not sum from source groups")
        probe = x[:64].copy()
        probe_p = np.asarray(model.booster_.predict(probe))
        signs = np.asarray(model.get_params()["monotone_constraints"])
        mono_violations = 0
        for column, sign in enumerate(signs):
            if sign == 0:
                continue
            changed = probe.copy()
            changed[:, column] += 0.1
            changed_p = np.asarray(model.booster_.predict(changed))
            mono_violations += int(np.count_nonzero(
                sign * (changed_p - probe_p) < -1e-12))
        if mono_violations:
            raise ValueError("A-derived monotone direction violated by fitted tree")
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        if (len(predictions["predictions"]) != len(test)
                or any(row["account_id"] != ids[int(i)] or
                       abs(row["probability"] - float(prob)) > 1e-14
                       for row, i, prob in zip(predictions["predictions"], test, p))):
            raise ValueError("Per-account predictions differ")
        threshold = lock["threshold_by_seed"][str(seed)]
        changed = int(np.count_nonzero((p >= threshold) != (neighbor_p >= threshold)))
        if changed != result["neighbor_group_mask_changed_classes"]:
            raise ValueError("Decisive group mask count differs")
        rows.append({"seed": seed, "account_count": len(test),
                     "fact_cells_replayed": len(test) * len(kept),
                     "concept_cells_replayed": len(test) * len(concept.names),
                     "neighbor_group_decision_changes": changed,
                     "monotone_probe_violations": mono_violations,
                     "max_probability_difference": 0.0,
                     "max_logit_additivity_error": float(np.max(np.abs(
                         contributions.sum(axis=1) + shap[:, -1] - logit)))})
    certificate = {
        "scope": "fox8_full_test_source_to_twenty_concepts_to_ab_group_tree_decision",
        "selection_lock_sha256": sha256_file(lock_path),
        "fixed_test_report_sha256": sha256_file(report_path),
        "raw_source_audit_sha256": sha256_file(audit_path),
        "rows": rows,
        "status": "all_accounts_concepts_group_contributions_and_masks_verified",
    }
    out = root / "research/certificates/fox8_targeted_20_concept_ab_tree_replay.json"
    out.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
