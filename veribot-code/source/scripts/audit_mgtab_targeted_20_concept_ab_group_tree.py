"""Replay the compact MGTAB source-to-concept-to-decision path independently."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_content
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_targeted_20_concept_ab_group_tree_dev import grouped
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    lock_path, report_path = folder / "selection_lock.json", folder / "fixed_test_report.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["selection_lock_sha256"] != sha256_file(lock_path):
        raise ValueError("MGTAB compact candidate lock differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    if report["source_features_sha256"] != sha256_file(data_dir / "features.pt"):
        raise ValueError("Published MGTAB feature source differs")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    rows = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        train, test = split["train"], split["test"]
        row = report["runs"][seed]
        if old["manifest"]["split_sha256"] != lock["split_sha256_by_seed"][str(seed)]:
            raise ValueError("Official split differs")
        run_dir = folder / f"seed{seed}"
        model_path, ref_path = run_dir / "model.joblib", run_dir / "reference.json.gz"
        saved_path, pred_path = run_dir / "test_evidence.npz", run_dir / "test_predictions.json"
        for key, path in (("model_sha256", model_path), ("reference_sha256", ref_path),
                          ("evidence_sha256", saved_path), ("predictions_sha256", pred_path)):
            if row[key] != sha256_file(path):
                raise ValueError("Frozen MGTAB artifact fingerprint differs")
        reference = LabelBlindReference.load(ref_path)
        fresh = LabelBlindReference(min_context_n=30).fit(facts[train], names)
        if reference.version != fresh.version:
            raise ValueError("Label-blind reference differs from train facts")
        a, _ = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        atomic_train = transform_with_support(reference, facts[train],
                                              split["supports"][train])
        selected_content = rank_content(atomic_train, labels[train], a, count=4)
        if [names[j] for j in selected_content] != lock["selected_content_by_seed"][str(seed)]:
            raise ValueError("A-only content ranking differs")
        kept, groups = grouped(names, lock["selected"]["scheme"], selected_content)
        train_concept, _ = collapse_group_tails(columns(atomic_train, kept), groups)
        atomic_test = transform_with_support(reference, facts[test],
                                             split["supports"][test])
        test_concept, winners = collapse_group_tails(columns(atomic_test, kept), groups)
        design = evidence_matrix(test_concept)
        directions_a = directions(evidence_matrix(train_concept)[a], labels[train][a])
        model = joblib.load(model_path)
        if not np.array_equal(np.asarray(model.get_params()["monotone_constraints"]),
                              directions_a):
            raise ValueError("A-derived monotone directions differ")
        group_cols = [[2*k, 2*k+1, 20+2*k, 21+2*k] for k in range(10)]
        one_group, counts = all_trees_one_group(model, group_cols)
        if not one_group or {str(k): v for k, v in counts.items()} != row["tree_count_by_group"]:
            raise ValueError("Source-group tree structure differs")
        probability = model.predict_proba(design)[:, 1]
        logit = np.asarray(model.booster_.predict(design, raw_score=True))
        shap = np.asarray(model.booster_.predict(design, pred_contrib=True))
        group_shap = np.column_stack([shap[:, cols].sum(axis=1)
                                      for cols in group_cols])
        zero = np.zeros_like(design)
        zero_logit = np.asarray(model.booster_.predict(zero, raw_score=True))
        zero_p = model.predict_proba(zero)[:, 1]
        mask_p = np.empty((len(test), 10))
        delta = np.empty((len(test), 10))
        for k, cols in enumerate(group_cols):
            masked = design.copy()
            masked[:, cols] = 0
            mask_p[:, k] = model.predict_proba(masked)[:, 1]
            delta[:, k] = logit - np.asarray(model.booster_.predict(masked, raw_score=True))
        with np.load(saved_path, allow_pickle=False) as saved:
            if (saved["ids"].astype(str).tolist() != [str(int(i)) for i in test]
                    or saved["retained_fact_names"].astype(str).tolist() !=
                    [names[j] for j in kept]
                    or not np.allclose(saved["retained_raw_values"],
                                       facts[test][:, kept], rtol=0, atol=0, equal_nan=True)
                    or not np.array_equal(saved["neighbor_supports"],
                                           split["supports"][test])
                    or saved["concept_names"].astype(str).tolist() != list(test_concept.names)
                    or not np.array_equal(saved["concept_scores"], test_concept.upper)
                    or not np.array_equal(saved["concept_winners"],
                        np.column_stack([winners[name] for name in test_concept.names]))
                    or not np.array_equal(saved["evidence_matrix"], design)
                    or not np.array_equal(saved["direction_signs"], directions_a)
                    or np.max(np.abs(saved["probability"] - probability)) > 1e-14
                    or np.max(np.abs(saved["logit"] - logit)) > 1e-13
                    or np.max(np.abs(saved["shap"] - shap)) > 1e-13
                    or np.max(np.abs(saved["group_shap"] - group_shap)) > 1e-13
                    or np.max(np.abs(saved["exact_group_delta"] - delta)) > 1e-13
                    or np.max(np.abs(saved["group_mask_probability"] - mask_p)) > 1e-14
                    or np.max(np.abs(saved["all_mask_probability"] - zero_p)) > 1e-14):
                raise ValueError("Published fact to concept to decision replay differs")
        if (np.max(np.abs(expit(logit) - probability)) > 1e-12
                or np.max(np.abs(shap.sum(axis=1) - logit)) > 1e-9
                or np.max(np.abs(logit - zero_logit - delta.sum(axis=1))) > 1e-9):
            raise ValueError("Frozen source-group score does not reconstruct")
        predictions = json.loads(pred_path.read_text(encoding="utf-8"))
        if (len(predictions["predictions"]) != len(test)
                or any(item["account_id"] != str(int(i)) or
                       abs(item["probability"] - float(p)) > 1e-14
                       for item, i, p in zip(predictions["predictions"], test, probability))):
            raise ValueError("Per-account prediction file differs")
        threshold = lock["threshold_by_seed"][str(seed)]
        changes = {name: int(np.count_nonzero(
            (mask_p[:, k] >= threshold) != (probability >= threshold)))
            for k, name in enumerate(groups)}
        if changes != row["group_mask_changed_classes"]:
            raise ValueError("Group-deletion decision changes differ")
        probe = design[:64]
        probe_base = np.asarray(model.booster_.predict(probe))
        violations = 0
        for k, sign in enumerate(directions_a):
            if sign:
                changed = probe.copy()
                changed[:, k] += 0.1
                violations += int(np.count_nonzero(
                    sign * (np.asarray(model.booster_.predict(changed)) - probe_base) < -1e-12))
        if violations:
            raise ValueError("Compact tree violates A-derived monotone direction")
        rows.append({"seed": seed, "accounts_replayed": len(test),
                     "retained_fact_cells_replayed": len(test) * len(kept),
                     "concept_cells_replayed": len(test) * len(test_concept.names),
                     "group_masks_replayed": len(test) * len(groups),
                     "monotone_probe_violations": violations,
                     "max_probability_difference": 0.0,
                     "max_score_group_additivity_error": float(np.max(np.abs(
                         logit - zero_logit - delta.sum(axis=1))))})
    cert = {"scope": "MGTAB_all_test_published_source_to_20_concept_decision_replay",
            "selection_lock_sha256": sha256_file(lock_path),
            "fixed_test_report_sha256": sha256_file(report_path),
            "content_source_level": "published_average_tweet_tensor_only",
            "runs": rows,
            "status": "all_test_facts_concepts_monotonicity_and_group_masks_verified"}
    path = root / "research/certificates/mgtab_targeted_20_concept_ab_group_tree_replay.json"
    path.write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
