"""Dev-only capacity screen: 20 grounded Fox8 concepts versus 74 atomic facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_fox8_neighbor_atomic_evidence_capacity_dev import (
    CONFIGS, evidence_matrix, load_cache, weighted_atomic)
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def screen(train_matrix: np.ndarray, dev_matrix: np.ndarray,
           train_labels: np.ndarray, dev_labels: np.ndarray,
           group_names: list[str]) -> list[dict]:
    results = []
    for leaves, rounds, min_leaf, l2 in CONFIGS:
        model = LGBMClassifier(
            objective="binary", n_estimators=rounds,
            num_leaves=leaves, min_child_samples=min_leaf,
            reg_lambda=l2, learning_rate=0.05,
            n_jobs=4, random_state=0, deterministic=True,
            force_col_wise=True, verbosity=-1)
        model.fit(train_matrix, train_labels)
        probability = model.predict_proba(dev_matrix)[:, 1]
        threshold = best_validation_threshold(probability, dev_labels)
        raw = np.asarray(model.booster_.predict(dev_matrix[:32], raw_score=True))
        contribution = np.asarray(model.booster_.predict(
            dev_matrix[:32], pred_contrib=True))
        if (contribution.shape != (32, train_matrix.shape[1] + 1)
                or np.max(np.abs(contribution.sum(axis=1) - raw)) > 1e-9):
            raise ValueError("concept tree evidence does not reconstruct score")
        result = {
            "config": [leaves, rounds, min_leaf, l2],
            "validation_at_half": evaluate(dev_labels, probability, 0.5),
            "selected_threshold": threshold,
            "validation_selected": evaluate(dev_labels, probability, threshold),
            "max_probe_logit_additivity_error": float(np.max(np.abs(
                contribution.sum(axis=1) - raw))),
        }
        neighbor_groups = [name for name in group_names
                           if name.startswith("native_neighbor_")]
        if len(neighbor_groups) == 1:
            group_index = group_names.index(neighbor_groups[0])
            masked = dev_matrix.copy()
            masked[:, 2 * group_index:2 * group_index + 2] = 0.0
            masked[:, 2 * group_index + 20:2 * group_index + 22] = 0.0
            masked_p = model.predict_proba(masked)[:, 1]
            result["neighbor_group_zero_changed_dev_classes"] = int(
                np.count_nonzero((probability >= threshold) !=
                                 (masked_p >= threshold)))
        results.append(result)
    return results


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    raw_path = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    source_audit_path = (root /
        "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json")
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if (source_audit["status"] != "sampled_raw_source_neighbor_means_verified"
            or source_audit["fact_artifact_sha256"] != sha256_file(raw_path)):
        raise ValueError("Fox8 native neighbor facts lack source audit")
    ids, new_names, new_values, supports = load_cache(raw_path)
    if len(new_names) != 31:
        raise ValueError("native neighbor fact count differs")
    prior_path = root / "output/v3_fox8_native_neighbor_profile_concepts/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    atomic_path = root / "output/v3_fox8_neighbor_atomic_evidence_capacity/dev_only.json"
    atomic = json.loads(atomic_path.read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None,
        structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    old_groups = json.loads((root / "research/fox8_20_concept_groups.json")
                            .read_text(encoding="utf-8"))
    removed = set(old_groups["presentation_flags"])
    candidate_groups = {group: members for group, members in old_groups.items()
                        if group != "presentation_flags"}
    candidate_groups["native_neighbor_profile"] = new_names
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        facts, labels, names, split = selected_facts(args, seed, old)
        if (len(names) != 43 or prior["runs"][seed]["split_sha256"] !=
                old["manifest"]["split_sha256"] or
                atomic["runs"][seed]["split_sha256"] !=
                old["manifest"]["split_sha256"]):
            raise ValueError("official Fox8 split or source vocabulary differs")
        import pandas as pd
        if pd.read_csv(args.data_dir / "features.csv")[
                "user_id"].astype(str).tolist() != ids:
            raise ValueError("native neighbor cache account order differs")
        all_facts = np.column_stack((facts, new_values))
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            all_facts[train], names + new_names)
        train_atomic = weighted_atomic(reference, all_facts[train],
                                       supports[train], 43)
        dev_atomic = weighted_atomic(reference, all_facts[dev],
                                     supports[dev], 43)
        control_train, _ = collapse_group_tails(
            columns(train_atomic, list(range(43))), old_groups)
        control_dev, _ = collapse_group_tails(
            columns(dev_atomic, list(range(43))), old_groups)
        kept = [j for j, name in enumerate(names + new_names)
                if name not in removed]
        candidate_train, _ = collapse_group_tails(
            columns(train_atomic, kept), candidate_groups)
        candidate_dev, _ = collapse_group_tails(
            columns(dev_atomic, kept), candidate_groups)
        if (len(control_train.names) != 20 or len(candidate_train.names) != 20
                or tuple(control_train.names) != tuple(control_dev.names)
                or tuple(candidate_train.names) != tuple(candidate_dev.names)):
            raise ValueError("20 concept bottleneck differs")
        control = screen(evidence_matrix(control_train),
                         evidence_matrix(control_dev), labels[train],
                         labels[dev], list(old_groups))
        candidate = screen(evidence_matrix(candidate_train),
                           evidence_matrix(candidate_dev), labels[train],
                           labels[dev], list(candidate_groups))
        old_result = prior["runs"][seed]
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "concept_count": 20,
            "old_ab_spline_dev": old_result["control"]["validation"],
            "neighbor_ab_spline_dev": old_result["candidate"]["validation"],
            "control_tree": control,
            "neighbor_tree": candidate,
            "atomic_74_best_dev": atomic["runs"][seed]["variants"][
                "raw_neighbor_74"]["model_results"][0][
                    "validation_at_half"],
        })
        print(json.dumps({
            "seed": seed,
            "control_errors": [row["validation_selected"]["errors"]
                               for row in control],
            "neighbor_errors": [row["validation_selected"]["errors"]
                                for row in candidate],
        }), flush=True)
    output = {
        "scope": "fox8_official_train_dev_only_twenty_concept_tree_capacity",
        "test_predictions_or_metrics_computed": False,
        "native_neighbor_fact_sha256": sha256_file(raw_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "prior_ab_concept_dev_sha256": sha256_file(prior_path),
        "atomic_74_dev_sha256": sha256_file(atomic_path),
        "configs": CONFIGS,
        "runs": runs,
    }
    path = root / "output/v3_fox8_neighbor_20_concept_tree/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
