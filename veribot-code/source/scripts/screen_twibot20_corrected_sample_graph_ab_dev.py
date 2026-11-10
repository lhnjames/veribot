"""Refit the same A/B evidence tree after including all provided sampled graph edges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from lightgbm import Booster
import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import design, fit_tree, score_model, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-facts", type=Path, required=True)
    parser.add_argument("--corrected-facts", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--old-reference", type=Path, required=True)
    parser.add_argument("--old-group-screen", type=Path, required=True)
    parser.add_argument("--old-decision-cards", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--corrected-reference", type=Path, required=True)
    parser.add_argument("--seed0-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    old_screen = json.loads(args.old_group_screen.read_text(encoding="utf-8"))
    old_cards = json.loads(args.old_decision_cards.read_text(encoding="utf-8"))
    corrected_manifest = json.loads(args.corrected_facts.with_suffix(".json").read_text(encoding="utf-8"))
    if (sha256(args.old_facts) != audit["fact_cache_sha256"]
            or sha256(args.corrected_facts) != corrected_manifest["fact_cache_sha256"]
            or sha256(args.support_audit) != corrected_manifest["support_edge_audit_sha256"]
            or sha256(args.old_reference) != compact["reference_artifact_sha256"]
            or sha256(args.old_group_screen) != old_cards["group_screen_sha256"]
            or old_screen["facts_sha256"] != sha256(args.old_facts)):
        raise ValueError("Old/corrected source or prior model audit differs")
    with np.load(args.old_facts, allow_pickle=False) as old, \
            np.load(args.corrected_facts, allow_pickle=False) as revised:
        if any(not np.array_equal(old[key], revised[key])
               for key in old.files if key != "values"):
            raise ValueError("Corrected fact cache altered a non-value array")
        split = old["split"].astype(str)
        labels = old["labels"].astype(int)
        names_all = old["names"].astype(str).tolist()
        contexts = old["contexts"].astype(str)
        old_values = old["values"].astype(float)
        corrected = revised["values"].astype(float)
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    old_reference = LabelBlindReference.load(args.old_reference)
    reproduced = LabelBlindReference(
        min_context_n=old_reference.min_context_n,
        support_prior=old_reference.support_prior).fit(
            old_values[train], names_all, contexts[train])
    if reproduced.version != old_reference.version:
        raise ValueError("Original train-only reference cannot be reproduced")
    corrected_reference = LabelBlindReference(
        min_context_n=old_reference.min_context_n,
        support_prior=old_reference.support_prior).fit(
            corrected[train], names_all, contexts[train])
    corrected_reference.save(args.corrected_reference)
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("20-fact vocabulary differs")
    keep = [names_all.index(name) for name in names]
    old_batch = evidence_subset(
        old_reference.transform(old_values, contexts=contexts), keep, names,
        old_values[:, names_all.index("tweet_sample_count")], True)
    new_batch = evidence_subset(
        corrected_reference.transform(corrected, contexts=contexts), keep, names,
        corrected[:, names_all.index("tweet_sample_count")], True)
    groups = source_groups(names)
    rows = []
    for seed in (0, 1, 2):
        a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                random_state=1000 + seed)
        old_signs = DualTailEvidenceReasoner._select_tail_signs(take(old_batch, a), labels[a])
        new_signs = DualTailEvidenceReasoner._select_tail_signs(take(new_batch, a), labels[a])
        signs = np.r_[new_signs[:, 0], new_signs[:, 1]]
        model = fit_tree(design(take(new_batch, b), signs), labels[b],
                         groups, signs, True, seed)
        x_dev = design(take(new_batch, dev), signs)
        result = score_model(model, x_dev, labels[dev], groups)
        prior = next(row for row in old_screen["rows"]
                     if row["support_aware"] and row["ab_seed"] == seed)
        row = {
            "ab_seed": seed,
            "train_a": len(a), "train_b": len(b), "dev_accounts": len(dev),
            "changed_tail_directions_vs_old_graph": int(np.count_nonzero(old_signs != new_signs)),
            "old_labeled_only_graph_dev": prior["constrained_group_tree"]["dev"],
            "all_provided_sample_graph_dev": result["dev"],
            "corrected_threshold": result["threshold"],
            "corrected_group_additivity_error": result[
                "max_dev_logit_group_reconstruction_error"],
            "corrected_tree_counts_by_group": result["tree_counts_by_group"],
        }
        if seed == 0:
            args.seed0_model.parent.mkdir(parents=True, exist_ok=True)
            model.booster_.save_model(str(args.seed0_model))
            frozen = Booster(model_file=str(args.seed0_model))
            raw = np.asarray(frozen.predict(x_dev, raw_score=True))
            if np.max(np.abs(raw - model.booster_.predict(x_dev, raw_score=True))) > 1e-12:
                raise ValueError("Corrected graph seed0 model reload differs")
            p = expit(raw)
            predicted = p >= result["threshold"]
            prior_predicted = np.asarray([card["prediction"]
                                          for card in old_cards["decision_cards"]], dtype=bool)
            if len(prior_predicted) != len(dev):
                raise ValueError("Old dev prediction population differs")
            row["seed0_paired_with_old_labeled_only_graph"] = {
                "class_changes": int(np.count_nonzero(predicted != prior_predicted)),
                "old_only_correct": int(np.count_nonzero(
                    (prior_predicted == labels[dev]) & (predicted != labels[dev]))),
                "corrected_only_correct": int(np.count_nonzero(
                    (prior_predicted != labels[dev]) & (predicted == labels[dev]))),
                "corrected_dev_predictions": [
                    {"account_id": card["account_id"],
                     "logit": float(raw[k]), "probability": float(p[k]),
                     "prediction": int(predicted[k]),
                     "evaluation_label": int(labels[dev[k]])}
                    for k, card in enumerate(old_cards["decision_cards"])],
            }
        rows.append(row)
        print(json.dumps({"seed": seed,
                          "old_dev_macro_f1": round(row["old_labeled_only_graph_dev"]["macro_f1"], 5),
                          "corrected_dev_macro_f1": round(row["all_provided_sample_graph_dev"]["macro_f1"], 5),
                          "tail_sign_changes": row["changed_tail_directions_vs_old_graph"]}),
              flush=True)
    report = {
        "scope": "TwiBot-20 all-provided-sampled-graph refit of A/B support-aware group tree, official train/dev only",
        "candidate_test_evaluated": False,
        "old_fact_cache_sha256": sha256(args.old_facts),
        "corrected_fact_cache_sha256": sha256(args.corrected_facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "old_reference_sha256": sha256(args.old_reference),
        "corrected_reference_sha256": sha256(args.corrected_reference),
        "corrected_reference_version": corrected_reference.version,
        "old_group_screen_sha256": sha256(args.old_group_screen),
        "corrected_seed0_model_sha256": sha256(args.seed0_model),
        "method": "same 20 facts, n/(n+20), 3 prespecified A/B splits, fixed group-restricted monotone tree config and dev threshold selection; only source graph degrees and their train-only label-blind reference are corrected",
        "rows": rows,
        "interpretation": "All official support-row neighbor declarations are now included in sampled graph degrees. These are still sampled-neighbor degrees, not complete Twitter graph degrees. Dev reuse for threshold calibration remains; comparison does not establish independent test or OOD improvement.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
