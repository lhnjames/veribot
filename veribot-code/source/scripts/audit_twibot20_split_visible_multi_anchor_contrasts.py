"""Measure how frozen group reasons change across observed opposite-class anchors."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from lightgbm import Booster
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


QUANTILES = (0.1, 0.3, 0.5, 0.7, 0.9)
SIGN_TOLERANCE = 1e-10


def minimal_masks(classes: np.ndarray, target: bool) -> tuple[int, ...]:
    preserving = [mask for mask in range(16) if bool(classes[mask]) == target]
    if 0 in preserving or 15 not in preserving:
        raise ValueError("Observed opposite-anchor endpoints do not bracket the decision")
    return tuple(mask for mask in preserving
                 if not any(other != mask and (other & mask) == other
                            for other in preserving))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--contrast", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    contrast = json.loads(args.contrast.read_text(encoding="utf-8"))
    if (sha256(args.facts) != cards["fact_cache_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]
            or contrast["decision_cards_sha256"] != sha256(args.cards)
            or contrast["model_sha256"] != sha256(args.model)
            or cards["graph_scope"]
            != "train/support/dev sampled neighbor declarations without test"):
        raise ValueError("Multi-anchor inputs differ from split-visible frozen model")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    rows = cards["decision_cards"]
    if ids[dev].tolist() != [row["account_id"] for row in rows]:
        raise ValueError("Decision card population differs from facts")
    terms = rows[0]["fact_terms"]
    names = [term["fact_name"] for term in terms]
    signs = np.r_[[term["upper_monotone_direction"] for term in terms],
                  [term["lower_monotone_direction"] for term in terms]]
    ref = LabelBlindReference.load(args.reference)
    train_evidence = evidence_subset(
        ref.transform(values[train], contexts=contexts[train]),
        [all_names.index(name) for name in names], names,
        values[train, all_names.index("tweet_sample_count")], True)
    x_train = design(train_evidence, signs)
    x_dev = np.asarray([
        [term["upper_model_input"] for term in row["fact_terms"]]
        + [term["lower_model_input"] for term in row["fact_terms"]]
        for row in rows])
    model = Booster(model_file=str(args.model))
    train_logit = np.asarray(model.predict(x_train, raw_score=True))
    dev_logit = np.asarray(model.predict(x_dev, raw_score=True))
    if np.max(np.abs(dev_logit - [row["full_logit"] for row in rows])) > 1e-12:
        raise ValueError("Frozen dev scores differ from cards")
    threshold = float(rows[0]["threshold"])
    train_prediction = expit(train_logit) >= threshold
    dev_prediction = expit(dev_logit) >= threshold
    if not np.array_equal(dev_prediction, [row["prediction"] for row in rows]):
        raise ValueError("Frozen dev predictions differ from cards")
    groups = source_groups(names)
    group_names = list(GROUPS)
    anchors: dict[int, list[int]] = {}
    for prediction in (0, 1):
        class_members = np.flatnonzero(train_prediction == prediction)
        sorted_members = sorted(class_members,
                                key=lambda j: (float(train_logit[j]), str(ids[train[j]])))
        key = "predicted_bot" if prediction else "predicted_human"
        original_id = contrast["anchors"][key]["account_id"]
        medoid = next((j for j in class_members if ids[train[j]] == original_id), None)
        if medoid is None:
            raise ValueError("Prior contrast anchor is not in predicted class")
        selected = [int(medoid)]
        for q in QUANTILES:
            j = int(sorted_members[round(q * (len(sorted_members) - 1))])
            if j not in selected:
                selected.append(j)
        if len(selected) < 5:
            raise ValueError("Too few distinct opposite-prediction anchors")
        anchors[prediction] = selected
    account_anchors: list[list[dict]] = [[] for _ in rows]
    max_additive_error = 0.0
    evaluated = 0
    for target in (0, 1):
        subset = np.flatnonzero(dev_prediction == target)
        for anchor_index in anchors[1 - target]:
            anchor_x = x_train[anchor_index]
            anchor_logit = float(train_logit[anchor_index])
            mixed = []
            for mask in range(16):
                candidate = np.tile(anchor_x, (len(subset), 1))
                for j, columns in enumerate(groups):
                    if mask & (1 << j):
                        candidate[:, columns] = x_dev[subset][:, columns]
                mixed.append(candidate)
            logits = np.asarray(model.predict(np.concatenate(mixed), raw_score=True))
            logits = logits.reshape(16, len(subset))
            if (np.max(np.abs(logits[0] - anchor_logit)) > 1e-12
                    or np.max(np.abs(logits[15] - dev_logit[subset])) > 1e-12):
                raise ValueError("Opposite anchor or dev endpoint differs")
            deltas = np.stack([logits[1 << j] - logits[0]
                               for j in range(len(groups))])
            additive = np.stack([logits[0]
                                 + sum(deltas[j] for j in range(len(groups))
                                       if mask & (1 << j))
                                 for mask in range(16)])
            max_additive_error = max(max_additive_error,
                                     float(np.max(np.abs(logits - additive))))
            classes = expit(logits) >= threshold
            for local, index in enumerate(subset):
                minimal = minimal_masks(classes[:, local], bool(target))
                core = 15
                for mask in minimal:
                    core &= mask
                account_anchors[index].append({
                    "anchor_id": str(ids[train[anchor_index]]),
                    "anchor_logit": anchor_logit,
                    "minimal_masks": list(minimal),
                    "core_mask": core,
                    "group_deltas": [float(deltas[j, local])
                                     for j in range(len(groups))],
                })
            evaluated += 16 * len(subset)
    if max_additive_error > 1e-9:
        raise ValueError("Group logits are not additive across anchors")
    prior_rows = {row["account_id"]: row for row in contrast["rows"]}
    results = []
    summary = {"bot": Counter(), "human": Counter()}
    stable_core_counts = {"bot": Counter(), "human": Counter()}
    sign_variation = {"bot": Counter(), "human": Counter()}
    for i, card in enumerate(rows):
        reports = account_anchors[i]
        if len(reports) != len(anchors[int(not bool(dev_prediction[i]))]):
            raise ValueError("Account did not receive every opposite-class anchor")
        prior = prior_rows[card["account_id"]]
        original = reports[0]
        prior_masks = [sum(1 << group_names.index(name) for name in group_set)
                       for group_set in prior["minimal_sufficient_group_sets"]]
        if (original["anchor_id"] != prior["opposite_predicted_class_anchor_id"]
                or sorted(original["minimal_masks"]) != sorted(prior_masks)):
            raise ValueError("Prior single-anchor contrast does not replay")
        core_masks = [report["core_mask"] for report in reports]
        stable_core = 15
        possible_core = 0
        for core in core_masks:
            stable_core &= core
            possible_core |= core
        varying_sign_groups = []
        for j, name in enumerate(group_names):
            signs_seen = {0 if abs(report["group_deltas"][j]) <= SIGN_TOLERANCE
                          else int(np.sign(report["group_deltas"][j]))
                          for report in reports}
            if len(signs_seen) > 1:
                varying_sign_groups.append(name)
        key = "bot" if dev_prediction[i] else "human"
        summary[key]["accounts"] += 1
        summary[key]["core_changes_with_anchor"] += len(set(core_masks)) > 1
        summary[key]["minimal_sets_change_with_anchor"] += (
            len({tuple(report["minimal_masks"]) for report in reports}) > 1)
        summary[key]["no_stable_core_group"] += stable_core == 0
        summary[key]["at_least_one_anchor_core_group"] += possible_core != 0
        for j, name in enumerate(group_names):
            if stable_core & (1 << j):
                stable_core_counts[key][name] += 1
        for name in varying_sign_groups:
            sign_variation[key][name] += 1
        results.append({
            "account_id": card["account_id"],
            "prediction": int(dev_prediction[i]),
            "stable_core_groups_across_anchors": [
                name for j, name in enumerate(group_names) if stable_core & (1 << j)],
            "groups_core_for_at_least_one_anchor": [
                name for j, name in enumerate(group_names) if possible_core & (1 << j)],
            "core_changes_with_anchor": len(set(core_masks)) > 1,
            "minimal_sets_change_with_anchor": (
                len({tuple(report["minimal_masks"]) for report in reports}) > 1),
            "groups_whose_signed_effect_changes_across_anchors": varying_sign_groups,
            "opposite_anchor_results": reports,
        })
    report = {
        "scope": "split-visible seed0 frozen group tree, multiple observed opposite-prediction anchors",
        "candidate_test_evaluated": False,
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "decision_cards_sha256": sha256(args.cards),
        "prior_contrast_sha256": sha256(args.contrast),
        "model_sha256": sha256(args.model),
        "anchor_selection": "for each frozen-model-predicted train class: prior class-conditional input medoid plus distinct training accounts at logit quantile ranks 0.1/0.3/0.5/0.7/0.9; no ground-truth label read for anchor selection",
        "anchors_by_prediction": {
            ("bot" if prediction else "human"): [
                {"account_id": str(ids[train[j]]), "logit": float(train_logit[j])}
                for j in anchors[prediction]] for prediction in (0, 1)},
        "groups": group_names,
        "signed_effect_zero_tolerance": SIGN_TOLERANCE,
        "direct_frozen_mixed_predictions": evaluated,
        "max_group_additivity_error": max_additive_error,
        "counts_by_dev_prediction": {key: dict(value) for key, value in summary.items()},
        "stable_core_group_counts": {key: dict(value)
                                     for key, value in stable_core_counts.items()},
        "signed_effect_changes_across_anchors": {key: dict(value)
                                                 for key, value in sign_variation.items()},
        "rows": results,
        "interpretation": "An opposite-prediction real training account prevents an empty explanation set, but core and minimal input-group sets can change when the real comparison account changes. These are frozen-model mixed-input contrasts, not feasible source-level interventions or unconditional unique reasons.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
