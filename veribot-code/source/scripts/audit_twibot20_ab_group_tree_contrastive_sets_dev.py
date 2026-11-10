"""Enumerate all source-group contrasts against observed opposite-prediction anchors."""

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
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def choose_anchor(x: np.ndarray, candidate_indices: np.ndarray) -> int:
    selected = x[candidate_indices]
    median = np.median(selected, axis=0)
    distance = np.sum((selected - median) ** 2, axis=1)
    return int(candidate_indices[np.argmin(distance)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corrected-screen", action="store_true",
                        help="use all-provided sampled-graph refit and cards")
    parser.add_argument("--split-visible-screen", action="store_true",
                        help="use split-visible sampled-graph refit and cards")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    if args.corrected_screen and args.split_visible_screen:
        raise ValueError("Choose one graph visibility protocol")
    if (sha256(args.facts) != cards["fact_cache_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]
            or sha256(args.group_screen) != cards["group_screen_sha256"]
            or (not (args.corrected_screen or args.split_visible_screen) and
                screen["compact_sha256"] != sha256(args.compact))):
        raise ValueError("Frozen source, model, or decision cards differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names_all = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if args.corrected_screen or args.split_visible_screen:
        if (screen["corrected_fact_cache_sha256" if args.corrected_screen
                   else "split_visible_facts_sha256"] != sha256(args.facts)
                or screen["corrected_reference_sha256" if args.corrected_screen
                          else "reference_sha256"] != sha256(args.reference)
                or screen["corrected_seed0_model_sha256" if args.corrected_screen
                          else "seed0_model_sha256"] != sha256(args.model)):
            raise ValueError("Source-visible screen does not bind source and model")
        terms = cards["decision_cards"][0]["fact_terms"]
        if [term["fact_name"] for term in terms] != names:
            raise ValueError("Corrected card fact order differs")
        signs = np.r_[[term["upper_monotone_direction"] for term in terms],
                      [term["lower_monotone_direction"] for term in terms]]
    else:
        selected = next(row for row in screen["rows"]
                        if row["support_aware"] and row["ab_seed"] == cards["a_seed"])
        signs = np.r_[[selected["a_signs_by_fact"][name]["upper"] for name in names],
                      [selected["a_signs_by_fact"][name]["lower"] for name in names]]
    full = LabelBlindReference.load(args.reference).transform(values, contexts=contexts)
    support = values[:, names_all.index("tweet_sample_count")]
    batch = evidence_subset(full, [names_all.index(name) for name in names],
                            names, support, True)
    if batch.reference_version != cards["reference_version_with_support_rule"]:
        raise ValueError("Support-aware reference differs")
    x_train = design(take(batch, train), signs)
    x_dev = design(take(batch, dev), signs)
    model = Booster(model_file=str(args.model))
    threshold_logit = float(cards["decision_cards"][0]["threshold_logit"])
    threshold = float(cards["decision_cards"][0]["threshold"])
    train_logit = np.asarray(model.predict(x_train, raw_score=True))
    train_pred = expit(train_logit) >= threshold
    anchor_local = {
        0: choose_anchor(x_train, np.flatnonzero(~train_pred)),
        1: choose_anchor(x_train, np.flatnonzero(train_pred)),
    }
    anchor_x = {label: x_train[index] for label, index in anchor_local.items()}
    anchor_logit = {label: float(train_logit[index])
                    for label, index in anchor_local.items()}
    card_rows = cards["decision_cards"]
    if ids[dev].tolist() != [row["account_id"] for row in card_rows]:
        raise ValueError("Dev account order differs")
    full_logit = np.asarray(model.predict(x_dev, raw_score=True))
    if np.max(np.abs(full_logit - [row["full_logit"] for row in card_rows])) > 1e-12:
        raise ValueError("Frozen dev scores differ from decision cards")
    prediction = expit(full_logit) >= threshold
    if not np.array_equal(prediction, [row["prediction"] for row in card_rows]):
        raise ValueError("Frozen dev classes differ from decision cards")
    opposite_anchor = np.asarray([anchor_x[int(not label)] for label in prediction])
    opposite_logit = np.asarray([anchor_logit[int(not label)] for label in prediction])
    if np.any((expit(opposite_logit) >= threshold) == prediction):
        raise ValueError("An anchor does not predict the opposite class")
    groups = source_groups(names)
    mixed = []
    for mask in range(1 << len(groups)):
        candidate = opposite_anchor.copy()
        for j, cols in enumerate(groups):
            if mask & (1 << j):
                candidate[:, cols] = x_dev[:, cols]
        mixed.append(candidate)
    mixed_x = np.concatenate(mixed, axis=0)
    mixed_logit = np.asarray(model.predict(mixed_x, raw_score=True)).reshape(
        len(mixed), len(dev))
    if (np.max(np.abs(mixed_logit[0] - opposite_logit)) > 1e-12
            or np.max(np.abs(mixed_logit[-1] - full_logit)) > 1e-12):
        raise ValueError("Mixed model inputs do not replay both endpoints")
    group_delta = np.stack([mixed_logit[1 << j] - mixed_logit[0]
                            for j in range(len(groups))])
    reconstructed = np.stack([
        mixed_logit[0] + sum(group_delta[j] for j in range(len(groups))
                             if mask & (1 << j))
        for mask in range(1 << len(groups))])
    max_error = float(np.max(np.abs(mixed_logit - reconstructed)))
    if max_error > 1e-9:
        raise ValueError("Exhaustive mixed group logits are not additive")
    mixed_class = expit(mixed_logit) >= threshold
    rows = []
    by_prediction = {"bot": Counter(), "human": Counter()}
    group_cores = {"bot": Counter(), "human": Counter()}
    for k, card in enumerate(card_rows):
        target = bool(prediction[k])
        preserving = [mask for mask in range(16) if mixed_class[mask, k] == target]
        if 0 in preserving or 15 not in preserving:
            raise ValueError("Opposite-anchor contrast endpoints differ")
        minimal = [mask for mask in preserving
                   if not any(other != mask and (other & mask) == other
                              for other in preserving)]
        if not minimal:
            raise ValueError("No minimal sufficient group combination")
        required = set(range(len(groups)))
        for mask in minimal:
            required &= {j for j in range(len(groups)) if mask & (1 << j)}
        key = "bot" if target else "human"
        by_prediction[key]["accounts"] += 1
        by_prediction[key]["unique_minimal" if len(minimal) == 1
                           else "multiple_minimal"] += 1
        by_prediction[key][f"smallest_size_{min(mask.bit_count() for mask in minimal)}"] += 1
        by_prediction[key][f"largest_size_{max(mask.bit_count() for mask in minimal)}"] += 1
        by_prediction[key]["at_least_one_core_group" if required else "no_core_group"] += 1
        for j in required:
            group_cores[key][list(GROUPS)[j]] += 1
        rows.append({
            "account_id": card["account_id"],
            "prediction": int(target),
            "opposite_predicted_class_anchor_id": str(ids[train[anchor_local[int(not target)]]]),
            "anchor_logit": float(opposite_logit[k]),
            "full_logit": float(full_logit[k]),
            "group_deltas_from_opposite_anchor": {
                name: float(group_delta[j, k]) for j, name in enumerate(GROUPS)},
            "minimal_sufficient_group_sets": [
                [name for j, name in enumerate(GROUPS) if mask & (1 << j)]
                for mask in minimal],
            "groups_in_every_minimal_set": [list(GROUPS)[j] for j in sorted(required)],
            "all_16_mixed_logits": [float(mixed_logit[mask, k]) for mask in range(16)],
        })
    report = {
        "scope": "all 2365 frozen seed0 group-tree dev decisions; label-blind observed opposite-prediction contrast"
                 + (", split-visible sampled graph" if args.split_visible_screen
                    else ", all provided sampled graph" if args.corrected_screen
                    else ", labeled-only sampled graph"),
        "candidate_test_evaluated": False,
        "decision_cards_sha256": sha256(args.decision_cards),
        "model_sha256": sha256(args.model),
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "anchor_selection": "within each frozen-model-predicted class on official train, select observed account closest to that predicted class's 40-coordinate model-input median; do not read ground-truth labels",
        "anchors": {
            "predicted_human": {"account_id": str(ids[train[anchor_local[0]]]),
                                "source_locator": f"Twibot-20/train.json[{positions[train[anchor_local[0]]]}]",
                                "logit": anchor_logit[0],
                                "margin": anchor_logit[0] - threshold_logit},
            "predicted_bot": {"account_id": str(ids[train[anchor_local[1]]]),
                              "source_locator": f"Twibot-20/train.json[{positions[train[anchor_local[1]]]}]",
                              "logit": anchor_logit[1],
                              "margin": anchor_logit[1] - threshold_logit},
        },
        "groups": list(GROUPS),
        "mixed_inputs_per_account": 16,
        "total_direct_frozen_mixed_predictions": len(dev) * 16,
        "max_exhaustive_group_additivity_error": max_error,
        "counts_by_original_prediction": {key: dict(value)
                                          for key, value in by_prediction.items()},
        "groups_in_all_minimal_sets_counts": {key: dict(value)
                                             for key, value in group_cores.items()},
        "rows": rows,
        "interpretation": "Every contrast starts from a real account that this frozen model predicts as the opposite class, so the empty set cannot already give the target decision. Minimal sets describe model-input substitutions between two observed accounts; their mixed combinations may not be realizable social records. They do not establish source-level causal influence, behavioral truth, or a unique unconditional reason.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
