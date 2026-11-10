"""Compare zero-vector and observed-account references for a frozen group tree."""

from __future__ import annotations

import argparse
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
        raise ValueError("Frozen model, sources, or decision cards differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
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
    reference = LabelBlindReference.load(args.reference)
    full = reference.transform(values, contexts=contexts)
    support = values[:, all_names.index("tweet_sample_count")]
    batch = evidence_subset(full, [all_names.index(name) for name in names],
                            names, support, True)
    if batch.reference_version != cards["reference_version_with_support_rule"]:
        raise ValueError("Support-aware reference rule differs")
    x_train = design(take(batch, train), signs)
    x_dev = design(take(batch, dev), signs)
    card_rows = cards["decision_cards"]
    if ids[dev].tolist() != [card["account_id"] for card in card_rows]:
        raise ValueError("Dev account order differs")
    from_cards = np.asarray([[term["upper_model_input"] for term in card["fact_terms"]] +
                             [term["lower_model_input"] for term in card["fact_terms"]]
                             for card in card_rows])
    if np.max(np.abs(from_cards - x_dev)) > 1e-12:
        raise ValueError("Dev evidence inputs do not replay decision cards")
    model = Booster(model_file=str(args.model))
    full_logit = np.asarray(model.predict(x_dev, raw_score=True))
    if np.max(np.abs(full_logit - [card["full_logit"] for card in card_rows])) > 1e-12:
        raise ValueError("Frozen model does not replay all dev decisions")
    median = np.median(x_train, axis=0)
    nearest = int(np.argmin(np.sum((x_train - median) ** 2, axis=1)))
    anchor_index = int(train[nearest])
    anchor_id = str(ids[anchor_index])
    anchor_x = x_train[nearest]
    anchor_logit = float(model.predict(anchor_x[None, :], raw_score=True)[0])
    zero_logit = np.asarray(model.predict(np.zeros_like(x_dev), raw_score=True))
    if np.max(np.abs(zero_logit - [card["all_zero_input_logit"] for card in card_rows])) > 1e-12:
        raise ValueError("Zero reference does not replay decision cards")
    groups = source_groups(names)
    anchor_effect = {}
    anchor_replaced = {}
    for group_name, cols in zip(GROUPS, groups, strict=True):
        replaced = x_dev.copy()
        replaced[:, cols] = anchor_x[cols]
        anchor_replaced[group_name] = np.asarray(model.predict(replaced, raw_score=True))
        anchor_effect[group_name] = full_logit - anchor_replaced[group_name]
    reconstruction = float(np.max(np.abs(full_logit - anchor_logit -
                                          sum(anchor_effect.values()))))
    if reconstruction > 1e-9:
        raise ValueError("Observed-anchor group effects do not reconstruct logits")
    threshold_logit = float(card_rows[0]["threshold_logit"])
    threshold = float(card_rows[0]["threshold"])
    rows = []
    group_counts = {}
    for group in GROUPS:
        zero_effect = np.asarray([card["group_logit_effects"][group]
                                  for card in card_rows])
        observed_effect = anchor_effect[group]
        group_counts[group] = {
            "nonzero_sign_reversals": int(np.count_nonzero(
                (zero_effect * observed_effect) < -1e-12)),
            "zero_replacement_class_changes": int(np.count_nonzero(
                (expit(np.asarray([card["group_zeroed_logits"][group]
                                   for card in card_rows])) >= threshold)
                != (expit(full_logit) >= threshold))),
            "observed_anchor_replacement_class_changes": int(np.count_nonzero(
                (expit(anchor_replaced[group]) >= threshold)
                != (expit(full_logit) >= threshold))),
        }
    for k, card in enumerate(card_rows):
        rows.append({
            "account_id": card["account_id"],
            "prediction": card["prediction"],
            "evaluation_label": card["evaluation_label"],
            "zero_reference_group_effects": card["group_logit_effects"],
            "observed_anchor_group_effects": {
                group: float(anchor_effect[group][k]) for group in GROUPS},
            "observed_anchor_group_replacement_logits": {
                group: float(anchor_replaced[group][k]) for group in GROUPS},
            "full_logit": float(full_logit[k]),
            "zero_baseline_margin": float(zero_logit[k] - threshold_logit),
            "observed_anchor_baseline_margin": anchor_logit - threshold_logit,
        })
    report = {
        "scope": "all frozen seed0 support-aware group-tree dev decisions, two explanation references"
                 + (", split-visible sampled graph" if args.split_visible_screen
                    else ", all provided sampled graph" if args.corrected_screen
                    else ", labeled-only sampled graph"),
        "candidate_test_evaluated": False,
        "decision_cards_sha256": sha256(args.decision_cards),
        "model_sha256": sha256(args.model),
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "anchor_selection": "training account with smallest Euclidean distance in 40 model-input coordinates to coordinate-wise train median; labels ignored",
        "observed_anchor_account_id": anchor_id,
        "observed_anchor_source_locator": f"Twibot-20/train.json[{positions[anchor_index]}]",
        "observed_anchor_sampled_tweets": int(support[anchor_index]),
        "observed_anchor_logit": anchor_logit,
        "observed_anchor_margin": anchor_logit - threshold_logit,
        "zero_reference_logit": float(zero_logit[0]),
        "zero_reference_margin": float(zero_logit[0] - threshold_logit),
        "group_counts": group_counts,
        "max_observed_anchor_group_reconstruction_error": reconstruction,
        "rows": rows,
        "interpretation": "Changing an explanation reference leaves the frozen prediction unchanged. Group replacement is a model-input substitution, not a lawful source correction. The observed anchor is a real training account selected without labels, not a normative Human account.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
