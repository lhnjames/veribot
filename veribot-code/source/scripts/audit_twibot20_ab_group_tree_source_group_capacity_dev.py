"""Dev-only retrained source-group capacity controls for the A/B evidence tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_mgtab_ab_group_additive_monotone_tree_dev import all_trees_one_group
from screen_twibot20_compact20_ab_group_tree_dev import design, fit_tree, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import metric, sha256, threshold_on_dev
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def subset_groups(full_groups: list[list[int]], selected_group_indices: list[int]) -> tuple[list[int], list[list[int]]]:
    columns = sorted(j for group_index in selected_group_indices
                     for j in full_groups[group_index])
    old_to_new = {old: new for new, old in enumerate(columns)}
    groups = [[old_to_new[j] for j in full_groups[group_index]]
              for group_index in selected_group_indices]
    if sorted(j for group in groups for j in group) != list(range(len(columns))):
        raise ValueError("Reduced groups do not partition the retained tails")
    return columns, groups


def evaluate_candidate(x_b: np.ndarray, y_b: np.ndarray,
                       x_dev: np.ndarray, y_dev: np.ndarray,
                       groups: list[list[int]], signs: np.ndarray,
                       seed: int) -> tuple[dict, np.ndarray]:
    model = fit_tree(x_b, y_b, groups, signs, True, seed)
    single_group, trees = all_trees_one_group(model, groups)
    if not single_group:
        raise ValueError("A retrained tree crosses evidence source groups")
    p = expit(np.asarray(model.booster_.predict(x_dev, raw_score=True)))
    threshold = threshold_on_dev(p, y_dev)
    prediction = p >= threshold
    return {"dev": metric(y_dev, p, threshold),
            "threshold": threshold,
            "tree_counts_by_retained_group_index": {
                str(index): trees.get(index, 0) for index in range(len(groups))}}, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--corrected-screen", action="store_true",
                        help="read the all-provided sampled-graph refit certificate")
    parser.add_argument("--split-visible-screen", action="store_true",
                        help="read the split-visible sampled-graph development certificate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    if args.corrected_screen and args.split_visible_screen:
        raise ValueError("Choose one source graph visibility protocol")
    if (sha256(args.facts) != screen[
            "corrected_fact_cache_sha256" if args.corrected_screen else
            "split_visible_facts_sha256" if args.split_visible_screen else "facts_sha256"]
            or sha256(args.reference) != screen[
                "corrected_reference_sha256" if args.corrected_screen else "reference_sha256"]
            or (not args.corrected_screen and sha256(args.compact) != screen["compact_sha256"])
            or sha256(args.group_screen) != cards["group_screen_sha256"]):
        raise ValueError("Source or previous group-tree screen differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("20-fact evidence vocabulary differs")
    sample_count = values[:, all_names.index("tweet_sample_count")]
    full = LabelBlindReference.load(args.reference).transform(values, contexts=contexts)
    batch = evidence_subset(full, [all_names.index(name) for name in names],
                            names, sample_count, True)
    groups = source_groups(names)
    rows = []
    for seed in (0, 1, 2):
        a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                random_state=1000 + seed)
        signs = DualTailEvidenceReasoner._select_tail_signs(take(batch, a), labels[a])
        signs = np.r_[signs[:, 0], signs[:, 1]]
        xb_full = design(take(batch, b), signs)
        xd_full = design(take(batch, dev), signs)
        baseline, baseline_prediction = evaluate_candidate(
            xb_full, labels[b], xd_full, labels[dev], groups, signs, seed)
        locked = next(row for row in screen["rows"]
                      if row["ab_seed"] == seed and
                      (args.corrected_screen or args.split_visible_screen
                       or row["support_aware"]))
        locked_threshold = (locked["corrected_threshold"] if args.corrected_screen
                            else locked["threshold"] if args.split_visible_screen
                            else locked["constrained_group_tree"]["threshold"])
        locked_dev = (locked["all_provided_sample_graph_dev"] if args.corrected_screen
                      else locked["dev"] if args.split_visible_screen
                      else locked["constrained_group_tree"]["dev"])
        if (abs(baseline["threshold"] - locked_threshold) > 1e-12
                or any(abs(baseline["dev"][key] - locked_dev[key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError("Full group model does not replay prior screen")
        if seed == 0 and not np.array_equal(
                baseline_prediction,
                [card["prediction"] for card in cards["decision_cards"]]):
            raise ValueError("Seed0 frozen decisions differ from cards")
        variants = {}
        for group_index, group_name in enumerate(GROUPS):
            for kind, selected in (("without", [j for j in range(len(groups))
                                                if j != group_index]),
                                   ("only", [group_index])):
                cols, reduced_groups = subset_groups(groups, selected)
                result, candidate_prediction = evaluate_candidate(
                    xb_full[:, cols], labels[b], xd_full[:, cols], labels[dev],
                    reduced_groups, signs[cols], seed)
                result["retained_source_groups"] = [list(GROUPS)[j] for j in selected]
                result["retained_tail_slots"] = len(cols)
                result["full_only_correct"] = int(np.count_nonzero(
                    (baseline_prediction == labels[dev]) &
                    (candidate_prediction != labels[dev])))
                result["variant_only_correct"] = int(np.count_nonzero(
                    (baseline_prediction != labels[dev]) &
                    (candidate_prediction == labels[dev])))
                result["class_changes_vs_full"] = int(np.count_nonzero(
                    baseline_prediction != candidate_prediction))
                variants[f"{kind}_{group_name}"] = result
        rows.append({"ab_seed": seed, "a_n": len(a), "b_n": len(b),
                     "dev_n": len(dev), "full_model": baseline,
                     "variants": variants})
        print(json.dumps({"seed": seed,
                          "full_macro_f1": baseline["dev"]["macro_f1"],
                          "without_macro_f1": {name: round(value["dev"]["macro_f1"], 5)
                                               for name, value in variants.items()
                                               if name.startswith("without_")},
                          "only_macro_f1": {name: round(value["dev"]["macro_f1"], 5)
                                            for name, value in variants.items()
                                            if name.startswith("only_")}}), flush=True)
    report = {
        "scope": "TwiBot-20 support-aware A/B monotone group-tree source-group retraining controls, official train/dev only"
                 + (", split-visible sampled graph" if args.split_visible_screen
                    else ", all provided sampled graph" if args.corrected_screen
                    else ", labeled-only sampled graph"),
        "candidate_test_evaluated": False,
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "group_screen_sha256": sha256(args.group_screen),
        "decision_cards_sha256": sha256(args.decision_cards),
        "method": "For 3 prespecified A/B splits, refit the same fixed group-restricted monotone tree with each source group excluded and with each source group alone. All variants use the same A directions, B rows, dev threshold-selection protocol, and 20-fact source representation; the omitted group is absent at fit and inference.",
        "comparison_note": "Retrained absence evaluates whether the other source groups can compensate on these dev accounts. It differs from replacing a group in the frozen full model, which measures that frozen function's dependence. It does not establish independent behavioral meaning or OOD value.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
