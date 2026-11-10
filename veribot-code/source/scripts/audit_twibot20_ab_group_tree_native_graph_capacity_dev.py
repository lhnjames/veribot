"""Separate native graph degree evidence from sampled neighbor-list counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from audit_twibot20_ab_group_tree_source_group_capacity_dev import evaluate_candidate
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


NATIVE_DEGREES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")
SAMPLED_LIST_COUNTS = ("sampled_following_count", "sampled_follower_count")


def variant_groups(names: list[str], retained_facts: set[str]) -> tuple[list[int], list[list[int]]]:
    width = len(names)
    selected = [j for j, name in enumerate(names) if name in retained_facts]
    cols = selected + [j + width for j in selected]
    old_to_new = {old: new for new, old in enumerate(cols)}
    grouped = []
    for members in GROUPS.values():
        group_old = [j for j, name in enumerate(names)
                     if name in members and name in retained_facts]
        if group_old:
            grouped.append([old_to_new[j] for j in group_old +
                            [j + width for j in group_old]])
    if sorted(j for group in grouped for j in group) != list(range(len(cols))):
        raise ValueError("Graph subgroup variant does not partition its tails")
    return cols, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--group-capacity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    prior = json.loads(args.group_capacity.read_text(encoding="utf-8"))
    if (sha256(args.facts) != screen["facts_sha256"]
            or sha256(args.reference) != screen["reference_sha256"]
            or sha256(args.compact) != screen["compact_sha256"]
            or sha256(args.group_screen) != prior["group_screen_sha256"]):
        raise ValueError("Source, reference or prior screens differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if set(GROUPS["native_follow_graph"]) != set(NATIVE_DEGREES) | set(SAMPLED_LIST_COUNTS):
        raise ValueError("Graph source fact partition differs")
    support = values[:, all_names.index("tweet_sample_count")]
    full = LabelBlindReference.load(args.reference).transform(values, contexts=contexts)
    batch = evidence_subset(full, [all_names.index(name) for name in names],
                            names, support, True)
    variants = {
        "without_global_native_degrees": set(names) - set(NATIVE_DEGREES),
        "without_sampled_list_counts": set(names) - set(SAMPLED_LIST_COUNTS),
        "only_global_native_degrees": set(NATIVE_DEGREES),
        "only_sampled_list_counts": set(SAMPLED_LIST_COUNTS),
    }
    rows = []
    for seed in (0, 1, 2):
        a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                                random_state=1000 + seed)
        signs = DualTailEvidenceReasoner._select_tail_signs(take(batch, a), labels[a])
        signs = np.r_[signs[:, 0], signs[:, 1]]
        xb = design(take(batch, b), signs)
        xd = design(take(batch, dev), signs)
        baseline, full_pred = evaluate_candidate(
            xb, labels[b], xd, labels[dev], source_groups(names), signs, seed)
        earlier = prior["rows"][seed]["full_model"]
        if any(abs(baseline["dev"][key] - earlier["dev"][key]) > 1e-12
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("Full model differs from prior source-group capacity audit")
        result = {}
        for variant_name, retained in variants.items():
            cols, groups = variant_groups(names, retained)
            report, predicted = evaluate_candidate(
                xb[:, cols], labels[b], xd[:, cols], labels[dev],
                groups, signs[cols], seed)
            report["retained_facts"] = [name for name in names if name in retained]
            report["full_only_correct"] = int(np.count_nonzero(
                (full_pred == labels[dev]) & (predicted != labels[dev])))
            report["variant_only_correct"] = int(np.count_nonzero(
                (full_pred != labels[dev]) & (predicted == labels[dev])))
            report["class_changes_vs_full"] = int(np.count_nonzero(full_pred != predicted))
            result[variant_name] = report
        rows.append({"ab_seed": seed, "full_model": baseline, "variants": result})
        print(json.dumps({"seed": seed,
                          "full_macro_f1": round(baseline["dev"]["macro_f1"], 5),
                          "variant_macro_f1": {name: round(row["dev"]["macro_f1"], 5)
                                               for name, row in result.items()}}), flush=True)
    report = {
        "scope": "TwiBot-20 dev-only support-aware A/B graph source subgroup retraining capacity audit",
        "candidate_test_evaluated": False,
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "compact_sha256": sha256(args.compact),
        "group_screen_sha256": sha256(args.group_screen),
        "source_group_capacity_sha256": sha256(args.group_capacity),
        "native_degrees": NATIVE_DEGREES,
        "sampled_list_counts": SAMPLED_LIST_COUNTS,
        "method": "same A/B split, signs, B fit, fixed group-tree config and dev threshold selection; refit after omitting only the global native degrees or only the account's sampled neighbor-list lengths; separately fit each subgroup alone",
        "rows": rows,
        "interpretation": "A raw native-edge degree is computable from the graph. Its mere presence in a fitted tree or a score effect does not prove extra predictive benefit after other source groups retrain. All comparisons are dev-only and do not assess graph semantics under population shift.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
