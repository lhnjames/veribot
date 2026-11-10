"""Check source-group isolation in the saved TwiBot-20 LightGBM model."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from lightgbm import Booster

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from train_v3_twibot20 import sha256


def split_features(node: dict) -> set[int]:
    if "split_index" not in node:
        return set()
    return ({int(node["split_feature"])}
            | split_features(node["left_child"])
            | split_features(node["right_child"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    if sha256(args.model) != cards["frozen_model_sha256"]:
        raise ValueError("Saved model differs from decision cards")
    terms = cards["decision_cards"][0]["fact_terms"]
    names = [term["fact_name"] for term in terms]
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("Decision card fact vocabulary differs")
    membership: dict[int, str] = {}
    for group, members in GROUPS.items():
        for name in members:
            j = names.index(name)
            for column in (j, j + len(names)):
                if column in membership:
                    raise ValueError("A tail belongs to multiple groups")
                membership[column] = group
    if set(membership) != set(range(2 * len(names))):
        raise ValueError("Source groups do not cover every model input")
    model = Booster(model_file=str(args.model)).dump_model()
    if model["max_feature_idx"] != 2 * len(names) - 1:
        raise ValueError("Saved model feature count differs from cards")
    expected_signs = ([int(term["upper_monotone_direction"]) for term in terms]
                      + [int(term["lower_monotone_direction"]) for term in terms])
    constraints = [int(value) for value in model["monotone_constraints"]]
    if constraints != expected_signs:
        raise ValueError("Saved model monotone directions differ from decision cards")
    groups_by_tree = Counter()
    constant_trees = 0
    split_nodes = 0
    used_features: set[int] = set()
    for tree in model["tree_info"]:
        features = split_features(tree["tree_structure"])
        if not features:
            constant_trees += 1
            continue
        groups = {membership[j] for j in features}
        if len(groups) != 1:
            raise ValueError(f"Tree {tree['tree_index']} crosses source groups: {groups}")
        groups_by_tree[next(iter(groups))] += 1
        split_nodes += int(tree["num_leaves"]) - 1
        used_features |= features
    if (dict(groups_by_tree) != cards["tree_counts_by_group"]
            or sum(groups_by_tree.values()) + constant_trees != len(model["tree_info"])):
        raise ValueError("Saved tree group counts differ from cards")
    unused = sorted(set(membership) - used_features)
    unused_tails = [{"column": j, "fact": names[j % len(names)],
                     "tail": "upper" if j < len(names) else "lower",
                     "source_group": membership[j]} for j in unused]
    unread_facts = [name for j, name in enumerate(names)
                    if j in unused and j + len(names) in unused]
    report = {
        "scope": "saved frozen model structure: "
                 + cards.get("graph_scope", "labeled-row sampled neighbor declarations only"),
        "model_sha256": sha256(args.model),
        "decision_cards_sha256": sha256(args.cards),
        "model_input_columns": 2 * len(names),
        "source_groups": {key: list(value) for key, value in GROUPS.items()},
        "trees_total": len(model["tree_info"]),
        "trees_by_single_source_group": dict(groups_by_tree),
        "constant_trees": constant_trees,
        "cross_group_trees": 0,
        "split_nodes": split_nodes,
        "used_input_columns": len(used_features),
        "unused_input_columns": unused,
        "unused_input_tails": unused_tails,
        "facts_with_neither_tail_used": unread_facts,
        "saved_monotone_constraints_match_a_directions": True,
        "status": "all_saved_trees_source_group_isolated",
        "interpretation": "Each saved tree uses columns from exactly one predefined source group, or is constant. Therefore the frozen logit is a bias plus four group functions. This structural property certifies group-level numerical decomposition, not original social causality or independent behavioral truth.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
