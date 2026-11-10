"""Trace each singly declared visible edge through a frozen split-visible model."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import zipfile

import ijson
from lightgbm import Booster
import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from audit_twibot20_support_edge_omission import graph_degrees
from screen_twibot20_compact20_ab_group_tree_dev import design, source_groups
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


GRAPH_DEGREES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if (sha256(args.zip) != cards["source_archive_sha256"]
            or sha256(args.facts) != cards["fact_cache_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]
            or cards["graph_scope"] != "train/support/dev sampled neighbor declarations without test"):
        raise ValueError("Split-visible source, facts, reference or model differ")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names_all = saved["names"].astype(str).tolist()
    if set(split) != {"train", "dev"}:
        raise ValueError("Test account row entered split-visible source cache")
    dev = np.flatnonzero(split == "dev")
    train_ids = set(ids[split == "train"].tolist())
    dev_ids = set(ids[dev].tolist())
    dev_index = {uid: k for k, uid in enumerate(ids[dev].tolist())}
    card_rows = cards["decision_cards"]
    if ids[dev].tolist() != [card["account_id"] for card in card_rows]:
        raise ValueError("Dev source and decision-card order differ")
    edges = set()
    locators = defaultdict(list)
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "support"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    owner = str(row["ID"])
                    neighbor = row.get("neighbor") or {}
                    for field, reverse in (("following", False), ("follower", True)):
                        for position, other in enumerate(neighbor.get(field) or []):
                            other = str(other)
                            if other == owner:
                                continue
                            edge = (other, owner) if reverse else (owner, other)
                            edges.add(edge)
                            if edge[0] in dev_ids or edge[1] in dev_ids:
                                locators[edge].append({
                                    "locator": f"Twibot-20/{part}.json[{ordinal}].neighbor.{field}[{position}]",
                                    "declaring_part": part,
                                    "declaring_account_id": owner,
                                    "field": field,
                                })
    if len(edges) != cards["official_graph_edges_recomputed"]:
        raise ValueError("Visible sampled graph edge population differs")
    source_degrees = graph_degrees(edges, dev_ids)
    for i in dev:
        uid = str(ids[i])
        cached = tuple(int(values[i, names_all.index(name)]) for name in GRAPH_DEGREES)
        if cached != source_degrees[uid]:
            raise ValueError(f"Visible graph source degree differs: {uid}")
    entries = []
    edited_values = []
    edited_contexts = []
    considered_edges = Counter()
    candidate_edges_incident_to_train = Counter()
    for edge, declarations in sorted(locators.items()):
        if len(declarations) != 1:
            continue
        declaration = declarations[0]
        part = declaration["declaring_part"]
        if part == "train":
            continue  # Editing a training source would require refitting the model.
        if part == "dev" and declaration["declaring_account_id"] not in dev_ids:
            raise ValueError("Dev declaration owner is outside dev")
        if part == "support" and declaration["declaring_account_id"] in (
                train_ids | dev_ids):
            raise ValueError("Support declaration owner overlaps train/dev")
        source, target = edge
        if source in train_ids or target in train_ids:
            candidate_edges_incident_to_train[part] += 1
            if part == "support":
                raise ValueError("Support source edit changes a train account; refit required")
        if part == "dev" and any(
                item["declaring_part"] in ("train", "support")
                for item in declarations):
            raise ValueError("Dev source edit changes the train-visible graph")
        reciprocal = (target, source) in edges
        considered_edges[part] += 1
        for uid, degree_field in ((source, GRAPH_DEGREES[0]),
                                  (target, GRAPH_DEGREES[1])):
            if uid not in dev_index:
                continue
            k = dev_index[uid]
            i = dev[k]
            revised = values[i].copy()
            revised[names_all.index(degree_field)] -= 1
            if reciprocal:
                revised[names_all.index(GRAPH_DEGREES[2])] -= 1
            if part == "dev" and declaration["declaring_account_id"] == uid:
                sampled_field = ("sampled_following_count" if declaration["field"] == "following"
                                 else "sampled_follower_count")
                revised[names_all.index(sampled_field)] -= 1
            if any(revised[names_all.index(name)] < 0 for name in GRAPH_DEGREES):
                raise ValueError("Edge removal produced negative graph degree")
            entries.append({
                "account_id": uid, "source_edge": [source, target],
                "source_declaration": declaration["locator"],
                "declaring_part": part,
                "reverse_edge_present": reciprocal,
                "original_graph_facts": {
                    name: (float(values[i, names_all.index(name)])
                           if np.isfinite(values[i, names_all.index(name)]) else None)
                    for name in (*GRAPH_DEGREES, "sampled_following_count",
                                 "sampled_follower_count")},
                "edited_graph_facts": {
                    name: (float(revised[names_all.index(name)])
                           if np.isfinite(revised[names_all.index(name)]) else None)
                    for name in (*GRAPH_DEGREES, "sampled_following_count",
                                 "sampled_follower_count")},
                "dev_index": k,
            })
            edited_values.append(revised)
            edited_contexts.append(contexts[i])
    if not entries:
        raise ValueError("No singly declared dev/support edges touch dev accounts")
    names = compact["retained_facts"]
    terms = card_rows[0]["fact_terms"]
    if [term["fact_name"] for term in terms] != names:
        raise ValueError("Evidence fact vocabulary differs")
    signs = np.r_[[term["upper_monotone_direction"] for term in terms],
                  [term["lower_monotone_direction"] for term in terms]]
    edited_values = np.asarray(edited_values)
    reference = LabelBlindReference.load(args.reference)
    batch = evidence_subset(
        reference.transform(edited_values, contexts=np.asarray(edited_contexts)),
        [names_all.index(name) for name in names], names,
        edited_values[:, names_all.index("tweet_sample_count")], True)
    edited_x = design(batch, signs)
    original_x = np.asarray([
        [term["upper_model_input"] for term in card_rows[item["dev_index"]]["fact_terms"]]
        + [term["lower_model_input"] for term in card_rows[item["dev_index"]]["fact_terms"]]
        for item in entries])
    graph_cols = source_groups(names)[list(GROUPS).index("native_follow_graph")]
    other_cols = [j for j in range(original_x.shape[1]) if j not in graph_cols]
    non_graph_error = float(np.max(np.abs(original_x[:, other_cols]
                                          - edited_x[:, other_cols])))
    if non_graph_error > 1e-12:
        raise ValueError("Visible edge edit changed non-graph model inputs")
    model = Booster(model_file=str(args.model))
    old_logit = np.asarray(model.predict(original_x, raw_score=True))
    card_logit = np.asarray([card_rows[item["dev_index"]]["full_logit"]
                             for item in entries])
    original_replay_error = float(np.max(np.abs(old_logit - card_logit)))
    if original_replay_error > 1e-12:
        raise ValueError("Original source logits differ from frozen decision cards")
    new_logit = np.asarray(model.predict(edited_x, raw_score=True))
    graph_replaced = original_x.copy()
    graph_replaced[:, graph_cols] = edited_x[:, graph_cols]
    response_error = float(np.max(np.abs(
        new_logit - model.predict(graph_replaced, raw_score=True))))
    if response_error > 1e-12:
        raise ValueError("Edited graph source does not replay frozen response")
    threshold = float(card_rows[0]["threshold"])
    by_part = {part: Counter() for part in ("dev", "support")}
    for j, item in enumerate(entries):
        k = item.pop("dev_index")
        item.update({
            "original_logit": float(old_logit[j]),
            "edited_logit": float(new_logit[j]),
            "logit_change": float(new_logit[j] - old_logit[j]),
            "original_class": int(expit(old_logit[j]) >= threshold),
            "edited_class": int(expit(new_logit[j]) >= threshold),
            "evaluation_label": int(card_rows[k]["evaluation_label"]),
        })
        counter = by_part[item["declaring_part"]]
        counter["affected_dev_decisions"] += 1
        counter["logit_changes"] += abs(item["logit_change"]) > 1e-12
        counter["class_changes"] += item["original_class"] != item["edited_class"]
    report = {
        "scope": "TwiBot-20 split-visible frozen group tree, unique dev/support edge declaration removal",
        "candidate_test_evaluated": False,
        "source_archive_sha256": sha256(args.zip),
        "facts_sha256": sha256(args.facts),
        "reference_sha256": sha256(args.reference),
        "decision_cards_sha256": sha256(args.decision_cards),
        "model_sha256": sha256(args.model),
        "visible_unique_edges": len(edges),
        "unique_declaration_edges_considered_by_part": dict(considered_edges),
        "candidate_edges_incident_to_train_by_part": {
            part: candidate_edges_incident_to_train.get(part, 0)
            for part in ("dev", "support")},
        "train_fit_graph_facts_changed_by_any_edit": False,
        "affected_dev_decisions": len(entries),
        "response_by_declaring_part": {part: dict(counter)
                                       for part, counter in by_part.items()},
        "max_non_graph_model_input_change": non_graph_error,
        "max_original_card_replay_error": original_replay_error,
        "max_graph_group_response_error": response_error,
        "edits": entries,
        "interpretation": "Each edit removes one dev or support sampled-neighbor declaration that uniquely supports a visible edge. A dev self-list count is decremented alongside all affected endpoint graph degrees; support edits change only endpoint degrees. Every eligible edit leaves the train-fit graph facts unchanged: dev declarations were unseen at fit, and none of the eligible support edges touches a train account. Training declarations are excluded because a training-source edit would require refitting. This measures a specified missing-observation response, not correction of a known erroneous edge or social causality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "edits"}, indent=2))


if __name__ == "__main__":
    main()
