"""Trace uniquely declared support edges through corrected sampled-graph decisions."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    if (sha256(args.zip) != audit["source_archive_sha256"]
            or sha256(args.facts) != cards["fact_cache_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]):
        raise ValueError("Official source, corrected facts or frozen model differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    dev = np.flatnonzero(split == "dev")
    dev_index = {uid: i for i, uid in enumerate(ids[dev].tolist())}
    card_rows = cards["decision_cards"]
    if ids[dev].tolist() != [card["account_id"] for card in card_rows]:
        raise ValueError("Corrected card population differs")
    names = compact["retained_facts"]
    terms = card_rows[0]["fact_terms"]
    if [term["fact_name"] for term in terms] != names:
        raise ValueError("Card fact vocabulary differs")
    signs = np.r_[[term["upper_monotone_direction"] for term in terms],
                  [term["lower_monotone_direction"] for term in terms]]
    old_edges = set()
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for row in ijson.items(stream, "item"):
                    uid = str(row["ID"])
                    neighbor = row.get("neighbor") or {}
                    old_edges.update((uid, str(other))
                                     for other in neighbor.get("following") or []
                                     if str(other) != uid)
                    old_edges.update((str(other), uid)
                                     for other in neighbor.get("follower") or []
                                     if str(other) != uid)
    if len(old_edges) != audit["labeled_only_unique_edges"]:
        raise ValueError("Labeled-row sampled edges differ")
    new_incident = {}
    for account in audit["changed_labeled_account_rows"]:
        for item in account["new_support_edge_declarations_incident"]:
            edge = (item["source"], item["target"])
            locators = tuple(item["support_declarations"])
            if edge in new_incident and new_incident[edge] != locators:
                raise ValueError("Support edge provenance disagrees across endpoints")
            new_incident[edge] = locators
    if len(new_incident) != audit["new_unique_support_edges_incident_to_labeled_accounts"]:
        raise ValueError("Incident new support edge population differs")
    if any(edge in old_edges for edge in new_incident):
        raise ValueError("New support edge also appears in labeled rows")
    entries = []
    edited = []
    edited_contexts = []
    for edge, locators in sorted(new_incident.items()):
        source, target = edge
        affected = [(source, 0), (target, 1)]
        if len(locators) != 1 or not any(uid in dev_index for uid, _ in affected):
            continue
        reciprocal = (target, source) in old_edges or (target, source) in new_incident
        for uid, which in affected:
            if uid not in dev_index:
                continue
            k = dev_index[uid]
            i = dev[k]
            revised = values[i].copy()
            revised[all_names.index(GRAPH_DEGREES[which])] -= 1
            if reciprocal:
                revised[all_names.index(GRAPH_DEGREES[2])] -= 1
            if any(revised[all_names.index(name)] < 0 for name in GRAPH_DEGREES):
                raise ValueError("Source edit made a negative graph degree")
            entries.append({
                "account_id": uid, "source_edge": [source, target],
                "source_declaration": locators[0],
                "reverse_edge_present": reciprocal,
                "original_graph_degrees": {name: int(values[i, all_names.index(name)])
                                           for name in GRAPH_DEGREES},
                "edited_graph_degrees": {name: int(revised[all_names.index(name)])
                                         for name in GRAPH_DEGREES},
                "dev_index": k,
            })
            edited.append(revised)
            edited_contexts.append(contexts[i])
    if not entries:
        raise ValueError("No uniquely declared support edge touches dev accounts")
    edited = np.asarray(edited)
    reference = LabelBlindReference.load(args.reference)
    keep = [all_names.index(name) for name in names]
    edited_batch = evidence_subset(
        reference.transform(edited, contexts=np.asarray(edited_contexts)),
        keep, names, edited[:, all_names.index("tweet_sample_count")], True)
    edited_x = design(edited_batch, signs)
    original_x = np.asarray([
        [term["upper_model_input"] for term in card_rows[row["dev_index"]]["fact_terms"]]
        + [term["lower_model_input"] for term in card_rows[row["dev_index"]]["fact_terms"]]
        for row in entries])
    graph_cols = source_groups(names)[list(GROUPS).index("native_follow_graph")]
    other_cols = [j for j in range(original_x.shape[1]) if j not in graph_cols]
    non_graph_error = float(np.max(np.abs(edited_x[:, other_cols] - original_x[:, other_cols])))
    if non_graph_error > 1e-12:
        raise ValueError("Support-edge source edit changed non-graph inputs")
    model = Booster(model_file=str(args.model))
    original_logit = np.asarray(model.predict(original_x, raw_score=True))
    card_logit = np.asarray([card_rows[row["dev_index"]]["full_logit"] for row in entries])
    source_replay_error = float(np.max(np.abs(original_logit - card_logit)))
    if source_replay_error > 1e-12:
        raise ValueError("Frozen original logits differ from decision cards")
    edited_logit = np.asarray(model.predict(edited_x, raw_score=True))
    group_replaced = original_x.copy()
    group_replaced[:, graph_cols] = edited_x[:, graph_cols]
    graph_replay_error = float(np.max(np.abs(
        edited_logit - model.predict(group_replaced, raw_score=True))))
    if graph_replay_error > 1e-12:
        raise ValueError("Edited graph input does not replay whole-model response")
    threshold = float(card_rows[0]["threshold"])
    for j, entry in enumerate(entries):
        k = entry.pop("dev_index")
        entry.update({
            "original_logit": float(original_logit[j]),
            "edited_logit": float(edited_logit[j]),
            "logit_change": float(edited_logit[j] - original_logit[j]),
            "original_class": int(expit(original_logit[j]) >= threshold),
            "edited_class": int(expit(edited_logit[j]) >= threshold),
            "evaluation_label": int(card_rows[k]["evaluation_label"]),
        })
    changed_score = sum(abs(entry["logit_change"]) > 1e-12 for entry in entries)
    changed_class = sum(entry["original_class"] != entry["edited_class"] for entry in entries)
    report = {
        "scope": "Corrected TwiBot-20 group tree: single unique support-edge declaration removal on official dev",
        "candidate_test_evaluated": False,
        "source_archive_sha256": sha256(args.zip),
        "corrected_facts_sha256": sha256(args.facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "decision_cards_sha256": sha256(args.decision_cards),
        "model_sha256": sha256(args.model),
        "incident_new_support_edges": len(new_incident),
        "unique_declaration_edge_edits_touching_dev": len(set(tuple(e["source_edge"]) for e in entries)),
        "affected_dev_account_decisions": len(entries),
        "dev_decisions_with_logit_change": changed_score,
        "dev_decisions_with_class_change": changed_class,
        "max_non_graph_model_input_change": non_graph_error,
        "max_original_card_replay_error": source_replay_error,
        "max_graph_group_replay_error": graph_replay_error,
        "edits": entries,
        "interpretation": "Each edit removes exactly one uniquely declared sampled edge and updates both endpoint degrees and reciprocity when applicable. It measures frozen-model sensitivity to a missing observed relation; no natural source error or bot-label correction is asserted. The released graph remains a sample, not the complete follower graph.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "edits"}, indent=2))


if __name__ == "__main__":
    main()
