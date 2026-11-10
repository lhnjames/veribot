"""Replay newly included official support edges through the old frozen group tree."""

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
from screen_twibot20_compact20_ab_support_dev import evidence_subset
from train_v3_twibot20 import sha256
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-facts", type=Path, required=True)
    parser.add_argument("--corrected-facts", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    cards = json.loads(args.decision_cards.read_text(encoding="utf-8"))
    support_audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    corrected_manifest = json.loads(args.corrected_facts.with_suffix(".json").read_text(encoding="utf-8"))
    if (sha256(args.old_facts) != cards["fact_cache_sha256"]
            or sha256(args.corrected_facts) != corrected_manifest["fact_cache_sha256"]
            or sha256(args.support_audit) != corrected_manifest["support_edge_audit_sha256"]
            or sha256(args.reference) != cards["reference_artifact_sha256"]
            or sha256(args.model) != cards["frozen_model_sha256"]
            or sha256(args.group_screen) != cards["group_screen_sha256"]
            or screen["compact_sha256"] != sha256(args.compact)):
        raise ValueError("Source or frozen model version differs")
    with np.load(args.old_facts, allow_pickle=False) as old, \
            np.load(args.corrected_facts, allow_pickle=False) as new:
        if set(old.files) != set(new.files):
            raise ValueError("Corrected cache field vocabulary differs")
        for key in old.files:
            if key != "values" and not np.array_equal(old[key], new[key]):
                raise ValueError(f"Corrected cache unexpectedly changed {key}")
        ids = old["ids"].astype(str)
        split = old["split"].astype(str)
        labels = old["labels"].astype(int)
        names_all = old["names"].astype(str).tolist()
        contexts = old["contexts"].astype(str)
        old_values = old["values"].astype(float)
        new_values = new["values"].astype(float)
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    selected = next(row for row in screen["rows"]
                    if row["support_aware"] and row["ab_seed"] == cards["a_seed"])
    signs = np.r_[[selected["a_signs_by_fact"][name]["upper"] for name in names],
                  [selected["a_signs_by_fact"][name]["lower"] for name in names]]
    reference = LabelBlindReference.load(args.reference)
    keep = [names_all.index(name) for name in names]
    old_batch = evidence_subset(reference.transform(old_values[dev], contexts=contexts[dev]),
                                keep, names, old_values[dev, names_all.index("tweet_sample_count")], True)
    new_batch = evidence_subset(reference.transform(new_values[dev], contexts=contexts[dev]),
                                keep, names, new_values[dev, names_all.index("tweet_sample_count")], True)
    old_x = design(old_batch, signs)
    new_x = design(new_batch, signs)
    groups = source_groups(names)
    graph_cols = groups[list(GROUPS).index("native_follow_graph")]
    other_cols = [j for j in range(old_x.shape[1]) if j not in graph_cols]
    if np.max(np.abs(new_x[:, other_cols] - old_x[:, other_cols])) > 1e-12:
        raise ValueError("Official support edges unexpectedly changed non-graph inputs")
    model = Booster(model_file=str(args.model))
    old_logit = np.asarray(model.predict(old_x, raw_score=True))
    new_logit = np.asarray(model.predict(new_x, raw_score=True))
    if ids[dev].tolist() != [card["account_id"] for card in cards["decision_cards"]] or \
            np.max(np.abs(old_logit - [card["full_logit"] for card in cards["decision_cards"]])) > 1e-12:
        raise ValueError("Old frozen dev decisions differ")
    graph_only = old_x.copy()
    graph_only[:, graph_cols] = new_x[:, graph_cols]
    response_replay = float(np.max(np.abs(
        new_logit - model.predict(graph_only, raw_score=True))))
    if response_replay > 1e-12:
        raise ValueError("Corrected graph response does not replay")
    threshold = float(cards["decision_cards"][0]["threshold"])
    old_class = expit(old_logit) >= threshold
    new_class = expit(new_logit) >= threshold
    changed_source = {row["account_id"]: row
                      for row in support_audit["changed_labeled_account_rows"]
                      if row["split"] == "dev"}
    rows = []
    for k, i in enumerate(dev):
        uid = str(ids[i])
        if uid not in changed_source:
            continue
        rows.append({
            "account_id": uid,
            "support_edge_source_details": changed_source[uid][
                "new_support_edge_declarations_incident"],
            "old_labeled_only_graph": changed_source[uid]["old_labeled_only_degrees"],
            "all_provided_sample_graph": changed_source[uid]["all_provided_degrees"],
            "old_logit": float(old_logit[k]),
            "new_logit_old_frozen_model": float(new_logit[k]),
            "logit_change": float(new_logit[k] - old_logit[k]),
            "old_class": int(old_class[k]),
            "new_class_old_frozen_model": int(new_class[k]),
            "evaluation_label": int(labels[i]),
        })
    if len(rows) != support_audit["labeled_accounts_with_degree_change_by_split"]["dev"]:
        raise ValueError("Dev source-changed account count differs")
    report = {
        "scope": "official dev graph-source correction response of old frozen seed0 A/B group tree",
        "candidate_test_evaluated": False,
        "old_fact_cache_sha256": sha256(args.old_facts),
        "corrected_fact_cache_sha256": sha256(args.corrected_facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "old_reference_sha256": sha256(args.reference),
        "old_model_sha256": sha256(args.model),
        "source_changed_dev_accounts": len(rows),
        "dev_logit_changed": int(np.count_nonzero(np.abs(new_logit - old_logit) > 1e-12)),
        "dev_class_changed": int(np.count_nonzero(old_class != new_class)),
        "dev_old_errors": int(np.count_nonzero(old_class != labels[dev])),
        "dev_new_errors_under_old_model": int(np.count_nonzero(new_class != labels[dev])),
        "max_non_graph_model_input_change": float(np.max(np.abs(new_x[:, other_cols] - old_x[:, other_cols]))),
        "max_graph_source_response_replay_error": response_replay,
        "changed_dev_rows": rows,
        "interpretation": "This is the old frozen model's response when the graph inputs are corrected to include official support-row sampled edges. The reference and model were trained on the old labeled-only graph, so this is a source-integrity sensitivity test, not an estimate of a retrained corrected system or external test performance.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key != "changed_dev_rows"}, indent=2))


if __name__ == "__main__":
    main()
