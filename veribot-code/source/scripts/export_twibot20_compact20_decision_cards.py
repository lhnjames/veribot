"""Export complete, signed, source-bound decisions for the compact 20-fact model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
import numpy as np
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.twibot20_source_certificate import direct_query
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


GROUPS = {
    "profile_counts": ("followers_count", "friends_count", "statuses_count",
                       "listed_count", "favourites_count"),
    "profile_presentation": ("protected", "default_profile", "default_profile_image",
                             "description_length"),
    "sampled_tweet_stats": ("retweet_fraction", "url_fraction", "hashtag_fraction",
                            "mention_fraction", "mean_tweet_length", "unique_tweet_fraction"),
    "native_follow_graph": ("sampled_following_count", "sampled_follower_count",
                            "graph_out_degree", "graph_in_degree", "graph_reciprocal_degree"),
}


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--source-edit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    compact_hash = sha256(args.compact)
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    edited = json.loads(args.source_edit.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip"
            or audit["source_archive_sha256"] != compact["source_archive_sha256"]
            or audit["fact_cache_sha256"] != compact["base_facts_sha256"]
            or edited["compact_certificate_sha256"] != compact_hash):
        raise ValueError("Full source audit or source edit differs from compact model")
    for key, path in (("source_archive_sha256", args.zip),
                      ("base_facts_sha256", args.facts),
                      ("reference_artifact_sha256", args.reference)):
        if sha256(path) != compact[key]:
            raise ValueError(f"Frozen source or reference differs: {key}")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        split = saved["split"].astype(str)
        positions = saved["positions"].astype(int)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        names = saved["names"].astype(str).tolist()
    test = np.flatnonzero(split == "test")
    test_ids = ids[test].tolist()
    test_set = set(test_ids)
    if ([row["account_id"] for row in compact["rows"]] != test_ids
            or [row["account_id"] for row in edited["rows"]] != test_ids):
        raise ValueError("Compact and source-edit test account order differs")
    edges: set[tuple[str, str]] = set()
    incident_declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    raw_test_rows = []
    with zipfile.ZipFile(args.zip) as zipped:
        for part in ("train", "dev", "test"):
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    if part == "test":
                        if ordinal >= len(test) or uid != test_ids[ordinal] \
                                or positions[test[ordinal]] != ordinal:
                            raise ValueError("Official test row order differs")
                        direct, _, context = account_facts(row)
                        if context != contexts[test[ordinal]] or not np.allclose(
                                direct, values[test[ordinal], :len(direct)],
                                rtol=0, atol=1e-12, equal_nan=True):
                            raise ValueError(f"Official direct facts differ: {uid}")
                        raw_test_rows.append(row)
                    neighbor = row.get("neighbor") or {}
                    for field, reverse in (("following", False), ("follower", True)):
                        for position, other_id in enumerate(neighbor.get(field) or []):
                            other = str(other_id)
                            if other == uid:
                                continue
                            edge = (other, uid) if reverse else (uid, other)
                            edges.add(edge)
                            if edge[0] in test_set or edge[1] in test_set:
                                incident_declarations[edge].append(
                                    f"Twibot-20/{part}.json[{ordinal}].neighbor.{field}[{position}]")
    if len(raw_test_rows) != len(test) or len(edges) != audit["distinct_directed_follow_edges"]:
        raise ValueError("Official source graph or test account count differs")
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    incident: dict[str, list[tuple[str, str, list[str]]]] = defaultdict(list)
    for (source, target), declarations in incident_declarations.items():
        if source in test_set:
            outgoing[source].add(target)
            incident[source].append((source, target, sorted(declarations)))
        if target in test_set:
            incoming[target].add(source)
            incident[target].append((source, target, sorted(declarations)))
    graph_names = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")
    for k, uid in enumerate(test_ids):
        recomputed = (len(outgoing[uid]), len(incoming[uid]),
                      len(outgoing[uid] & incoming[uid]))
        expected = tuple(int(values[test[k], names.index(name)]) for name in graph_names)
        if recomputed != expected:
            raise ValueError(f"Official graph degree differs: {uid}")
    reference = LabelBlindReference.load(args.reference)
    if reference.names != tuple(names):
        raise ValueError("Frozen reference vocabulary differs")
    evidence = reference.transform(values[test], contexts=contexts[test])
    retained = compact["retained_facts"]
    if len(retained) != 20 or set(retained) != set().union(*map(set, GROUPS.values())):
        raise ValueError("Compact source grouping differs")
    coefficient = compact["coefficient_by_fact"]
    intercept = float(compact["intercept"])
    threshold = float(compact["threshold"])
    threshold_logit = float(logit(threshold))
    cards = []
    max_logit_error = 0.0
    max_group_error = 0.0
    for k, i in enumerate(test):
        uid = test_ids[k]
        raw_row = raw_test_rows[k]
        source_hash = canonical_hash({name: value for name, value in raw_row.items()
                                      if name != "label"})
        entries = sorted(incident[uid], key=lambda item: (item[0], item[1]))
        graph_source = {
            "query": "deduplicate directed following/follower declarations in official train/dev/test involving this account",
            "incident_edges": len(entries),
            "source_declarations": sum(len(refs) for _, _, refs in entries),
            "incident_declarations_sha256": canonical_hash(entries),
            "out_degree": len(outgoing[uid]),
            "in_degree": len(incoming[uid]),
            "reciprocal_degree": len(outgoing[uid] & incoming[uid]),
        }
        terms = []
        for name in retained:
            j = names.index(name)
            raw_value = values[i, j]
            upper_encoded = float(np.log1p(evidence.upper[k, j]) * evidence.reliability[k, j])
            lower_encoded = float(np.log1p(evidence.lower[k, j]) * evidence.reliability[k, j])
            upper_effect = upper_encoded * coefficient[name]["upper"]
            lower_effect = lower_encoded * coefficient[name]["lower"]
            source = (graph_source if name in graph_names
                      else direct_query("test", int(positions[i]), name, raw_row))
            terms.append({
                "fact_name": name,
                "source_query": source,
                "source_value": float(raw_value) if np.isfinite(raw_value) else None,
                "missing": not bool(np.isfinite(raw_value)),
                "upper_tail_score": float(evidence.upper[k, j]),
                "lower_tail_score": float(evidence.lower[k, j]),
                "reliability": float(evidence.reliability[k, j]),
                "upper_encoded": upper_encoded,
                "lower_encoded": lower_encoded,
                "upper_logit_effect": upper_effect,
                "lower_logit_effect": lower_effect,
                "total_logit_effect": upper_effect + lower_effect,
            })
        reconstructed = intercept + sum(term["total_logit_effect"] for term in terms)
        saved_row = compact["rows"][k]
        logit_error = abs(reconstructed - saved_row["compact_logit"])
        max_logit_error = max(max_logit_error, logit_error)
        if logit_error > 1e-10 or int(expit(reconstructed) >= threshold) != saved_row["compact_class"]:
            raise ValueError(f"Card does not reconstruct frozen decision: {uid}")
        group_effects = {group: sum(term["total_logit_effect"] for term in terms
                                     if term["fact_name"] in members)
                         for group, members in GROUPS.items()}
        group_error = max(abs(group_effects[group] - saved_row["group_effects"][group])
                          for group in GROUPS)
        max_group_error = max(max_group_error, group_error)
        if group_error > 1e-10:
            raise ValueError(f"Card group effects differ from frozen decision: {uid}")
        first_edit = edited["rows"][k]
        if abs(first_edit["original_logit"] - reconstructed) > 1e-10:
            raise ValueError("Source edit does not begin at this frozen score")
        prediction = int(saved_row["compact_class"])
        cards.append({
            "account_id": uid,
            "official_source_locator": f"Twibot-20/test.json[{positions[i]}]",
            "source_record_without_label_sha256": source_hash,
            "graph_source": graph_source,
            "reference_version": reference.version,
            "model_certificate_sha256": compact_hash,
            "intercept": intercept,
            "threshold": threshold,
            "threshold_logit": threshold_logit,
            "baseline_margin": intercept - threshold_logit,
            "fact_terms": terms,
            "group_logit_effects": group_effects,
            "full_logit": reconstructed,
            "full_margin": reconstructed - threshold_logit,
            "full_probability": float(expit(reconstructed)),
            "prediction": prediction,
            "evaluation_label": int(labels[i]),
            "prediction_correct": prediction == int(labels[i]),
            "positive_terms": [term["fact_name"] for term in terms
                               if term["total_logit_effect"] > 1e-12],
            "negative_terms": [term["fact_name"] for term in terms
                               if term["total_logit_effect"] < -1e-12],
            "zero_terms": [term["fact_name"] for term in terms
                           if abs(term["total_logit_effect"]) <= 1e-12],
            "first_sampled_tweet_record_edit": {
                "source_locator": first_edit["first_tweet_locator"],
                "edited_logit": first_edit["first_removed_logit"],
                "edited_class": first_edit["first_removed_class"],
                "class_changed": prediction != first_edit["first_removed_class"],
            },
        })
    exemplars = {}
    for name, label, prediction in (("correct_bot", 1, 1), ("correct_human", 0, 0),
                                    ("human_wrongly_called_bot", 0, 1),
                                    ("bot_wrongly_called_human", 1, 0)):
        selected = next(card for card in cards if card["evaluation_label"] == label
                        and card["prediction"] == prediction)
        exemplars[name] = selected["account_id"]
    exemplar_graph_declarations = {
        category: [{"source": source, "target": target, "declarations": refs}
                   for source, target, refs in sorted(incident[uid],
                                                      key=lambda item: (item[0], item[1]))]
        for category, uid in exemplars.items()
    }
    result = {
        "dataset": "twibot20",
        "scope": "all official test decisions of frozen compact 20-fact model",
        "source_archive_sha256": compact["source_archive_sha256"],
        "base_facts_sha256": compact["base_facts_sha256"],
        "full_source_audit_sha256": sha256(args.source_audit),
        "reference_artifact_sha256": compact["reference_artifact_sha256"],
        "compact_certificate_sha256": compact_hash,
        "source_edit_certificate_sha256": sha256(args.source_edit),
        "official_labeled_graph_edges_recomputed": len(edges),
        "cards": len(cards),
        "fact_terms_per_card": len(retained),
        "max_logit_reconstruction_error": max_logit_error,
        "max_group_reconstruction_error": max_group_error,
        "correct_cards": sum(card["prediction_correct"] for card in cards),
        "wrong_cards": sum(not card["prediction_correct"] for card in cards),
        "exemplar_ids_first_in_official_order": exemplars,
        "exemplar_graph_edge_declarations": exemplar_graph_declarations,
        "interpretation": "source facts and frozen classifier arithmetic; behavioral semantic truth and correction utility require independent evaluation",
        "decision_cards": cards,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("decision_cards", "exemplar_graph_edge_declarations")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
