"""Freeze one prespecified A/B group tree and export source-bound dev decisions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import ijson
from lightgbm import Booster
import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_twibot20_compact20_capacity import GROUPS
from screen_twibot20_compact20_ab_group_tree_dev import (
    design, fit_tree, score_model, source_groups)
from screen_twibot20_compact20_ab_support_dev import evidence_subset, take
from train_v3_twibot20 import sha256
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.twibot20_facts import account_facts
from verifiable_bot.twibot20_source_certificate import direct_query
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


GRAPH_NAMES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")
SEED = 0


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_official_source(archive: Path, ids: np.ndarray, positions: np.ndarray,
                         values: np.ndarray, contexts: np.ndarray,
                         names: list[str], dev: np.ndarray,
                         expected_edge_count: int,
                         source_parts: tuple[str, ...] = ("train", "dev", "test")) -> tuple[list[dict], dict[str, dict], dict[str, list]]:
    dev_ids = ids[dev].tolist()
    dev_set = set(dev_ids)
    rows: list[dict] = []
    edges: set[tuple[str, str]] = set()
    declarations: dict[tuple[str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive) as zipped:
        for part in source_parts:
            with zipped.open(f"Twibot-20/{part}.json") as stream:
                for ordinal, row in enumerate(ijson.items(stream, "item")):
                    uid = str(row["ID"])
                    if part == "dev":
                        k = len(rows)
                        if k >= len(dev) or uid != dev_ids[k] or positions[dev[k]] != ordinal:
                            raise ValueError("Official dev source order differs")
                        direct, _, context = account_facts(row)
                        if context != contexts[dev[k]] or not np.allclose(
                                direct, values[dev[k], :len(direct)],
                                rtol=0, atol=1e-12, equal_nan=True):
                            raise ValueError(f"Official direct facts differ: {uid}")
                        rows.append(row)
                    neighbor = row.get("neighbor") or {}
                    for field, reverse in (("following", False), ("follower", True)):
                        for loc, other in enumerate(neighbor.get(field) or []):
                            other = str(other)
                            if other == uid:
                                continue
                            edge = (other, uid) if reverse else (uid, other)
                            edges.add(edge)
                            if edge[0] in dev_set or edge[1] in dev_set:
                                declarations[edge].append(
                                    f"Twibot-20/{part}.json[{ordinal}].neighbor.{field}[{loc}]")
    if len(rows) != len(dev) or len(edges) != expected_edge_count:
        raise ValueError("Official dev population or native graph differs")
    outgoing: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    incident: dict[str, list] = defaultdict(list)
    for (source, target), locators in declarations.items():
        item = [source, target, sorted(locators)]
        if source in dev_set:
            outgoing[source].add(target)
            incident[source].append(item)
        if target in dev_set:
            incoming[target].add(source)
            incident[target].append(item)
    graph_source = {}
    for k, uid in enumerate(dev_ids):
        observed = (len(outgoing[uid]), len(incoming[uid]),
                    len(outgoing[uid] & incoming[uid]))
        cached = tuple(int(values[dev[k], names.index(name)]) for name in GRAPH_NAMES)
        if observed != cached:
            raise ValueError(f"Native graph facts differ: {uid}")
        entries = sorted(incident[uid])
        graph_source[uid] = {
            "query": "deduplicated directed following/follower declarations in official "
                     + "/".join(source_parts),
            "incident_edges": len(entries),
            "source_declarations": sum(len(item[2]) for item in entries),
            "incident_declarations_sha256": canonical_hash(entries),
            "out_degree": observed[0], "in_degree": observed[1],
            "reciprocal_degree": observed[2],
        }
    return rows, graph_source, incident


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--compact", type=Path, required=True)
    parser.add_argument("--group-screen", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--support-edge-audit", type=Path)
    parser.add_argument("--corrected-screen", type=Path)
    parser.add_argument("--visibility-audit", type=Path)
    parser.add_argument("--split-visible-screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compact = json.loads(args.compact.read_text(encoding="utf-8"))
    screen = json.loads(args.group_screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    corrected_mode = args.corrected_screen is not None
    split_visible_mode = args.split_visible_screen is not None
    if corrected_mode and split_visible_mode:
        raise ValueError("Choose one graph visibility protocol")
    with_support = corrected_mode or split_visible_mode
    if with_support != (args.support_edge_audit is not None):
        raise ValueError("Provided support graph requires support-edge audit")
    if split_visible_mode != (args.visibility_audit is not None):
        raise ValueError("Split-visible graph requires visibility audit")
    support_audit = (json.loads(args.support_edge_audit.read_text(encoding="utf-8"))
                     if with_support else None)
    corrected_screen = (json.loads(args.corrected_screen.read_text(encoding="utf-8"))
                        if corrected_mode else None)
    split_screen = (json.loads(args.split_visible_screen.read_text(encoding="utf-8"))
                    if split_visible_mode else None)
    visibility = (json.loads(args.visibility_audit.read_text(encoding="utf-8"))
                  if split_visible_mode else None)
    if (sha256(args.zip) != audit["source_archive_sha256"]
            or (not with_support and sha256(args.facts) != audit["fact_cache_sha256"])
            or (not with_support and sha256(args.reference) != compact["reference_artifact_sha256"])
            or (not with_support and screen["facts_sha256"] != sha256(args.facts))
            or screen["compact_sha256"] != sha256(args.compact)):
        raise ValueError("Official ZIP, facts, reference or prior screen differs")
    if corrected_mode and (support_audit["source_archive_sha256"] != sha256(args.zip)
                           or corrected_screen["corrected_fact_cache_sha256"] != sha256(args.facts)
                           or corrected_screen["corrected_reference_sha256"] != sha256(args.reference)
                           or corrected_screen["support_edge_audit_sha256"] != sha256(args.support_edge_audit)
                           or corrected_screen["corrected_seed0_model_sha256"] != sha256(args.model)):
        raise ValueError("Corrected graph source, reference or model differs")
    if split_visible_mode and (
            support_audit["source_archive_sha256"] != sha256(args.zip)
            or visibility["source_archive_sha256"] != sha256(args.zip)
            or visibility["support_edge_audit_sha256"] != sha256(args.support_edge_audit)
            or split_screen["split_visible_facts_sha256"] != sha256(args.facts)
            or split_screen["reference_sha256"] != sha256(args.reference)
            or split_screen["seed0_model_sha256"] != sha256(args.model)
            or sha256(args.visibility_audit) != json.loads(
                args.facts.with_suffix(".json").read_text(encoding="utf-8"))[
                    "visibility_audit_sha256"]
            or visibility["corrected_fact_cache_sha256"] != json.loads(
                args.facts.with_suffix(".json").read_text(encoding="utf-8"))[
                    "source_full_fact_cache_sha256"]):
        raise ValueError("Split-visible graph source, reference or model differs")
    if audit["status"] != "verified_all_labeled_account_fact_cells_against_official_zip":
        raise ValueError("Full source audit status differs")
    with np.load(args.facts, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        positions = saved["positions"].astype(int)
        split = saved["split"].astype(str)
        labels = saved["labels"].astype(int)
        values = saved["values"].astype(float)
        contexts = saved["contexts"].astype(str)
        all_names = saved["names"].astype(str).tolist()
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    names = compact["retained_facts"]
    if len(names) != 20 or set(names) != set().union(*map(set, GROUPS.values())):
        raise ValueError("Compact vocabulary differs")
    chosen = (next(row for row in corrected_screen["rows"] if row["ab_seed"] == SEED)
              if corrected_mode else next(row for row in split_screen["rows"]
                                          if row["ab_seed"] == SEED)
              if split_visible_mode else next(row for row in screen["rows"]
                                              if row["support_aware"] and row["ab_seed"] == SEED))
    a, b = train_test_split(train, test_size=.5, stratify=labels[train],
                            random_state=1000 + SEED)
    reference = LabelBlindReference.load(args.reference)
    full = reference.transform(values, contexts=contexts)
    sample_count = values[:, all_names.index("tweet_sample_count")]
    if not np.isfinite(sample_count[train]).all() or not np.isfinite(sample_count[dev]).all():
        raise ValueError("Tweet support is incomplete")
    batch = evidence_subset(full, [all_names.index(name) for name in names],
                            names, sample_count, True)
    signs_by_fact = DualTailEvidenceReasoner._select_tail_signs(take(batch, a), labels[a])
    signs = np.r_[signs_by_fact[:, 0], signs_by_fact[:, 1]]
    observed_signs = {name: {"upper": int(signs[j]),
                             "lower": int(signs[j + len(names)])}
                      for j, name in enumerate(names)}
    expected_signs = (chosen["tail_directions_by_fact"] if split_visible_mode
                      else chosen["a_signs_by_fact"] if not corrected_mode else None)
    if expected_signs is not None and observed_signs != expected_signs:
        raise ValueError("A direction signs differ from recorded screen")
    x_b = design(take(batch, b), signs)
    x_dev = design(take(batch, dev), signs)
    groups = source_groups(names)
    model = fit_tree(x_b, labels[b], groups, signs, True, SEED)
    score = score_model(model, x_dev, labels[dev], groups)
    target = (chosen["all_provided_sample_graph_dev"] if corrected_mode
              else chosen["dev"] if split_visible_mode
              else chosen["constrained_group_tree"]["dev"])
    target_threshold = (chosen["corrected_threshold"] if corrected_mode
                        else chosen["threshold"] if split_visible_mode
                        else chosen["constrained_group_tree"]["threshold"])
    if abs(score["threshold"] - target_threshold) > 1e-12 or \
            any(abs(score["dev"][key] - target[key]) > 1e-12
                for key in ("acc", "macro_f1", "auroc")):
        raise ValueError("Frozen model does not replay selected dev screen")
    if with_support:
        if np.max(np.abs(model.booster_.predict(x_dev, raw_score=True)
                         - Booster(model_file=str(args.model)).predict(
                             x_dev, raw_score=True))) > 1e-12:
            raise ValueError("Corrected frozen model does not replay refit")
    else:
        args.model.parent.mkdir(parents=True, exist_ok=True)
        model.booster_.save_model(str(args.model))
    frozen = Booster(model_file=str(args.model))
    raw = np.asarray(frozen.predict(x_dev, raw_score=True))
    if np.max(np.abs(raw - model.booster_.predict(x_dev, raw_score=True))) > 1e-12:
        raise ValueError("Reloaded model differs")
    probability = expit(raw)
    threshold = float(score["threshold"])
    threshold_logit = float(logit(threshold))
    baseline = np.asarray(frozen.predict(np.zeros_like(x_dev), raw_score=True))
    masked_group = {}
    group_effect = {}
    for group_name, group_cols in zip(GROUPS, groups, strict=True):
        masked = x_dev.copy()
        masked[:, group_cols] = 0.0
        masked_group[group_name] = np.asarray(frozen.predict(masked, raw_score=True))
        group_effect[group_name] = raw - masked_group[group_name]
    replay_error = float(np.max(np.abs(raw - baseline - sum(group_effect.values()))))
    if replay_error > 1e-9:
        raise ValueError("Frozen group effects do not reconstruct the decision")
    raw_rows, graph_source, incident = read_official_source(
        args.zip, ids, positions, values, contexts, all_names, dev,
        (visibility["train_support_dev_unique_edges"] if split_visible_mode
         else support_audit["union_unique_edges"] if corrected_mode
         else audit["distinct_directed_follow_edges"]),
        (("train", "dev", "support") if split_visible_mode else
         ("train", "dev", "test", "support") if corrected_mode else
         ("train", "dev", "test")))
    dev_batch = take(batch, dev)
    cards = []
    for k, i in enumerate(dev):
        uid = str(ids[i])
        row = raw_rows[k]
        terms = []
        for j, name in enumerate(names):
            original = values[i, all_names.index(name)]
            source = (graph_source[uid] if name in GRAPH_NAMES
                      else direct_query("dev", int(positions[i]), name, row))
            terms.append({
                "fact_name": name, "source_query": source,
                "source_value": float(original) if np.isfinite(original) else None,
                "missing": not bool(np.isfinite(original)),
                "upper_tail_score": float(dev_batch.upper[k, j]),
                "lower_tail_score": float(dev_batch.lower[k, j]),
                "reliability": float(dev_batch.reliability[k, j]),
                "upper_model_input": float(x_dev[k, j]),
                "lower_model_input": float(x_dev[k, j + len(names)]),
                "upper_monotone_direction": int(signs[j]),
                "lower_monotone_direction": int(signs[j + len(names)]),
            })
        prediction = int(probability[k] >= threshold)
        cards.append({
            "account_id": uid,
            "official_source_locator": f"Twibot-20/dev.json[{positions[i]}]",
            "source_record_without_label_sha256": canonical_hash(
                {name: value for name, value in row.items() if name != "label"}),
            "model_sha256": sha256(args.model),
            "reference_version": batch.reference_version,
            "tweet_sample_support_count": int(sample_count[i]),
            "fact_terms": terms,
            "all_zero_input_logit": float(baseline[k]),
            "threshold": threshold,
            "threshold_logit": threshold_logit,
            "baseline_margin": float(baseline[k] - threshold_logit),
            "group_logit_effects": {name: float(effect[k])
                                    for name, effect in group_effect.items()},
            "group_zeroed_logits": {name: float(value[k])
                                    for name, value in masked_group.items()},
            "full_logit": float(raw[k]),
            "full_margin": float(raw[k] - threshold_logit),
            "full_probability": float(probability[k]),
            "prediction": prediction,
            "evaluation_label": int(labels[i]),
            "prediction_correct": prediction == int(labels[i]),
        })
    exemplars = {}
    for category, truth, prediction in (("correct_bot", 1, 1),
                                        ("correct_human", 0, 0),
                                        ("human_wrongly_called_bot", 0, 1),
                                        ("bot_wrongly_called_human", 1, 0)):
        k = next(k for k, card in enumerate(cards)
                 if card["evaluation_label"] == truth and card["prediction"] == prediction)
        masked_terms = []
        for j, name in enumerate(names):
            masked = x_dev[k:k+1].copy()
            masked[:, [j, j + len(names)]] = 0.0
            without = float(frozen.predict(masked, raw_score=True)[0])
            masked_terms.append({"fact_name": name,
                                 "full_context_removal_logit_change": float(raw[k] - without),
                                 "class_changed": bool((expit(without) >= threshold) !=
                                                       cards[k]["prediction"])})
        card = cards[k]
        exemplars[category] = {
            "account_id": card["account_id"],
            "conditional_fact_removal": masked_terms,
            "conditional_effects_are_not_additive": True,
            "native_incident_edge_declarations": sorted(incident[card["account_id"]]),
        }
    report = {
        "scope": ("seed0 support-aware A/B group-tree dev decision audit on split-visible sampled graph"
                  if split_visible_mode else
                  "seed0 support-aware A/B group-tree dev decision audit on all provided sampled graph"
                  if corrected_mode else "prespecified seed0 support-aware A/B group-tree dev decision audit"),
        "candidate_test_evaluated": False,
        "source_archive_sha256": sha256(args.zip),
        "fact_cache_sha256": sha256(args.facts),
        "reference_artifact_sha256": sha256(args.reference),
        "full_source_audit_sha256": sha256(args.source_audit),
        "group_screen_sha256": sha256(args.split_visible_screen if split_visible_mode
                                      else args.corrected_screen if corrected_mode
                                      else args.group_screen),
        "frozen_model_path": str(args.model.resolve()),
        "frozen_model_sha256": sha256(args.model),
        "reference_version_with_support_rule": batch.reference_version,
        "support_rule": "six sampled-tweet facts weighted by n/(n+20)",
        "a_seed": SEED, "a_n": len(a), "b_n": len(b),
        "dev_n": len(dev), "dev_metric": score["dev"],
        "model_structure": "A-sign dual-tail monotone trees, each tree within one source group",
        "tree_counts_by_group": score["tree_counts_by_group"],
        "official_graph_edges_recomputed": (visibility["train_support_dev_unique_edges"]
                                             if split_visible_mode else
                                             support_audit["union_unique_edges"] if corrected_mode
                                             else audit["distinct_directed_follow_edges"]),
        "graph_scope": ("train/support/dev sampled neighbor declarations without test"
                        if split_visible_mode else
                        "all provided sampled neighbor declarations"
                        if corrected_mode else "labeled-row sampled neighbor declarations only"),
        "max_full_logit_group_reconstruction_error": replay_error,
        "correct_cards": sum(card["prediction_correct"] for card in cards),
        "wrong_cards": sum(not card["prediction_correct"] for card in cards),
        "exemplars": exemplars,
        "decision_cards": cards,
        "interpretation": "This certifies source facts and actual function behavior of the frozen group tree. Individual fact removal is conditional and does not add to group effects. The all-zero point is mathematical, not a real account. Dev threshold calibration and evaluation share accounts. Independent behavior truth and correction utility are separate questions.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("decision_cards", "exemplars")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
