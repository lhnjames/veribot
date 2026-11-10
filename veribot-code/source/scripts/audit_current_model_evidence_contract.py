"""Inventory the actual frozen classifier inputs and blind-task alignment.

This is a read-only contract audit. It does not evaluate predictions or labels.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GROUP_CERT = ROOT / "research/certificates/best_source_group_effects.json"
DEPTH_CERT = ROOT / "research/certificates/readable_reason_source_depth.json"
BOTSIM_SCHEMA = ROOT / "output/v3_botsim_observed_means_12/dev_only.json"
TWIBOT_MODEL = ROOT / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz"
PILOT = ROOT / "research/annotation_pilot_v1/rater_materials/blind_packet.jsonl"
OUT = ROOT / "research/certificates/current_model_evidence_contract.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    groups = json.loads(GROUP_CERT.read_text(encoding="utf-8"))
    depth = json.loads(DEPTH_CERT.read_text(encoding="utf-8"))
    botsim = json.loads(BOTSIM_SCHEMA.read_text(encoding="utf-8"))
    packet = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(packet) != 120 or len({row["blind_id"] for row in packet}) != 120:
        raise ValueError("Pilot task population changed")

    atomic_groups = botsim["atomic_groups"]
    atomic_names = [name for members in atomic_groups.values() for name in members]
    if len(atomic_names) != 22 or len(set(atomic_names)) != 22:
        raise ValueError("BotSim atomic source inventory changed")

    runs = []
    inventories: dict[str, list[str]] = {}
    for row in groups["runs"]:
        dataset, seed = row["dataset"], row["seed"]
        path = Path(row["group_effect_path"])
        if not path.is_absolute():
            path = ROOT / path
        if digest(path) != row["group_effect_sha256"]:
            raise ValueError(f"Group effect artifact changed: {dataset}/{seed}")
        with np.load(path, allow_pickle=False) as saved:
            names = saved["fact_names"].astype(str).tolist()
            group_names = saved["group_names"].astype(str).tolist()
        if len(names) != row["fact_count"] or len(set(names)) != len(names):
            raise ValueError(f"Fact vocabulary changed: {dataset}/{seed}")
        if set(group_names) != set(row["source_groups"]):
            raise ValueError(f"Group vocabulary changed: {dataset}/{seed}")
        if dataset in inventories and inventories[dataset] != names:
            raise ValueError(f"Seed vocabularies differ: {dataset}")
        inventories[dataset] = names
        runs.append({"dataset": dataset, "seed": seed, "fact_count": len(names),
                     "source_group_count": len(group_names),
                     "model_sha256": row["source_model_sha256"],
                     "group_effect_sha256": row["group_effect_sha256"]})

    if sorted((row["dataset"], row["seed"]) for row in runs) != [
        (dataset, seed) for dataset in ("botsim", "fox8", "mgtab")
        for seed in range(3)]:
        raise ValueError("Expected nine frozen-model run inventories")
    botsim_names = [f"{group}:{tail}_tail_max" for group in atomic_groups
                    for tail in ("high", "low")]
    if inventories["botsim"] != botsim_names:
        raise ValueError("BotSim group-tail inputs differ from source groups")
    if len(inventories["fox8"]) != 74:
        raise ValueError("Fox8 source fact inventory changed")
    mgtab_names = inventories["mgtab"]
    if (len(mgtab_names) != 150 or
            not all(name.startswith("published_profile_coordinate_")
                    for name in mgtab_names[:20]) or
            not all(name.startswith("published_tweet_projection_")
                    for name in mgtab_names[86:])):
        raise ValueError("MGTAB source-depth partitions changed")

    with np.load(TWIBOT_MODEL, allow_pickle=False) as saved:
        structured = [str(name).split(":", 1)[0]
                      for name in saved["structured_names"].astype(str)]
        vocab = saved["vocabulary"].astype(str)
        active = vocab[np.flatnonzero(saved["coefficient"][len(structured):])]
    if len(set(structured)) != 22 or len(active) != 109:
        raise ValueError("TwiBot sparse frozen input inventory changed")

    pilot_counts = Counter((row["dataset"], row["concept"]) for row in packet)
    named_input = {
        "twibot20": set(structured) | {f"unigram:{token}" for token in active},
        "botsim24": set(atomic_names) | set(inventories["botsim"]),
        "fox8": set(inventories["fox8"]),
        "mgtab": set(mgtab_names),
    }
    aligned = sum(count for (dataset, concept), count in pilot_counts.items()
                  if concept in named_input[dataset])
    if aligned:
        raise ValueError("Blind pilot has a newly aligned classifier input")

    totals = depth["totals"]
    if totals["accounts"] != 5400 or totals["claims"] != 28007:
        raise ValueError("Readable reason source population changed")
    if sum(sum(levels.values()) for levels in
           depth["source_levels_by_dataset"].values()) != totals["claims"]:
        raise ValueError("Source-depth claim counts do not sum to displayed claims")
    report = {
        "status": "frozen_input_and_source_depth_contract_checked",
        "scope": "current_saved_models_and_existing_blind_pilot",
        "source_sha256": {
            "group_effect_certificate": digest(GROUP_CERT),
            "source_depth_certificate": digest(DEPTH_CERT),
            "botsim_atomic_group_schema": digest(BOTSIM_SCHEMA),
            "twibot20_sparse_model": digest(TWIBOT_MODEL),
            "blind_pilot_packet": digest(PILOT),
        },
        "runs": runs,
        "model_input_contract": {
            "twibot20": {"structured_unique_facts": 22,
                         "active_literal_unigrams": 109,
                         "source_level": "published_profile_activity_sampled_graph_and_sampled_tweet_text"},
            "botsim": {"atomic_numeric_facts": 22,
                       "classifier_group_tail_concepts": 12,
                       "source_groups": {group: len(members)
                                         for group, members in atomic_groups.items()},
                       "source_level": "published_simulated_user_activity_post_and_native_reply_objects"},
            "fox8": {"numeric_source_facts": 74,
                     "source_level": "raw_profile_sampled_tweets_and_native_interaction_pointers"},
            "mgtab": {"published_profile_coordinates": 20,
                      "published_relation_and_neighbor_facts": 66,
                      "published_average_tweet_tensor_projections": 64,
                      "source_level": "published_profile_edge_and_average_tweet_tensors"},
        },
        "blind_pilot": {
            "tasks": len(packet),
            "concept_task_counts": [
                {"dataset": dataset, "concept": concept, "tasks": count}
                for (dataset, concept), count in sorted(pilot_counts.items())],
            "exact_named_input_tasks": aligned,
            "fox8_or_mgtab_tasks": sum(count for (dataset, _), count
                                       in pilot_counts.items()
                                       if dataset in {"fox8", "mgtab"}),
        },
        "readable_reason_scope": {
            "accounts": totals["accounts"],
            "displayed_claims": totals["claims"],
            "mgtab_projection_only_accounts":
                totals["accounts_with_only_projection_displayed"],
            "mgtab_projection_only_class_critical_accounts":
                totals["accounts_with_only_projection_class_critical"],
            "source_levels_by_dataset": depth["source_levels_by_dataset"],
        },
        "interpretation": (
            "These inputs and source depths support literal source and frozen-model "
            "response audits. The existing blind pilot does not independently validate "
            "the behavioral meaning of any named input in these models. A learned "
            "evidence-semantic truth estimator is not established by this contract."),
        "labels_or_test_metrics_read": False,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print({"runs": len(runs), "blind_tasks": len(packet),
           "exact_named_input_tasks": aligned,
           "displayed_claims": totals["claims"]})
    print(OUT)


if __name__ == "__main__":
    main()
