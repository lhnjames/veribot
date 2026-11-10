"""Check whether blind semantic tasks evaluate concepts used by frozen classifiers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
PACKET = ROOT / "research/annotation_pilot_v1/rater_materials/blind_packet.jsonl"
TWIBOT_MODEL = ROOT / "output/v3_twibot20_sparse_text_evidence/sparse_l1_c1_model.npz"
BOTSIM_SCHEMA = ROOT / "output/v3_botsim_observed_means_12/dev_only.json"
OUT = ROOT / "research/certificates/semantic_pilot_model_alignment.json"


def main() -> None:
    packet = [json.loads(line) for line in PACKET.read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if len(packet) != 120 or len({row["blind_id"] for row in packet}) != 120:
        raise ValueError("Blind packet differs from audited 120-task pilot")
    with np.load(TWIBOT_MODEL, allow_pickle=False) as model:
        structured = [str(name).split(":", 1)[0] for name in model["structured_names"]]
        vocabulary = model["vocabulary"].astype(str)
        active = vocabulary[np.flatnonzero(model["coefficient"][len(structured):])]
    botsim = json.loads(BOTSIM_SCHEMA.read_text(encoding="utf-8"))
    if not all(row["concept_count"] == 12 for row in botsim["runs"]):
        raise ValueError("BotSim model schema differs")
    botsim_facts = {fact for facts in botsim["atomic_groups"].values() for fact in facts}
    concept_inventory = {
        "twibot20": set(structured) | {f"unigram:{word}" for word in active},
        "botsim24": botsim_facts,
    }
    task_counts = Counter((row["dataset"], row["concept"]) for row in packet)
    rows = []
    for (dataset, concept), count in sorted(task_counts.items()):
        if dataset not in concept_inventory:
            raise ValueError("Unexpected packet dataset")
        rows.append({
            "dataset": dataset, "blind_task_concept": concept, "tasks": count,
            "exact_named_model_input": concept in concept_inventory[dataset],
            "model_dependency_status": (
                "not_a_frozen_classifier_input" if concept not in concept_inventory[dataset]
                else "named_input_only_further_dependency_check_needed"),
        })
    report = {
        "scope": "frozen_model_input_alignment_of_existing_semantic_blind_pilot",
        "pilot_packet_sha256": sha256_file(PACKET),
        "twibot_model_sha256": sha256_file(TWIBOT_MODEL),
        "botsim_model_schema_sha256": sha256_file(BOTSIM_SCHEMA),
        "tasks": len(packet),
        "counts_by_dataset_and_concept": rows,
        "direct_named_model_concept_task_count": sum(
            row["tasks"] for row in rows if row["exact_named_model_input"]),
        "fox8_and_mgtab_task_count": sum(
            row["tasks"] for row in packet if row["dataset"] in {"fox8", "mgtab"}),
        "model_input_examples": {
            "twibot20_structured": sorted(set(structured)),
            "twibot20_active_unigram_count": len(active),
            "botsim24_atomic_facts": sorted(botsim_facts),
        },
        "interpretation": (
            "The pilot probes semantic task definitions and rater feasibility. "
            "Its judgments cannot directly validate evidence inputs used by the "
            "current frozen TwiBot-20, BotSim, Fox8, or MGTAB classifiers."),
        "account_labels_or_test_metrics_read": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print({"tasks": report["tasks"],
           "direct_named_model_concept_task_count":
               report["direct_named_model_concept_task_count"],
           "fox8_and_mgtab_task_count": report["fox8_and_mgtab_task_count"]})
    print(OUT)


if __name__ == "__main__":
    main()
