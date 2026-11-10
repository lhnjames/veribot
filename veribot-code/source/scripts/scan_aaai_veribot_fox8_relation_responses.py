"""Measure native relation-source responses of frozen VeriBot HGT and RelNet."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source/hgt_import/code"))

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.fox8_native_interaction_audit import audit

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
FEATURES = ROOT / "data/processed/aaai_veribot/fox8_23/features.csv"
FACTS = ROOT / "data/processed/fox8_v3/native_interaction_facts.npz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_relation_response_population.json"


def event_key(event: dict) -> tuple[str, str, str]:
    return str(event["source_id"]), str(event["target_id"]), event["relation"]


def remove_single_pointer_event(
    baseline: dict[str, torch.Tensor], index: dict[str, int], event: dict,
) -> dict[str, torch.Tensor]:
    if event["count"] != 1 or len(event["locators"]) != 1:
        raise ValueError("one raw pointer must back the selected edge")
    source, target, relation = event_key(event)
    i, j = index[source], index[target]
    edited = dict(baseline)
    for name, first, second in ((relation, i, j), (f"rev_{relation}", j, i)):
        edge = baseline[name]
        selected = (edge[0] == first) & (edge[1] == second)
        if int(selected.sum()) != 1:
            raise ValueError("one native edge was not found in model input")
        edited[name] = edge[:, ~selected].contiguous()
    return edited


def direct_logits(hgt, relnet, x, edges, target):
    hgt_graph = {("user", name, "user"): value for name, value in edges.items()}
    with torch.inference_mode():
        h = float(hgt({"user": x}, hgt_graph)["bot_logit"][target])
        r = float(relnet(x, edges)[target])
    return h, r, (h + r) / 2


def main() -> None:
    torch.set_num_threads(4)
    source_audit = audit(RAW, FEATURES, FACTS)
    if source_audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators":
        raise ValueError("native relation source audit did not pass")
    x, _, split, ids, graph, pack = load_inputs()
    saved = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    if saved["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("frozen user input differs")
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(pack["relnet_edges"]),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(saved["model_state"]["hgt"])
    relnet.load_state_dict(saved["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()
    with torch.inference_mode():
        baseline_h = hgt({"user": x}, graph)["bot_logit"].numpy()
        baseline_r = relnet(x, pack["relnet_edges"]).numpy()
    predictions = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    by_id = {row["account_id"]: row for row in predictions["predictions"]}
    if any(abs(float(baseline_h[i]) - by_id[ids[int(i)]]["hgt_logit"]) > 1e-5 or
           abs(float(baseline_r[i]) - by_id[ids[int(i)]]["relnet_logit"]) > 1e-5
           for i in split["test"]):
        raise ValueError("saved frozen test logits do not replay")
    index = {uid: i for i, uid in enumerate(ids)}
    event_index = json.loads(EVENTS.read_text(encoding="utf-8"))["events"]
    incident: dict[str, list[dict]] = {}
    for event in event_index:
        if event["count"] != 1:
            continue
        for uid in (str(event["source_id"]), str(event["target_id"])):
            incident.setdefault(uid, []).append(event)
    results = []
    missing_pair = []
    for test_position, target in enumerate(map(int, split["test"]), 1):
        uid = ids[target]
        ordered = sorted(incident.get(uid, ()),
                         key=lambda row: (*event_key(row),
                                          row["locators"][0]["line"],
                                          row["locators"][0]["tweet_index"]))
        if len(ordered) < 2:
            missing_pair.append(uid)
            continue
        before = (float(baseline_h[target]), float(baseline_r[target]),
                  float((baseline_h[target] + baseline_r[target]) / 2))
        effects = []
        for event in ordered[:2]:
            edited = remove_single_pointer_event(pack["relnet_edges"], index, event)
            observed = direct_logits(hgt, relnet, x, edited, target)
            effects.append({
                "source_id": (f"fox8:{event['relation']}:{event['source_id']}:"
                              f"{event['target_id']}:{event['locators'][0]['line']}:"
                              f"{event['locators'][0]['tweet_index']}"),
                "source_key": list(event_key(event)),
                "raw_locator": event["locators"][0],
                "hgt_delta_logit": observed[0] - before[0],
                "relnet_delta_logit": observed[1] - before[1],
                "fused_delta_logit": observed[2] - before[2],
                "fused_class_changed": (before[2] >= 0) != (observed[2] >= 0),
            })
        results.append({"account_id": uid, "baseline_fused_logit": before[2],
                        "eligible_single_pointer_events": len(ordered),
                        "candidate": effects[0], "control": effects[1]})
        if test_position % 50 == 0:
            print(json.dumps({"processed_test_position": test_position,
                              "eligible_so_far": len(results)}), flush=True)
    if not results:
        raise ValueError("no test accounts have equal-budget native relation sources")
    candidate = np.asarray([row["candidate"]["fused_delta_logit"] for row in results])
    control = np.asarray([row["control"]["fused_delta_logit"] for row in results])
    summary = {
        "fixed_test_unique_accounts": len(split["test"]),
        "eligible_two_single_pointer_incident_events": len(results),
        "without_two_single_pointer_events": len(missing_pair),
        "candidate_nonzero_fused_response": int(np.count_nonzero(np.abs(candidate) > 1e-6)),
        "control_nonzero_fused_response": int(np.count_nonzero(np.abs(control) > 1e-6)),
        "candidate_class_changes": sum(row["candidate"]["fused_class_changed"] for row in results),
        "control_class_changes": sum(row["control"]["fused_class_changed"] for row in results),
        "both_single_pointer_edits_change_class": sum(
            row["candidate"]["fused_class_changed"] and
            row["control"]["fused_class_changed"] for row in results),
        "candidate_larger_absolute_delta_than_control": int(
            np.count_nonzero(np.abs(candidate) > np.abs(control))),
        "candidate_median_absolute_delta": float(np.median(np.abs(candidate))),
        "control_median_absolute_delta": float(np.median(np.abs(control))),
        "candidate_max_absolute_delta": float(np.max(np.abs(candidate))),
        "control_max_absolute_delta": float(np.max(np.abs(control))),
    }
    output = {
        "selection_rule": "for each fixed test account, choose lexicographically first two distinct incident raw relation events with exactly one target-user pointer each, before model-output inspection",
        "operation": "null one raw reply or retweet target-user pointer; delete its forward and reverse typed edges; keep named account features unchanged",
        "source_audit": source_audit,
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "event_index_sha256": sha256_file(EVENTS),
        "summary": summary, "ineligible_account_ids": missing_pair,
        "accounts": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
