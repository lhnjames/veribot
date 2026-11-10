"""Screen every singly sourced native relation incident to fixed boundary accounts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "source"), str(ROOT / "source/scripts"),
                str(ROOT / "source/hgt_import/code")]

from hgt_model import VerifiableHGT
from relnet_reimplementation import GatedRelNet
from scan_aaai_veribot_fox8_relation_responses import (
    event_key, remove_single_pointer_event,
)
from train_aaai_veribot_fox8 import load_inputs, sha256_file, tensor_hash
from verifiable_bot.aaai_fox8_native_graph import native_relation_edges

RAW = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
EVENTS = ROOT / "data/processed/fox8_v3/interaction_event_index.json"
RUN = ROOT / "output/aaai_veribot_fox8/seed0"
BOUNDARY = ROOT / "research/certificates/aaai_veribot_fox8_fixed_threshold_profile_screen.json"
OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_boundary_relation_screen.json"


def main() -> None:
    torch.set_num_threads(4)
    selected = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    accounts = [row["account_id"] for row in selected["accounts"]]
    payload = json.loads(EVENTS.read_text(encoding="utf-8"))
    if payload["raw_source_sha256"] != sha256_file(RAW):
        raise ValueError("source-index fingerprint differs from raw records")
    policy = json.loads((RUN / "decision_policy.json").read_text(encoding="utf-8"))
    predictions = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(RUN / "model.pt", map_location="cpu", weights_only=False)
    if (policy["checkpoint_sha256"] != sha256_file(RUN / "model.pt")
            or policy["saved_predictions_sha256"] != sha256_file(RUN / "test_predictions.json")
            or selected["checkpoint_sha256"] != policy["checkpoint_sha256"]
            or selected["threshold_logit"] != policy["logit_threshold"]):
        raise ValueError("frozen decision contract differs")
    x, _, split, ids, graph, pack = load_inputs()
    if checkpoint["input_contract"]["feature_sha256"] != tensor_hash(x):
        raise ValueError("frozen account input differs")
    test_ids = {ids[int(i)] for i in split["test"]}
    if set(accounts) - test_ids:
        raise ValueError("boundary screen contains a non-test account")
    by_id = {row["account_id"]: row for row in predictions["predictions"]}
    index = {uid: i for i, uid in enumerate(ids)}
    hgt = VerifiableHGT(pack["metadata"], {"user": x.shape[1]},
                        hidden_channels=128, out_channels=1,
                        num_layers=2, heads=4, dropout=0.3)
    relnet = GatedRelNet(x.shape[1], hidden_channels=128,
                         relation_names=tuple(pack["relnet_edges"]),
                         num_layers=2, dropout=0.3)
    hgt.load_state_dict(checkpoint["model_state"]["hgt"])
    relnet.load_state_dict(checkpoint["model_state"]["relnet"])
    hgt.eval()
    relnet.eval()

    def forward(edges: dict[str, torch.Tensor], target: int) -> dict[str, float]:
        typed = {("user", name, "user"): edge for name, edge in edges.items()}
        with torch.inference_mode():
            h = float(hgt({"user": x}, typed)["bot_logit"][target])
            r = float(relnet(x, edges)[target])
        return {"hgt_logit": h, "relnet_logit": r,
                "fused_logit": float(0.5 * (torch.tensor(h, dtype=torch.float32)
                                            + torch.tensor(r, dtype=torch.float32)))}

    baseline = pack["relnet_edges"]
    if any(tensor_hash(baseline[name]) != tensor_hash(graph[("user", name, "user")])
           for name in baseline):
        raise ValueError("HGT and RelNet do not share the source graph")
    incidents: dict[str, list[dict]] = {uid: [] for uid in accounts}
    for event in payload["events"]:
        if int(event["count"]) != 1 or len(event["locators"]) != 1:
            continue
        for uid in (str(event["source_id"]), str(event["target_id"])):
            if uid in incidents:
                incidents[uid].append(event)
    rows = []
    threshold = float(policy["logit_threshold"])
    for uid in accounts:
        target = index[uid]
        before = forward(baseline, target)
        if any(abs(before[name] - float(by_id[uid][name])) > 1e-5 for name in before):
            raise ValueError(f"saved prediction does not replay: {uid}")
        effects = []
        ordered = sorted(incidents[uid], key=lambda e: (*event_key(e),
                         int(e["locators"][0]["line"]),
                         int(e["locators"][0]["tweet_index"])))
        for event in ordered:
            edited = remove_single_pointer_event(baseline, index, event)
            rebuilt, _ = native_relation_edges(
                raw_path=RAW, event_index_path=EVENTS, account_ids=ids,
                excluded_events=frozenset((event_key(event),)))
            if any(not torch.equal(edited[name], rebuilt[name]) for name in edited):
                raise ValueError("direct edge edit differs from full source-index rebuild")
            after = forward(rebuilt, target)
            locator = event["locators"][0]
            effects.append({
                "source_key": list(event_key(event)),
                "raw_locator": locator,
                "before": before, "after": after,
                "hgt_delta_logit": after["hgt_logit"] - before["hgt_logit"],
                "relnet_delta_logit": after["relnet_logit"] - before["relnet_logit"],
                "fused_delta_logit": after["fused_logit"] - before["fused_logit"],
                "final_class_changed": ((after["fused_logit"] >= threshold)
                                        != (before["fused_logit"] >= threshold)),
                "edited_graph_sha256": {name: tensor_hash(edge)
                                        for name, edge in rebuilt.items()},
            })
        rows.append({"account_id": uid, "eligible_single_pointer_events": len(ordered),
                     "effects": effects})
        print(json.dumps({"account_id": uid, "events": len(ordered),
                          "flips": sum(e["final_class_changed"] for e in effects)}), flush=True)
    output = {
        "scope": "all singly sourced reply/retweet relations incident to the 20 preselected boundary accounts",
        "operation": "clear one raw target-user pointer; remove forward and reverse graph edges; preserve all node features",
        "boundary_screen_sha256": sha256_file(BOUNDARY),
        "event_index_sha256": sha256_file(EVENTS),
        "raw_source_sha256": sha256_file(RAW),
        "checkpoint_sha256": sha256_file(RUN / "model.pt"),
        "threshold_logit": threshold,
        "selected_accounts": len(accounts),
        "eligible_operations": sum(len(row["effects"]) for row in rows),
        "class_changing_operations": sum(e["final_class_changed"] for row in rows
                                         for e in row["effects"]),
        "accounts": rows,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"eligible_operations": output["eligible_operations"],
                      "class_changing_operations": output["class_changing_operations"]}),
          flush=True)


if __name__ == "__main__":
    main()
