"""Trace every source-level target class flip through frozen HypBot layers."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
sys.path.insert(0, str(ROOT / "source"))
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, REPO, RUN,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402


SOURCE = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
ONE_CASE = ROOT / "research/certificates/hypbot_fox8_seven_pointer_neural_path.json"
OUT = ROOT / "research/certificates/hypbot_fox8_all_flip_neural_paths.json"
OUT_ALL = ROOT / "research/certificates/hypbot_fox8_all_incident_neural_paths.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("class_flips", "all_incident"),
                        default="class_flips")
    options = parser.parse_args()
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    one = json.loads(ONE_CASE.read_text(encoding="utf-8"))
    saved = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = saved["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    events = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))["events"]
    if (source["model_sha256"] != file_sha256(checkpoint_path) or
            source["graph_sha256"] != graph_meta["graph_sha256"] or
            source["raw_source_sha256"] != file_sha256(RAW) or
            source["source_index_sha256"] != file_sha256(EVENT_INDEX) or
            source["node_feature_sha256"] != hashes["feature_sha256"] or
            hashes["split_sha256"] != manifest["split_sha256"] or
            graph_meta != manifest["native_graph"] or
            one["model_sha256"] != source["model_sha256"] or
            one["graph_sha256"] != source["graph_sha256"]):
        raise ValueError("Frozen source/model/graph contract changed")
    index = {uid: i for i, uid in enumerate(ids)}
    test_ids = {ids[int(i)] for i in split["test"]}
    pair_events = defaultdict(list)
    for event in events:
        a, b = event["source_id"], event["target_id"]
        if a in index and b in index and a != b:
            pair_events[tuple(sorted((a, b)))].append(event)
    graph_pairs = {tuple(sorted((ids[int(u)], ids[int(v)])))
                   for u, v in edge.T.tolist()}
    if graph_pairs != set(pair_events):
        raise ValueError("Raw source pairs do not equal frozen native graph pairs")
    by_account = defaultdict(set)
    for pair in pair_events:
        for uid in pair:
            by_account[uid].add(pair)
    flips = [row for row in source["rows"]
             if row["original_class"] != row["edited_class"]]
    if len(flips) != source["test_accounts_whose_category_changed"] or len(flips) != 25:
        raise ValueError("Previously audited target-flip scope changed")
    if any(row["account_id"] not in test_ids for row in flips):
        raise ValueError("Flip target outside fixed test partition")
    selected = (flips if options.scope == "class_flips" else
                [row for row in source["rows"]
                 if row["incident_unique_undirected_pairs"] > 0])
    if options.scope == "all_incident" and len(selected) != \
            source["test_accounts_with_incident_source_pairs"]:
        raise ValueError("Incident-source population changed")

    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        original_adj = mean_adjacency(edge, len(ids))
        h1_old = model.sage1_act(model.sage1(x, original_adj))
        h2_old = model.sage2_act(model.sage2(h1_old, original_adj))
        node_readout, _ = model._readout(
            model.enc_node, model.warp_node, model.sectors_node, x)
        original_logits = model(x, original_adj)
        radius_old, direction_old = model.enc_graph(h2_old)
        std_old = radius_old.detach().std() + 1e-6

        def graph_readout(target: int, radius: torch.Tensor,
                          direction: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            unit = direction[target:target+1]
            gamma = model.warp_graph(unit)
            entropy, alignment, _ = model.sectors_graph(unit, gamma)
            r = radius[target:target+1] / scale
            h = entropy / (math.log(model.K) + 1e-6)
            return torch.stack((r, h, alignment, r * alignment,
                                gamma * alignment), dim=-1)

        def state(target: int, radius: torch.Tensor,
                  direction: torch.Tensor, scale: torch.Tensor) -> dict:
            graph = graph_readout(target, radius, direction, scale)
            logit = float(model.head(torch.cat((
                node_readout[target:target+1], graph), dim=-1)).squeeze().item())
            prob = float(torch.sigmoid(torch.tensor(logit)).item())
            return {"logit": logit, "probability": prob,
                    "class": int(prob >= manifest["threshold"])}

        rows = []
        max_error = 0.0
        for count, prior in enumerate(selected, 1):
            uid = prior["account_id"]
            target = index[uid]
            pairs = by_account[uid]
            pointers = sum(len(event["locators"]) for pair in pairs
                           for event in pair_events[pair])
            if pointers != prior["raw_target_user_pointers_cleared"]:
                raise ValueError("Source pointer count differs from event index")
            remove = {tuple(sorted((index[a], index[b]))) for a, b in pairs}
            keep = torch.as_tensor([
                tuple(sorted((int(u), int(v)))) not in remove
                for u, v in edge.T.tolist()], dtype=torch.bool)
            if int((~keep).sum()) != 2*prior["incident_unique_undirected_pairs"]:
                raise ValueError("Source-backed removed pair count changed")
            adj_new = mean_adjacency(edge[:, keep], len(ids))
            h1_new = model.sage1_act(model.sage1(x, adj_new))
            h2_new = model.sage2_act(model.sage2(h1_new, adj_new))
            edited_logits = model(x, adj_new)
            radius_new, direction_new = model.enc_graph(h2_new)
            std_new = radius_new.detach().std() + 1e-6
            states = {
                "original": state(target, radius_old, direction_old, std_old),
                "scale_only": state(target, radius_old, direction_old, std_new),
                "local_only": state(target, radius_new, direction_new, std_old),
                "both": state(target, radius_new, direction_new, std_new),
            }
            errors = [
                abs(states["original"]["logit"]-prior["original_logit"]),
                abs(states["both"]["logit"]-prior["edited_logit"]),
                abs(states["original"]["logit"]-float(original_logits[target])),
                abs(states["both"]["logit"]-float(edited_logits[target])),
            ]
            if "edited_target_logit_with_original_population_radius_scale" in prior:
                errors.append(abs(states["local_only"]["logit"]-
                    prior["edited_target_logit_with_original_population_radius_scale"]))
            err = max(errors)
            max_error = max(max_error, err)
            if err > 1e-6 or \
                    states["original"]["class"] != prior["original_class"] or \
                    states["both"]["class"] != prior["edited_class"] or \
                    ("edited_target_class_with_original_population_radius_scale" in prior
                     and states["local_only"]["class"] != prior[
                         "edited_target_class_with_original_population_radius_scale"]):
                raise ValueError(f"Frozen path does not replay target {uid}")
            a = states["original"]["logit"]
            b = states["scale_only"]["logit"]
            c = states["local_only"]["logit"]
            d = states["both"]["logit"]
            rows.append({
                "account_id": uid,
                "source_pointers_cleared": prior["raw_target_user_pointers_cleared"],
                "native_pairs_cleared": prior["incident_unique_undirected_pairs"],
                "target_sage1_embedding_l2_change": float(torch.linalg.vector_norm(
                    h1_new[target]-h1_old[target]).item()),
                "target_sage2_embedding_l2_change": float(torch.linalg.vector_norm(
                    h2_new[target]-h2_old[target]).item()),
                "global_graph_radius_std_change": float(std_new-std_old),
                "states": states,
                "local_only_logit_change": c-a,
                "global_scale_only_logit_change": b-a,
                "local_scale_interaction_logit": d-c-b+a,
                "total_logit_change": d-a,
                "maximum_replay_error": err,
            })
            if count % 5 == 0:
                print(f"traced {count}/{len(selected)}", flush=True)

    one_row = next(row for row in rows if row["account_id"] == one["account_id"])
    if max(abs(one_row["states"]["original"]["logit"]-
               one["four_path_states"]["local_old_scale_old"]["logit"]),
           abs(one_row["states"]["both"]["logit"]-
               one["four_path_states"]["local_new_scale_new"]["logit"])) > 1e-6:
        raise ValueError("Population trace differs from original case trace")
    report = {
        "status": ("all_source_flipped_hypbot_targets_neural_paths_traced"
                   if options.scope == "class_flips" else
                   "all_source_incident_hypbot_targets_neural_paths_traced"),
        "scope": ("all 25 target class flips in the 220 incident-source Fox8 seed0 test edits"
                  if options.scope == "class_flips" else
                  "all 220 source-incident Fox8 seed0 test target edits"),
        "source_operation": source["source_operation"],
        "source_certificate_sha256": file_sha256(SOURCE),
        "single_case_certificate_sha256": file_sha256(ONE_CASE),
        "model_sha256": file_sha256(checkpoint_path),
        "graph_sha256": graph_meta["graph_sha256"],
        "node_feature_sha256": hashes["feature_sha256"],
        "threshold": float(manifest["threshold"]),
        "targets": len(rows),
        "local_only_class_flips": sum(r["states"]["original"]["class"] !=
                                      r["states"]["local_only"]["class"]
                                      for r in rows),
        "global_scale_only_class_flips": sum(
            r["states"]["original"]["class"] !=
            r["states"]["scale_only"]["class"] for r in rows),
        "all_targets_with_both_sage_layers_changed": sum(
            r["target_sage1_embedding_l2_change"] > 1e-6 and
            r["target_sage2_embedding_l2_change"] > 1e-6 for r in rows),
        "maximum_manual_to_saved_logit_difference": max_error,
        "labels_loaded_by_data_loader_but_not_used_for_selection_or_analysis": True,
        "known_correct_source_repair": False,
        "behavioral_semantics_claimed": False,
        "rows": rows,
    }
    if options.scope == "all_incident":
        report.update({
            "target_class_flips": sum(r["states"]["original"]["class"] !=
                                      r["states"]["both"]["class"] for r in rows),
            "target_local_only_abs_logit_change_gt_1e_6": sum(
                abs(r["local_only_logit_change"]) > 1e-6 for r in rows),
            "target_global_scale_only_abs_logit_change_gt_1e_6": sum(
                abs(r["global_scale_only_logit_change"]) > 1e-6 for r in rows),
            "target_both_sage_layers_changed_gt_1e_6": sum(
                r["target_sage1_embedding_l2_change"] > 1e-6 and
                r["target_sage2_embedding_l2_change"] > 1e-6 for r in rows),
        })
        if report["target_class_flips"] != 25:
            raise ValueError("Full incident scope class flips differ from saved source report")
    output = OUT if options.scope == "class_flips" else OUT_ALL
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print({k: report[k] for k in (
        "targets", "local_only_class_flips", "global_scale_only_class_flips",
        "all_targets_with_both_sage_layers_changed",
        "maximum_manual_to_saved_logit_difference")})
    if options.scope == "all_incident":
        print({k: report[k] for k in (
            "target_class_flips", "target_local_only_abs_logit_change_gt_1e_6",
            "target_global_scale_only_abs_logit_change_gt_1e_6",
            "target_both_sage_layers_changed_gt_1e_6")})
    print(output)


if __name__ == "__main__":
    main()
