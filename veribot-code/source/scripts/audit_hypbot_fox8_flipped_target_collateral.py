"""Audit other-account category changes from each source-backed target edit."""

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
from audit_hypbot_fox8_raw_edge_response import (  # noqa: E402
    DATA, EVENT_INDEX, RAW, RAW_AUDIT, REPO, RUN,
)
from export_hypbot_native_graph_paired import native_symmetric_graph  # noqa: E402
from train_v2_detector import file_sha256, load_data  # noqa: E402

SOURCE_REPORT = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
OUT = ROOT / "research/certificates/hypbot_fox8_flipped_target_collateral.json"
OUT_ALL = ROOT / "research/certificates/hypbot_fox8_all_target_collateral.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("flipped_targets", "all_incident"),
                        default="flipped_targets")
    options = parser.parse_args()
    sys.path.insert(0, str(REPO.resolve()))
    from hypbot.data import mean_adjacency
    from hypbot.model import HypBot
    from train import PRESETS

    torch.set_num_threads(4)
    source = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    event_index = json.loads(EVENT_INDEX.read_text(encoding="utf-8"))
    raw_audit = json.loads(RAW_AUDIT.read_text(encoding="utf-8"))
    frozen = json.loads((RUN / "test_predictions.json").read_text(encoding="utf-8"))
    manifest = frozen["manifest"]
    checkpoint_path = RUN / "model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = argparse.Namespace(dataset="fox8", data_dir=DATA, account_file=None,
                              raw_source=[RAW], knn_k=10)
    x, _, split, ids, _, _, hashes, _ = load_data(args, 0)
    edge, graph_meta = native_symmetric_graph(ROOT, "fox8", ids)
    if (source["raw_source_sha256"] != file_sha256(RAW)
            or source["source_index_sha256"] != file_sha256(EVENT_INDEX)
            or source["full_raw_interaction_audit_sha256"] != file_sha256(RAW_AUDIT)
            or source["model_sha256"] != file_sha256(checkpoint_path)
            or source["graph_sha256"] != graph_meta["graph_sha256"]
            or raw_audit["raw_source_sha256"] != file_sha256(RAW)
            or raw_audit["event_index_sha256"] != file_sha256(EVENT_INDEX)
            or hashes["feature_sha256"] != manifest["node_feature_sha256"]
            or hashes["split_sha256"] != manifest["split_sha256"]
            or checkpoint["split_sha256"] != hashes["split_sha256"]
            or graph_meta != manifest["native_graph"]):
        raise ValueError("Frozen model/source contract differs")
    index = {uid: i for i, uid in enumerate(ids)}
    pair_events = defaultdict(list)
    for event in event_index["events"]:
        a, b = event["source_id"], event["target_id"]
        if a in index and b in index and a != b:
            pair_events[tuple(sorted((a, b)))].append(event)
    graph_pairs = {tuple(sorted((ids[int(u)], ids[int(v)])))
                   for u, v in edge.T.tolist()}
    if graph_pairs != set(pair_events):
        raise ValueError("Source pairs differ from model graph")
    membership = {}
    for name in ("train", "val", "test"):
        for i in map(int, split[name]):
            membership[i] = name
    model = HypBot(x.shape[1], K=manifest["num_sectors"], gated=False,
                   **PRESETS["fox8"]["model"], graph_width=x.shape[1],
                   sparse_adj=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        baseline = model(x, mean_adjacency(edge, len(ids))).cpu().numpy()
        baseline_graph_embedding = model.aggregate(x, mean_adjacency(edge, len(ids)))
        radius_std = float(model.enc_graph(baseline_graph_embedding)[0].std())
        node_readout, _ = model._readout(model.enc_node, model.warp_node,
                                        model.sectors_node, x)

        def fixed_scale_logits(graph_embedding):
            radius, unit = model.enc_graph(graph_embedding)
            gamma = model.warp_graph(unit)
            entropy, alignment, _ = model.sectors_graph(unit, gamma)
            r_norm = radius / (radius_std + 1e-6)
            h_norm = entropy / (math.log(model.K) + 1e-6)
            graph_readout = torch.stack([
                r_norm, h_norm, alignment, r_norm * alignment,
                gamma * alignment], dim=-1)
            return model.head(torch.cat((node_readout, graph_readout),
                                        dim=-1)).squeeze(-1).cpu().numpy()

        if np.max(np.abs(fixed_scale_logits(baseline_graph_embedding) - baseline)) > 1e-6:
            raise ValueError("Original population scale does not replay baseline")
    threshold = float(manifest["threshold"])
    original_class = 1 / (1 + np.exp(-baseline)) >= threshold
    selected = ([r for r in source["rows"]
                 if r["original_class"] != r["edited_class"]]
                if options.scope == "flipped_targets" else
                [r for r in source["rows"]
                 if r["incident_unique_undirected_pairs"] > 0])
    if len(selected) != (25 if options.scope == "flipped_targets" else 220):
        raise ValueError("Selected target population differs")
    rows = []
    for row in selected:
        uid = row["account_id"]
        incident_pairs = [pair for pair in pair_events if uid in pair]
        remove_vertices = {tuple(sorted((index[a], index[b])))
                           for a, b in incident_pairs}
        keep = torch.as_tensor([
            tuple(sorted((int(u), int(v)))) not in remove_vertices
            for u, v in edge.T.tolist()], dtype=torch.bool)
        if int((~keep).sum()) != 2 * len(incident_pairs):
            raise ValueError("Incident graph edit size differs")
        with torch.no_grad():
            edited_edge = edge[:, keep]
            graph_embedding = model.aggregate(x, mean_adjacency(edited_edge, len(ids)))
            edited = model.forward_features(x, graph_embedding)[0].cpu().numpy()
            fixed = fixed_scale_logits(graph_embedding)
        j = index[uid]
        if (len(incident_pairs) != row["incident_unique_undirected_pairs"]
                or abs(float(baseline[j]) - row["original_logit"]) > 1e-6
                or abs(float(edited[j]) - row["edited_logit"]) > 1e-6
                or ("edited_target_logit_with_original_population_radius_scale" in row
                    and abs(float(fixed[j]) - row[
                        "edited_target_logit_with_original_population_radius_scale"]) > 1e-6)):
            raise ValueError("Target edit does not replay independent source report")
        new_class = 1 / (1 + np.exp(-edited)) >= threshold
        fixed_class = 1 / (1 + np.exp(-fixed)) >= threshold
        original_probability = 1 / (1 + np.exp(-baseline))
        edited_probability = 1 / (1 + np.exp(-edited))
        fixed_probability = 1 / (1 + np.exp(-fixed))
        class_flip_indices = np.flatnonzero(original_class != new_class)
        other_flip_indices = [int(i) for i in class_flip_indices if i != j]
        fixed_other_flip_indices = [int(i) for i in np.flatnonzero(
            original_class != fixed_class) if i != j]
        delta = edited - baseline
        graph_delta = torch.max(torch.abs(
            graph_embedding - baseline_graph_embedding), dim=1).values.cpu().numpy()
        rows.append({
            "target_account_id": uid,
            "incident_graph_pairs_cleared": len(incident_pairs),
            "source_pointers_cleared": row["raw_target_user_pointers_cleared"],
            "target_original_logit": float(baseline[j]),
            "target_edited_logit": float(edited[j]),
            "target_fixed_scale_edited_logit": float(fixed[j]),
            "target_original_class": int(original_class[j]),
            "target_edited_class": int(new_class[j]),
            "target_fixed_scale_edited_class": int(fixed_class[j]),
            "all_accounts_abs_logit_change_gt_1e_6": int(np.count_nonzero(
                np.abs(delta) > 1e-6)),
            "all_accounts_graph_embedding_change_gt_1e_6": int(np.count_nonzero(
                graph_delta > 1e-6)),
            "all_accounts_class_flips": len(class_flip_indices),
            "other_accounts_class_flips": len(other_flip_indices),
            "other_test_accounts_class_flips": sum(
                membership.get(i) == "test" for i in other_flip_indices),
            "other_accounts_class_flips_fixed_population_scale": len(
                fixed_other_flip_indices),
            "other_flipped_accounts": [{
                "account_id": ids[i],
                "split": membership.get(i, "unassigned"),
                "original_logit": float(baseline[i]),
                "original_probability_float32": float(original_probability[i]),
                "original_probability_minus_threshold": float(
                    original_probability[i] - threshold),
                "edited_logit": float(edited[i]),
                "edited_probability_float32": float(edited_probability[i]),
                "fixed_scale_edited_logit": float(fixed[i]),
                "fixed_scale_edited_probability_float32": float(
                    fixed_probability[i]),
                "original_class": int(original_class[i]),
                "edited_class": int(new_class[i]),
                "fixed_scale_edited_class": int(fixed_class[i]),
                "target_graph_embedding_max_abs_change": float(graph_delta[i]),
            } for i in other_flip_indices],
        })
        if len(rows) % 50 == 0:
            print(f"evaluated {len(rows)}/{len(selected)} target operations", flush=True)
    report = {
        "scope": ("all 25 Fox8 seed0 test targets that changed class"
                  if options.scope == "flipped_targets"
                  else "all 220 Fox8 seed0 test targets with source-backed native relations")
                 + " when all incident native relation pointers were cleared",
        "source_operation": source["source_operation"],
        "not_an_observed_source_correction": True,
        "official_paper_hypbot_graph_configuration": False,
        "source_report_sha256": file_sha256(SOURCE_REPORT),
        "raw_source_sha256": file_sha256(RAW),
        "event_index_sha256": file_sha256(EVENT_INDEX),
        "graph_sha256": graph_meta["graph_sha256"],
        "model_sha256": file_sha256(checkpoint_path),
        "threshold": threshold,
        "threshold_logit": float(math.log(threshold / (1 - threshold))),
        "total_model_accounts": len(ids),
        "target_operations": len(rows),
        "target_class_flips": sum(r["target_original_class"] !=
                                  r["target_edited_class"] for r in rows),
        "target_class_flips_fixed_population_scale": sum(
            r["target_original_class"] != r["target_fixed_scale_edited_class"]
            for r in rows),
        "operations_with_other_class_flips": sum(
            r["other_accounts_class_flips"] > 0 for r in rows),
        "total_other_class_flip_events": sum(
            r["other_accounts_class_flips"] for r in rows),
        "total_other_test_class_flip_events": sum(
            r["other_test_accounts_class_flips"] for r in rows),
        "operations_with_other_class_flips_fixed_population_scale": sum(
            r["other_accounts_class_flips_fixed_population_scale"] > 0
            for r in rows),
        "total_other_class_flip_events_fixed_population_scale": sum(
            r["other_accounts_class_flips_fixed_population_scale"]
            for r in rows),
        "distinct_other_flipped_account_ids": sorted({
            entry["account_id"] for r in rows
            for entry in r["other_flipped_accounts"]}),
        "other_flip_events_with_zero_graph_embedding_change": sum(
            entry["target_graph_embedding_max_abs_change"] == 0.0
            for r in rows for entry in r["other_flipped_accounts"]),
        "other_flip_events_original_probability_exactly_threshold": sum(
            entry["original_probability_minus_threshold"] == 0.0
            for r in rows for entry in r["other_flipped_accounts"]),
        "rows": rows,
    }
    destination = OUT if options.scope == "flipped_targets" else OUT_ALL
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
