"""Write compact MGTAB source-to-concept-to-decision cards for every test row."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_profile_schema import PROFILE_SCHEMA


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_targeted_20_concept_ab_group_tree"
    cert_path = root / "research/certificates/mgtab_targeted_20_concept_ab_group_tree_replay.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    report_path = folder / "fixed_test_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if cert["fixed_test_report_sha256"] != sha256_file(report_path):
        raise ValueError("Independent source-to-decision replay differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    features = data_dir / "features.pt"
    edges = data_dir / "edge_index.pt"
    edge_types = data_dir / "edge_type.pt"
    source_hashes = {"features": sha256_file(features),
                     "edges": sha256_file(edges),
                     "edge_types": sha256_file(edge_types)}
    if source_hashes["features"] != report["source_features_sha256"]:
        raise ValueError("Published MGTAB feature source differs")
    lock = json.loads((folder / "selection_lock.json").read_text(encoding="utf-8"))
    manifest = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        projection = root / ("output/v3_mgtab_ab_spline_content/selected/"
                             f"seed{seed}/projection.npz")
        projection_hash = sha256_file(projection)
        threshold = lock["threshold_by_seed"][str(seed)]
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as arr:
            ids = arr["ids"].astype(str).tolist()
            facts = arr["retained_fact_names"].astype(str).tolist()
            raw_values = arr["retained_raw_values"].copy()
            concepts = arr["concept_names"].astype(str).tolist()
            concept_values = arr["concept_scores"].copy()
            winners = arr["concept_winners"].copy()
            probability = arr["probability"].copy()
            logit = arr["logit"].copy()
            shap = arr["shap"].copy()
            group_shap = arr["group_shap"].copy()
            group_delta = arr["exact_group_delta"].copy()
            group_mask = arr["group_mask_probability"].copy()
            all_mask = arr["all_mask_probability"].copy()
            labels = arr["label"].copy()
        groups = [concepts[2*k].split(":", 1)[0] for k in range(10)]
        path = run_dir / "test_evidence_cards.jsonl.gz"
        located = 0
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for j, uid in enumerate(ids):
                evidence = []
                for k, concept in enumerate(concepts):
                    winner = int(winners[j, k])
                    name = facts[winner] if winner >= 0 else None
                    raw = float(raw_values[j, winner]) if winner >= 0 else None
                    if raw is not None and not np.isfinite(raw):
                        raw = None
                    locator = None
                    if name is not None:
                        located += 1
                        row_id = int(uid)
                        if name.startswith("published_profile_coordinate_"):
                            column = int(name[-2:])
                            locator = {
                                "level": "published_normalized_profile_tensor",
                                "features_sha256": source_hashes["features"],
                                "row": row_id, "column": column,
                                "paper_table_a5_alias": PROFILE_SCHEMA[column][0]}
                        elif name.startswith("published_tweet_projection_"):
                            locator = {
                                "level": "published_average_tweet_tensor_projection",
                                "features_sha256": source_hashes["features"],
                                "row": row_id, "input_columns": "20:788",
                                "projection_sha256": projection_hash,
                                "output_coordinate": int(name[-3:])}
                        else:
                            locator = {
                                "level": "published_typed_edge_and_profile_tensors",
                                "edge_index_sha256": source_hashes["edges"],
                                "edge_type_sha256": source_hashes["edge_types"],
                                "features_sha256": source_hashes["features"],
                                "account_row": row_id, "derived_fact": name}
                    evidence.append({
                        "concept": concept,
                        "value": float(concept_values[j, k]),
                        "winning_fact": name,
                        "winning_fact_value": raw,
                        "source_locator": locator,
                        "model_logit_attribution": float(shap[j, k] + shap[j, k+20])})
                group_effects = {
                    group: {"model_logit_attribution": float(group_shap[j, k]),
                            "exact_zero_mask_logit_difference": float(group_delta[j, k]),
                            "probability_without_group": float(group_mask[j, k]),
                            "changes_class_without_group": bool(
                                (group_mask[j, k] >= threshold) !=
                                (probability[j] >= threshold))}
                    for k, group in enumerate(groups)}
                card = {
                    "account_id": uid, "label": int(labels[j]),
                    "prediction": int(probability[j] >= threshold),
                    "probability": float(probability[j]),
                    "threshold": float(threshold),
                    "logit": float(logit[j]),
                    "intercept_logit": float(shap[j, -1]),
                    "zero_evidence_logit": float(logit[j] - group_delta[j].sum()),
                    "zero_evidence_probability": float(all_mask[j]),
                    "group_effects": group_effects,
                    "evidence": evidence,
                    "content_provenance_level": "published_average_tweet_tensor",
                }
                if (abs(sum(x["model_logit_attribution"] for x in group_effects.values()) +
                        card["intercept_logit"] - card["logit"]) > 1e-9):
                    raise ValueError("Card source-group logit contributions differ")
                stream.write(json.dumps(card, ensure_ascii=False) + "\n")
        manifest.append({"seed": seed, "account_cards": len(ids),
                         "concept_slots": len(ids) * len(concepts),
                         "source_locators": located,
                         "group_mask_decisions": len(ids) * len(groups),
                         "sha256": sha256_file(path)})
        print(json.dumps(manifest[-1]), flush=True)
    output = {"scope": "MGTAB_all_fixed_test_twenty_concept_source_cards",
              "independent_replay_sha256": sha256_file(cert_path),
              "source_hashes": source_hashes,
              "runs": manifest}
    (folder / "evidence_cards_manifest.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
