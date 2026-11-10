"""Write account-level source, concept, contribution, and mask cards."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np

from gate0_shallow_tree import sha256_file


def source_location(name: str, account_id: str, source: dict) -> dict:
    if name == "native_interaction_neighbor_mean_statuses_count":
        return {
            "raw_profile_gzip_sha256": source["raw_source_sha256"],
            "native_event_index_sha256": source["event_index_sha256"],
            "neighbor_profiles": source["neighbors"][account_id]["raw_neighbor_values"],
        }
    if name.startswith("native_"):
        return {"native_event_index_sha256": source["event_index_sha256"],
                "event_positions_zero_based": source["events"][account_id],
                "account_id": account_id}
    if name in {"followers_count", "friends_count", "statuses_count",
                "favourites_count", "listed_count", "description_len", "name_len",
                "screen_name_len", "has_location", "has_url", "has_description",
                "ff_ratio"}:
        return {"feature_table_sha256": source["published_feature_sha256"],
                "raw_gzip_line": source["raw_lines"][account_id],
                "account_id": account_id}
    return {"raw_tweet_archive_sha256": source["raw_source_sha256"],
            "raw_gzip_line": source["raw_lines"][account_id],
            "account_id": account_id}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    cert_path = root / "research/certificates/fox8_targeted_20_concept_ab_tree_replay.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    report = json.loads((folder / "fixed_test_ab_group_tree.json")
                        .read_text(encoding="utf-8"))
    audit = json.loads((root / "research/certificates/fox8_neighbor_volume_full_raw_source.json")
                       .read_text(encoding="utf-8"))
    trace_path = root / audit["trace_path"]
    if (cert["status"] != "all_accounts_concepts_group_contributions_and_masks_verified"
            or audit["trace_sha256"] != sha256_file(trace_path)):
        raise ValueError("Source-to-decision replay or raw locator trace differs")
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        neighbors = {item["account_id"]: item for item in
                     (json.loads(line) for line in stream)}
    raw_path = root / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    if sha256_file(raw_path) != audit["raw_source_sha256"]:
        raise ValueError("Raw account record source differs")
    raw_lines = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            uid = str(json.loads(line)["user_id"])
            raw_lines.setdefault(uid, line_number)
    event_path = root / "data/processed/fox8_v3/interaction_event_index.json"
    if sha256_file(event_path) != audit["event_index_sha256"]:
        raise ValueError("Native event source differs")
    events_by_account = {uid: [] for uid in neighbors}
    for position, event in enumerate(json.loads(event_path.read_text(encoding="utf-8"))["events"]):
        for uid in (str(event["source_id"]), str(event["target_id"])):
            if uid in events_by_account:
                events_by_account[uid].append(position)
    source = {"raw_source_sha256": audit["raw_source_sha256"],
              "event_index_sha256": audit["event_index_sha256"],
              "published_feature_sha256": audit["published_feature_sha256"],
              "neighbors": neighbors, "raw_lines": raw_lines,
              "events": events_by_account}
    manifest = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        saved_path = run_dir / "ab_group_tree_test_evidence.npz"
        row = report["runs"][seed]
        threshold = json.loads((folder / "selection_lock_ab_group_tree.json")
                               .read_text(encoding="utf-8"))["threshold_by_seed"][str(seed)]
        with np.load(saved_path, allow_pickle=False) as arr:
            ids = arr["ids"].astype(str).tolist()
            facts = arr["retained_fact_names"].astype(str).tolist()
            concept_names = arr["concept_names"].astype(str).tolist()
            raw = arr["retained_raw_values"].copy()
            concepts = arr["concept_scores"].copy()
            model_input = arr["model_input"].copy()
            winners = arr["winning_fact_index"].copy()
            group_contribution = arr["group_contribution"].copy()
            shap = arr["shap"].copy()
            logit = arr["logit"].copy()
            probability = arr["probability"].copy()
            neighbor_mask = arr["neighbor_mask_probability"].copy()
            all_mask = arr["all_mask_probability"].copy()
            labels = arr["label"].copy()
        model = joblib.load(run_dir / "ab_group_tree.joblib")
        if np.max(np.abs(model.predict_proba(model_input)[:, 1] - probability)) > 1e-14:
            raise ValueError("Frozen model probability differs from cards")
        group_names = [concept_names[2*k].split(":", 1)[0] for k in range(10)]
        group_mask_probabilities = np.empty((len(ids), 10))
        for k in range(10):
            masked = model_input.copy()
            masked[:, [2*k, 2*k+1, 20+2*k, 21+2*k]] = 0
            group_mask_probabilities[:, k] = model.predict_proba(masked)[:, 1]
        if np.max(np.abs(group_mask_probabilities[:, 9] - neighbor_mask)) > 1e-14:
            raise ValueError("Native neighbor group mask differs")
        group_decision_changes = {
            name: int(np.count_nonzero((group_mask_probabilities[:, k] >= threshold) !=
                                       (probability >= threshold)))
            for k, name in enumerate(group_names)}
        path = run_dir / "ab_group_tree_test_cards.jsonl.gz"
        located = 0
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            for j, uid in enumerate(ids):
                evidence = []
                for k, name in enumerate(concept_names):
                    winner = int(winners[j, k])
                    fact_name = facts[winner] if winner >= 0 else None
                    located += int(fact_name is not None)
                    raw_value = float(raw[j, winner]) if winner >= 0 else None
                    if raw_value is not None and not np.isfinite(raw_value):
                        raw_value = None
                    evidence.append({
                        "concept": name,
                        "value": float(concepts[j, k]),
                        "winning_source_fact": fact_name,
                        "winning_source_value": raw_value,
                        "source_locator": (source_location(fact_name, uid, source)
                                           if fact_name else None),
                        "logit_attribution_from_model": float(shap[j, k] + shap[j, k + 20]),
                    })
                card = {
                    "account_id": uid,
                    "label": int(labels[j]),
                    "prediction": int(probability[j] >= threshold),
                    "probability": float(probability[j]),
                    "threshold": float(threshold),
                    "intercept_logit": float(shap[j, -1]),
                    "logit": float(logit[j]),
                    "source_group_logit_attributions": {
                        group_names[k]: float(group_contribution[j, k])
                        for k in range(10)},
                    "source_group_zero_mask": {
                        group_names[k]: {
                            "probability": float(group_mask_probabilities[j, k]),
                            "changes_class": bool(
                                (group_mask_probabilities[j, k] >= threshold) !=
                                (probability[j] >= threshold))}
                        for k in range(10)},
                    "native_neighbor_volume_mask_probability": float(neighbor_mask[j]),
                    "native_neighbor_volume_mask_changes_class": bool(
                        (neighbor_mask[j] >= threshold) !=
                        (probability[j] >= threshold)),
                    "all_evidence_mask_probability": float(all_mask[j]),
                    "all_evidence_mask_changes_class": bool(
                        (all_mask[j] >= threshold) !=
                        (probability[j] >= threshold)),
                    "evidence": evidence,
                }
                if abs(sum(card["source_group_logit_attributions"].values()) +
                       card["intercept_logit"] - card["logit"]) > 1e-9:
                    raise ValueError("Card group attributions do not reconstruct score")
                stream.write(json.dumps(card, ensure_ascii=False) + "\n")
        manifest.append({"seed": seed, "cards": len(ids),
                         "concept_slots": len(ids) * len(concept_names),
                         "source_locators": located,
                         "sha256": sha256_file(path),
                         "group_zero_mask_changed_classes": group_decision_changes,
                         "neighbor_mask_changed_classes": row[
                             "neighbor_group_mask_changed_classes"]})
    output = {"scope": "Fox8_all_fixed_test_account_source_to_decision_cards",
              "replay_certificate_sha256": sha256_file(cert_path),
              "full_raw_source_trace_sha256": sha256_file(trace_path),
              "runs": manifest}
    out = folder / "evidence_cards_manifest.json"
    out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
