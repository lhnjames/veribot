"""Check every public-facing evidence card against frozen sources and model."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import joblib
import numpy as np

from gate0_shallow_tree import sha256_file


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_fox8_targeted_neighbor_20_concepts"
    manifest_path = folder / "evidence_cards_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    replay_path = root / "research/certificates/fox8_targeted_20_concept_ab_tree_replay.json"
    if manifest["replay_certificate_sha256"] != sha256_file(replay_path):
        raise ValueError("Card manifest differs from independent replay")
    source_audit = json.loads((root / "research/certificates/fox8_neighbor_volume_full_raw_source.json")
                              .read_text(encoding="utf-8"))
    trace_path = root / source_audit["trace_path"]
    if manifest["full_raw_source_trace_sha256"] != sha256_file(trace_path):
        raise ValueError("Raw source locator trace differs")
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        raw_trace = {row["account_id"]: row for row in
                     (json.loads(line) for line in stream)}
    lock = json.loads((folder / "selection_lock_ab_group_tree.json")
                      .read_text(encoding="utf-8"))
    results = []
    for seed in (0, 1, 2):
        run_dir = folder / f"seed{seed}"
        card_path = run_dir / "ab_group_tree_test_cards.jsonl.gz"
        if manifest["runs"][seed]["sha256"] != sha256_file(card_path):
            raise ValueError("Card artifact differs")
        with np.load(run_dir / "ab_group_tree_test_evidence.npz", allow_pickle=False) as arr:
            ids = arr["ids"].astype(str).tolist()
            names = arr["concept_names"].astype(str).tolist()
            facts = arr["retained_fact_names"].astype(str).tolist()
            values = arr["retained_raw_values"].copy()
            concept_values = arr["concept_scores"].copy()
            winners = arr["winning_fact_index"].copy()
            model_input = arr["model_input"].copy()
            shap = arr["shap"].copy()
            logit = arr["logit"].copy()
            probability = arr["probability"].copy()
            labels = arr["label"].copy()
        model = joblib.load(run_dir / "ab_group_tree.joblib")
        threshold = lock["threshold_by_seed"][str(seed)]
        group_names = [names[2*k].split(":", 1)[0] for k in range(10)]
        masked = np.empty((len(ids), 10))
        for k in range(10):
            x = model_input.copy()
            x[:, [2*k, 2*k+1, 20+2*k, 21+2*k]] = 0
            masked[:, k] = model.predict_proba(x)[:, 1]
        with gzip.open(card_path, "rt", encoding="utf-8") as stream:
            cards = [json.loads(line) for line in stream]
        if len(cards) != len(ids):
            raise ValueError("Account card count differs")
        concept_count, source_count = 0, 0
        changes = {name: 0 for name in group_names}
        for j, card in enumerate(cards):
            uid = ids[j]
            if (card["account_id"] != uid or card["label"] != int(labels[j])
                    or card["prediction"] != int(probability[j] >= threshold)
                    or abs(card["probability"] - probability[j]) > 1e-14
                    or abs(card["logit"] - logit[j]) > 1e-13
                    or len(card["evidence"]) != 20):
                raise ValueError("Account decision card differs")
            for k, item in enumerate(card["evidence"]):
                concept_count += 1
                winner = int(winners[j, k])
                expected_name = facts[winner] if winner >= 0 else None
                expected_value = (float(values[j, winner]) if winner >= 0
                                  and np.isfinite(values[j, winner]) else None)
                if (item["concept"] != names[k]
                        or abs(item["value"] - concept_values[j, k]) > 1e-14
                        or item["winning_source_fact"] != expected_name
                        or item["winning_source_value"] != expected_value
                        or abs(item["logit_attribution_from_model"] -
                               float(shap[j, k] + shap[j, k+20])) > 1e-13):
                    raise ValueError("Fact-to-concept or attribution card differs")
                if winner >= 0:
                    source_count += 1
                    locator = item["source_locator"]
                    if locator is None or ("account_id" in locator and
                                           locator["account_id"] != uid):
                        raise ValueError("Source account locator differs")
                    if expected_name == "native_interaction_neighbor_mean_statuses_count":
                        if locator["neighbor_profiles"] != raw_trace[uid]["raw_neighbor_values"]:
                            raise ValueError("Raw neighbor profile locator differs")
            if abs(sum(card["source_group_logit_attributions"].values()) +
                   card["intercept_logit"] - logit[j]) > 1e-9:
                raise ValueError("Card group contributions do not reconstruct logit")
            for k, name in enumerate(group_names):
                item = card["source_group_zero_mask"][name]
                flip = bool((masked[j, k] >= threshold) != (probability[j] >= threshold))
                if abs(item["probability"] - masked[j, k]) > 1e-14 or item["changes_class"] != flip:
                    raise ValueError("Source group mask card differs")
                changes[name] += int(flip)
        if (concept_count != manifest["runs"][seed]["concept_slots"]
                or source_count != manifest["runs"][seed]["source_locators"]
                or changes != manifest["runs"][seed]["group_zero_mask_changed_classes"]):
            raise ValueError("Card manifest totals differ")
        results.append({"seed": seed, "accounts_verified": len(cards),
                        "concepts_verified": concept_count,
                        "source_locators_verified": source_count,
                        "group_mask_decisions_verified": len(cards) * len(group_names)})
    certificate = {"scope": "Fox8_all_twenty_concept_account_cards_independently_verified",
                   "manifest_sha256": sha256_file(manifest_path),
                   "runs": results, "status": "all_account_cards_match_sources_and_frozen_model"}
    path = root / "research/certificates/fox8_targeted_20_concept_cards.json"
    path.write_text(json.dumps(certificate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
