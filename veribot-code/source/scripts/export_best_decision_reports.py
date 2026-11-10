"""Export immutable readable reports for all selected-model fixed-test cards."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.best_decision_report import render_report
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
CARD_CERT = ROOT / "research/certificates/best_decision_reasons_full_test.json"
CARD_READBACK = ROOT / "research/certificates/best_decision_reasons_readback.json"
SOURCE_CERT = ROOT / "research/certificates/best_verified_evidence_full_test.json"
SOURCE_SELECTOR = ROOT / "research/certificates/best_evidence_source_selector_resolution.json"
GROUP_CERT = ROOT / "research/certificates/best_source_group_effects.json"
MINIMUM_CERT = ROOT / "research/certificates/minimal_sufficient_source_groups.json"
CONTENT_CERT = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
CONTENT_READBACK = ROOT / "research/certificates/mgtab_content_source_centroid_response_readback.json"
STABILITY_CERT = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"
STABILITY_READBACK = ROOT / "research/certificates/mgtab_content_source_baseline_stability_readback.json"
LOCAL_CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"
LOCAL_READBACK = ROOT / "research/certificates/mgtab_content_local_source_replacement_readback.json"
K5_CERT = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement.json"
K5_READBACK = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement_readback.json"
DIRECTION_CERT = ROOT / "research/certificates/best_reason_direction_balance.json"
DIRECTION_ROWS = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
OUT_DIR = ROOT / "output/v3_best_decision_reports"
OUT_CERT = ROOT / "research/certificates/best_decision_reports_full_test.json"


def main() -> None:
    cards = json.loads(CARD_CERT.read_text(encoding="utf-8"))
    readback = json.loads(CARD_READBACK.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CERT.read_text(encoding="utf-8"))
    selectors = json.loads(SOURCE_SELECTOR.read_text(encoding="utf-8"))
    groups = json.loads(GROUP_CERT.read_text(encoding="utf-8"))
    minimum = json.loads(MINIMUM_CERT.read_text(encoding="utf-8"))
    content = json.loads(CONTENT_CERT.read_text(encoding="utf-8"))
    content_readback = json.loads(CONTENT_READBACK.read_text(encoding="utf-8"))
    stability = json.loads(STABILITY_CERT.read_text(encoding="utf-8"))
    stability_readback = json.loads(STABILITY_READBACK.read_text(
        encoding="utf-8"))
    local = json.loads(LOCAL_CERT.read_text(encoding="utf-8"))
    local_readback = json.loads(LOCAL_READBACK.read_text(encoding="utf-8"))
    k5 = json.loads(K5_CERT.read_text(encoding="utf-8"))
    k5_readback = json.loads(K5_READBACK.read_text(encoding="utf-8"))
    direction = json.loads(DIRECTION_CERT.read_text(encoding="utf-8"))
    if (content["status"] !=
            "published_content_row_replacement_and_input_mask_compared" or
            content_readback["status"] !=
            "all_published_content_source_responses_read_back" or
            content_readback["source_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            content["balanced_rows_sha256"] != sha256_file(DIRECTION_ROWS) or
            content["group_certificate_sha256"] != sha256_file(GROUP_CERT) or
            stability["status"] !=
            "two_label_blind_published_content_replacements_compared" or
            stability_readback["status"] !=
            "all_published_content_replacement_stability_rows_read_back" or
            stability_readback["source_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            stability["prior_centroid_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            stability["prior_centroid_readback_sha256"] !=
            sha256_file(CONTENT_READBACK) or
            local["status"] !=
            "local_label_blind_published_content_donor_response" or
            local_readback["status"] !=
            "local_published_content_donor_rows_and_probabilities_read_back" or
            local_readback["source_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            local["prior_baseline_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            local["prior_baseline_readback_sha256"] !=
            sha256_file(STABILITY_READBACK) or
            k5["status"] !=
            "five_nearest_published_content_donor_responses" or
            k5_readback["status"] !=
            "five_local_donor_rows_distances_and_probabilities_read_back" or
            k5_readback["source_certificate_sha256"] !=
            sha256_file(K5_CERT) or
            k5["prior_local_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            k5["prior_local_readback_sha256"] !=
            sha256_file(LOCAL_READBACK)):
        raise ValueError("MGTAB source counterfactual lacks readback")
    if (direction["reason_certificate_sha256"] != sha256_file(CARD_CERT) or
            direction["account_sidecar_sha256"] != sha256_file(DIRECTION_ROWS) or
            len(direction["runs"]) != 9):
        raise ValueError("balanced decision display differs from source cards")
    with gzip.open(DIRECTION_ROWS, "rt", encoding="utf-8") as stream:
        balanced_rows = [json.loads(line) for line in stream]
    balanced_by_account = {(row["dataset"], row["seed"], row["account_id"]): row
                           for row in balanced_rows}
    if len(balanced_rows) != 5400 or len(balanced_by_account) != 5400:
        raise ValueError("balanced display population or IDs differ")
    direction_by_run = {(row["dataset"], row["seed"]): row
                        for row in direction["runs"]}
    group_by_run = {(row["dataset"], row["seed"]): row for row in groups["runs"]}
    minimum_by_run = {(row["dataset"], row["seed"]): row
                      for row in minimum["runs"]}
    content_by_seed = {row["seed"]: row for row in content["runs"]}
    stability_by_seed = {row["seed"]: row for row in stability["runs"]}
    local_by_seed = {row["seed"]: row for row in local["runs"]}
    k5_by_seed = {row["seed"]: row for row in k5["runs"]}
    if (set(content_by_seed) != {0, 1, 2} or
            set(stability_by_seed) != {0, 1, 2} or
            set(local_by_seed) != {0, 1, 2} or
            set(k5_by_seed) != {0, 1, 2}):
        raise ValueError("MGTAB source counterfactual runs differ")
    content_source_path = resolve_mgtab_source_dir(ROOT) / "features.pt"
    if (cards["total_accounts"] != 5400 or len(cards["runs"]) != 9 or
            len(group_by_run) != 9 or len(minimum_by_run) != 9 or
            readback["reason_certificate_sha256"] != sha256_file(CARD_CERT) or
            selectors["source_card_certificate_sha256"] !=
            sha256_file(SOURCE_CERT) or selectors["cards"] != 5400 or
            readback["total_displayed_source_items"] != 27000):
        raise ValueError("selected reasons lack independent source/effect readback")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in cards["runs"]:
        group_run = group_by_run[run["dataset"], run["seed"]]
        minimum_run = minimum_by_run[run["dataset"], run["seed"]]
        direction_run = direction_by_run[run["dataset"], run["seed"]]
        if (direction_run["accounts"] != run["accounts"] or
                direction_run["reason_cards_sha256"] !=
                run["reason_cards_sha256"]):
            raise ValueError("balanced run and frozen reason card differ")
        group_path = Path(group_run["group_effect_path"])
        if sha256_file(group_path) != group_run["group_effect_sha256"]:
            raise ValueError("frozen source group effect matrix differs")
        with np.load(group_path, allow_pickle=False) as group_data:
            group_ids = group_data["ids"].astype(str)
            group_names = group_data["group_names"].astype(str)
            group_full = group_data["full_probability"].astype(float)
            group_without = group_data["without_group_probability"].astype(float)
            group_threshold = float(group_data["threshold"])
        if (len(group_ids) != run["accounts"] or
                group_without.shape != (len(group_ids), len(group_names)) or
                set(group_names) != set(group_run["source_groups"])):
            raise ValueError("source group population or vocabulary differs")
        minimum_path = Path(minimum_run["subset_matrix_path"])
        if sha256_file(minimum_path) != minimum_run["subset_matrix_sha256"]:
            raise ValueError("full source-group subset matrix differs")
        with np.load(minimum_path, allow_pickle=False) as subset_data:
            subset_ids = subset_data["ids"].astype(str)
            subset_masks = subset_data["subset_mask"].astype(int)
            subset_probs = subset_data["subset_probability"].astype(float)
            subset_full = subset_data["full_probability"].astype(float)
            subset_threshold = float(subset_data["threshold"])
        empty_positions = np.flatnonzero(subset_masks == 0)
        if (len(empty_positions) != 1 or
                not np.array_equal(subset_ids, group_ids) or
                subset_probs.shape[0] != len(group_ids) or
                not np.allclose(subset_full, group_full, rtol=0, atol=3e-16)
                or subset_threshold != group_threshold):
            raise ValueError("source-group subset population or model differs")
        empty_probs = subset_probs[:, int(empty_positions[0])]
        content_probabilities = None
        content_matrix_sha256 = None
        exemplar_probabilities = None
        stability_matrix_sha256 = None
        exemplar_row = None
        local_probabilities = None
        local_rows = None
        local_distances = None
        centroid_distances = None
        local_matrix_sha256 = None
        k5_matrix_sha256 = None
        k5_rows = None
        k5_distances = None
        k5_probabilities = None
        if run["dataset"] == "mgtab":
            content_run = content_by_seed[run["seed"]]
            content_path = Path(content_run["result_path"])
            if (sha256_file(content_path) != content_run["result_sha256"] or
                    sha256_file(content_source_path) !=
                    content_run["published_features_sha256"]):
                raise ValueError("published content source response changed")
            with np.load(content_path, allow_pickle=False) as source_data:
                content_ids = source_data["ids"].astype(str)
                content_full = source_data["full_probability"].astype(float)
                content_probabilities = source_data[
                    "centroid_source_probability"].astype(float)
                content_mask = source_data[
                    "encoded_group_zero_probability"].astype(float)
                content_threshold = float(source_data["threshold"])
            content_j = group_names.astype(str).tolist().index(
                "published_averaged_tweet_projection")
            if (not np.array_equal(content_ids, group_ids) or
                    np.max(np.abs(content_full-group_full)) > 1e-10 or
                    np.max(np.abs(content_mask-group_without[:, content_j]))
                    > 1e-10 or content_threshold != group_threshold):
                raise ValueError("source response and model group matrix differ")
            content_matrix_sha256 = content_run["result_sha256"]
            stability_run = stability_by_seed[run["seed"]]
            stability_path = Path(stability_run["result_path"])
            if (sha256_file(stability_path) != stability_run["result_sha256"]
                    or stability_run["prior_result_sha256"] !=
                    content_run["result_sha256"] or
                    stability_run["published_features_sha256"] !=
                    content_run["published_features_sha256"]):
                raise ValueError("observed train-row response changed")
            with np.load(stability_path, allow_pickle=False) as source_data:
                stability_ids = source_data["ids"].astype(str)
                stability_full = source_data["full_probability"].astype(float)
                stability_centroid = source_data[
                    "centroid_probability"].astype(float)
                exemplar_probabilities = source_data[
                    "train_exemplar_probability"].astype(float)
                stability_threshold = float(source_data["threshold"])
                exemplar_row = int(source_data["nearest_train_row"])
            if (not np.array_equal(stability_ids, group_ids) or
                    np.max(np.abs(stability_full-group_full)) > 1e-10 or
                    np.max(np.abs(stability_centroid-content_probabilities))
                    > 1e-10 or stability_threshold != group_threshold or
                    exemplar_row != stability_run["nearest_train_source_row"]):
                raise ValueError("two published source operations differ")
            stability_matrix_sha256 = stability_run["result_sha256"]
            local_run = local_by_seed[run["seed"]]
            local_path = Path(local_run["result_path"])
            if (sha256_file(local_path) != local_run["result_sha256"] or
                    local_run["prior_result_sha256"] !=
                    stability_run["result_sha256"] or
                    local_run["published_features_sha256"] !=
                    content_run["published_features_sha256"]):
                raise ValueError("local published donor response changed")
            with np.load(local_path, allow_pickle=False) as source_data:
                local_ids = source_data["ids"].astype(str)
                local_full = source_data["full_probability"].astype(float)
                local_centroid = source_data[
                    "centroid_probability"].astype(float)
                local_global = source_data[
                    "global_donor_probability"].astype(float)
                local_probabilities = source_data[
                    "local_donor_probability"].astype(float)
                local_rows = source_data["local_donor_row"].astype(int)
                local_distances = source_data[
                    "local_donor_distance_in_64d_projection"].astype(float)
                centroid_distances = source_data[
                    "centroid_distance_in_64d_projection"].astype(float)
                local_threshold = float(source_data["threshold"])
            if (not np.array_equal(local_ids, group_ids) or
                    np.max(np.abs(local_full-group_full)) > 1e-10 or
                    np.max(np.abs(local_centroid-content_probabilities))
                    > 1e-10 or
                    np.max(np.abs(local_global-exemplar_probabilities))
                    > 1e-10 or local_threshold != group_threshold):
                raise ValueError("local donor fixed-test population differs")
            local_matrix_sha256 = local_run["result_sha256"]
            k5_run = k5_by_seed[run["seed"]]
            k5_path = Path(k5_run["result_path"])
            if (sha256_file(k5_path) != k5_run["result_sha256"] or
                    k5_run["prior_result_sha256"] !=
                    local_run["result_sha256"] or
                    k5_run["published_features_sha256"] !=
                    content_run["published_features_sha256"]):
                raise ValueError("five-neighbor published donor response changed")
            with np.load(k5_path, allow_pickle=False) as source_data:
                k5_ids = source_data["ids"].astype(str)
                k5_full = source_data["full_probability"].astype(float)
                k5_center = source_data[
                    "centroid_probability"].astype(float)
                k5_rows = source_data["donor_rows"].astype(int)
                k5_distances = source_data[
                    "donor_distances_in_64d_projection"].astype(float)
                k5_probabilities = source_data[
                    "donor_probabilities"].astype(float)
                k5_threshold = float(source_data["threshold"])
            if (not np.array_equal(k5_ids, group_ids) or
                    not np.array_equal(k5_rows[:, 0], local_rows) or
                    np.max(np.abs(k5_full-group_full)) > 1e-10 or
                    np.max(np.abs(k5_center-content_probabilities)) > 1e-10
                    or np.max(np.abs(k5_probabilities[:, 0]-local_probabilities))
                    > 1e-10 or
                    np.max(np.abs(k5_distances[:, 0]-local_distances))
                    > 1e-10 or k5_threshold != group_threshold):
                raise ValueError("five-neighbor fixed-test population differs")
            k5_matrix_sha256 = k5_run["result_sha256"]
        path = Path(run["reason_cards_file"])
        if sha256_file(path) != run["reason_cards_sha256"]:
            raise ValueError("reason card changed after exact-effect verification")
        target = OUT_DIR / f"{run['dataset']}_seed{run['seed']}.jsonl.gz"
        count = 0
        claims = 0
        all_zero_same_class = 0
        counterfactual_count = 0
        with gzip.open(path, "rt", encoding="utf-8") as stream, \
                target.open("wb") as sink, gzip.GzipFile(
                    filename="", mode="wb", fileobj=sink, mtime=0) as compressed:
            for position, line in enumerate(stream):
                card = json.loads(line)
                balance = balanced_by_account[
                    (run["dataset"], run["seed"], card["account_id"])]
                displayed = card["displayed_items"]
                balanced = balance["balanced_displayed_items"]
                if (balance["prediction"] != card["prediction"] or
                        balance["reason_category"] != card["reason_category"] or
                        balance["source_card_sha256"] !=
                        card["source_card_sha256"] or
                        balance["reason_card_sha256"] !=
                        run["reason_cards_sha256"] or
                        balanced[:len(displayed)] != displayed or
                        len(balanced) != len(displayed) + int(
                            balance["opposing_omitted_from_display"]) or
                        len(balanced) not in (5, 6) or
                        (len(balanced) == 6 and balanced[-1]["role"] !=
                         "contrary_score_relevant")):
                    raise ValueError("balanced item is not bound to original five")
                if (card["account_id"] != group_ids[position] or
                        abs(card["probability"]-group_full[position]) > 3e-16 or
                        card["threshold"] != group_threshold or
                        abs(float(balance["all_evidence_zero_probability"])
                            - float(empty_probs[position])) > 1e-14):
                    raise ValueError("reason and source group account differ")
                effects = [{"name": str(name),
                    "fact_count": group_run["source_groups"][str(name)].get(
                        "fact_count", group_run["source_groups"][str(name)].get(
                            "concept_count")),
                    "probability_without_group": float(group_without[position, j])}
                    for j, name in enumerate(group_names)]
                balanced_card = {**card, "displayed_items": balanced,
                    "all_evidence_zero_probability": float(empty_probs[position])}
                counterfactual = None
                if content_probabilities is not None:
                    counterfactual = {
                        "operation":
                            "replace_published_average_tweet_row_with_train_centroid",
                        "source_probability": float(content_probabilities[
                            position]),
                        "exemplar_probability": float(exemplar_probabilities[
                            position]),
                        "local_donor_probability": float(local_probabilities[
                            position]),
                        "nearest_train_row": exemplar_row,
                        "nearest_train_source_id": (
                            f"{content_source_path}#row={exemplar_row},"
                            "columns=20:788"),
                        "local_donor_row": int(local_rows[position]),
                        "local_donor_source_id": (
                            f"{content_source_path}#row={int(local_rows[position])},"
                            "columns=20:788"),
                        "local_donor_distance_in_64d_projection": float(
                            local_distances[position]),
                        "centroid_distance_in_64d_projection": float(
                            centroid_distances[position]),
                        "local_k5_donor_rows": k5_rows[position].tolist(),
                        "local_k5_donor_probabilities":
                            k5_probabilities[position].tolist(),
                        "local_k5_donor_distances":
                            k5_distances[position].tolist(),
                        "local_k5_source_ids": [
                            f"{content_source_path}#row={int(source_row)},"
                            "columns=20:788"
                            for source_row in k5_rows[position]],
                        "encoded_group_zero_probability": float(content_mask[
                            position]),
                        "source_ids": [f"{content_source_path}#row="
                                       f"{card['account_id']},columns=20:788"],
                        "certificate_sha256": sha256_file(CONTENT_CERT),
                        "stability_certificate_sha256": sha256_file(
                            STABILITY_CERT),
                        "local_donor_certificate_sha256": sha256_file(
                            LOCAL_CERT),
                        "local_k5_certificate_sha256": sha256_file(K5_CERT),
                    }
                report = render_report(balanced_card, use_api=False,
                    group_effects=effects,
                    group_effect_matrix_sha256=group_run["group_effect_sha256"],
                    source_counterfactual=counterfactual)
                if (report["account_id"] != card["account_id"] or
                        report["prediction"] != card["prediction"] or
                        len(report["fixed_claims"]) != len(balanced)):
                    raise ValueError("fixed report differs from reason card")
                claims += len(report["fixed_claims"])
                counterfactual_count += report["source_counterfactual"] is not None
                all_zero_same_class += int(
                    (empty_probs[position] >= group_threshold) ==
                    bool(card["prediction"]))
                compressed.write((json.dumps(report, ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
                count += 1
        if (count != run["accounts"] or
                claims != run["displayed_source_items"] +
                direction_run["six_item_balanced_cards"] or
                all_zero_same_class != minimum_run[
                    "empty_input_same_class_count"] or
                counterfactual_count != (count if run["dataset"] ==
                                         "mgtab" else 0)):
            raise ValueError("report population or claim count differs")
        runs.append({"dataset": run["dataset"], "seed": run["seed"],
                     "accounts": count, "fixed_claims": claims,
                     "reason_cards_sha256": run["reason_cards_sha256"],
                     "six_item_balanced_cards": direction_run[
                         "six_item_balanced_cards"],
                     "all_zero_same_class_accounts": all_zero_same_class,
                     "minimum_subset_matrix_sha256": minimum_run[
                         "subset_matrix_sha256"],
                     "source_counterfactual_accounts": counterfactual_count,
                     "source_counterfactual_matrix_sha256":
                         content_matrix_sha256,
                     "source_counterfactual_stability_matrix_sha256":
                         stability_matrix_sha256,
                     "source_counterfactual_local_matrix_sha256":
                         local_matrix_sha256,
                     "source_counterfactual_local_k5_matrix_sha256":
                         k5_matrix_sha256,
                     "group_effect_matrix_sha256": group_run["group_effect_sha256"],
                     "reports_path": str(target),
                     "reports_sha256": sha256_file(target)})
        print({"dataset": run["dataset"], "seed": run["seed"],
               "accounts": count, "claims": claims}, flush=True)
    OUT_CERT.write_text(json.dumps({
        "status": "all_best_decisions_have_bidirectional_immutable_source_reports",
        "scope": "nine_fixed_test_runs_5400_accounts",
        "language_api_used": False,
        "reason_card_certificate_sha256": sha256_file(CARD_CERT),
        "reason_readback_sha256": sha256_file(CARD_READBACK),
        "source_selector_certificate_sha256": sha256_file(SOURCE_SELECTOR),
        "source_group_certificate_sha256": sha256_file(GROUP_CERT),
        "minimum_subset_certificate_sha256": sha256_file(MINIMUM_CERT),
        "mgtab_content_response_certificate_sha256": sha256_file(CONTENT_CERT),
        "mgtab_content_response_readback_sha256": sha256_file(
            CONTENT_READBACK),
        "mgtab_content_stability_certificate_sha256": sha256_file(
            STABILITY_CERT),
        "mgtab_content_stability_readback_sha256": sha256_file(
            STABILITY_READBACK),
        "mgtab_content_local_certificate_sha256": sha256_file(LOCAL_CERT),
        "mgtab_content_local_readback_sha256": sha256_file(LOCAL_READBACK),
        "mgtab_content_local_k5_certificate_sha256": sha256_file(K5_CERT),
        "mgtab_content_local_k5_readback_sha256": sha256_file(K5_READBACK),
        "direction_balance_certificate_sha256": sha256_file(DIRECTION_CERT),
        "direction_balance_sidecar_sha256": sha256_file(DIRECTION_ROWS),
        "report_renderer_sha256": sha256_file(ROOT /
            "source/verifiable_bot/best_decision_report.py"),
        "total_accounts": sum(run["accounts"] for run in runs),
        "total_fixed_claims": sum(run["fixed_claims"] for run in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT_CERT)


if __name__ == "__main__":
    main()
