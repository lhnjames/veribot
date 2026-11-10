"""Independently read every fixed report against its source and effect card."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.mgtab_source_path import resolve_mgtab_source_dir


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/best_decision_reports_full_test.json"
REASONS = ROOT / "research/certificates/best_decision_reasons_full_test.json"
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
OUT = ROOT / "research/certificates/best_decision_reports_readback.json"
CATEGORY_TEXT = {
    "single_fact_class_critical":
        "At least one displayed source input is individually class-critical.",
    "distributed_class_dependence":
        "No single displayed input changes the class when withheld, but setting all evidence inputs to zero does.",
    "no_class_change_from_all_evidence":
        "Setting all evidence inputs to zero leaves the class unchanged; displayed items affect the frozen score.",
}


def expected_source_claim(dataset: str, uid: str, item: dict) -> str:
    """Rebuild permissible literal wording without calling the report renderer."""
    name = item["source_fact"]
    value = float(item["source_value"])
    level = item["source_level"]
    query = item["source_query"]
    suffix = f"{name} = {value:.17g}."
    if dataset == "botsim":
        if level == "published_user_summary":
            observed = int(query["activity_object_count"])
            status = query["cross_source_status"]
            if status == "object_count_matches_declared_summary":
                if observed != value:
                    raise ValueError("published quantities marked matching differ")
                ending = f"The released activity file contains {observed} corresponding objects."
            elif status == "object_count_differs_from_declared_summary":
                if observed == value:
                    raise ValueError("published quantities marked different match")
                ending = (f"The released activity file contains {observed} "
                          "corresponding objects; the two published quantities differ.")
            else:
                raise ValueError("unknown cross-source status")
            return f"For account {uid}, the published user table declares {suffix} {ending}"
        prefix = {
            "published_user_profile": "the published user profile yields",
            "published_activity_objects": "the released post/comment objects yield",
            "published_native_reply_objects": "the released reply and parent objects yield",
        }.get(level)
    elif dataset == "fox8":
        prefix = {
            "raw_profile_tweets_or_native_interaction":
                "the cited raw account, sampled tweet, or native interaction records yield",
        }.get(level)
    elif dataset == "mgtab":
        prefix = {
            "published_averaged_tweet_tensor_train_only_projection":
                "the published average-tweet tensor and frozen train-only projection yield",
            "published_normalized_profile_tensor":
                "the published normalized profile tensor gives",
            "published_typed_relation_tensors_and_derived_fact":
                "the published typed relation tensors yield",
        }.get(level)
    else:
        raise ValueError("unknown dataset")
    if prefix is None:
        raise ValueError("unknown source level")
    return f"For account {uid}, {prefix} {suffix}"


def main() -> None:
    manifest = json.loads(CERT.read_text(encoding="utf-8"))
    reason_manifest = json.loads(REASONS.read_text(encoding="utf-8"))
    group_manifest = json.loads(GROUP_CERT.read_text(encoding="utf-8"))
    minimum_manifest = json.loads(MINIMUM_CERT.read_text(encoding="utf-8"))
    content_manifest = json.loads(CONTENT_CERT.read_text(encoding="utf-8"))
    content_readback = json.loads(CONTENT_READBACK.read_text(encoding="utf-8"))
    stability_manifest = json.loads(STABILITY_CERT.read_text(encoding="utf-8"))
    stability_readback = json.loads(STABILITY_READBACK.read_text(
        encoding="utf-8"))
    local_manifest = json.loads(LOCAL_CERT.read_text(encoding="utf-8"))
    local_readback = json.loads(LOCAL_READBACK.read_text(encoding="utf-8"))
    k5_manifest = json.loads(K5_CERT.read_text(encoding="utf-8"))
    k5_readback = json.loads(K5_READBACK.read_text(encoding="utf-8"))
    direction_manifest = json.loads(DIRECTION_CERT.read_text(encoding="utf-8"))
    if (manifest["direction_balance_certificate_sha256"] !=
            sha256_file(DIRECTION_CERT) or
            manifest["direction_balance_sidecar_sha256"] !=
            sha256_file(DIRECTION_ROWS) or
            direction_manifest["reason_certificate_sha256"] !=
            sha256_file(REASONS) or
            direction_manifest["account_sidecar_sha256"] !=
            sha256_file(DIRECTION_ROWS)):
        raise ValueError("balanced display certificate binding differs")
    with gzip.open(DIRECTION_ROWS, "rt", encoding="utf-8") as stream:
        balance_rows = [json.loads(line) for line in stream]
    balanced_by_account = {(row["dataset"], row["seed"], row["account_id"]): row
                           for row in balance_rows}
    if len(balance_rows) != 5400 or len(balanced_by_account) != 5400:
        raise ValueError("balanced display population differs")
    direction_by_run = {(row["dataset"], row["seed"]): row
                        for row in direction_manifest["runs"]}
    group_by_run = {(row["dataset"], row["seed"]): row
                    for row in group_manifest["runs"]}
    minimum_by_run = {(row["dataset"], row["seed"]): row
                      for row in minimum_manifest["runs"]}
    content_by_seed = {row["seed"]: row for row in content_manifest["runs"]}
    stability_by_seed = {row["seed"]: row for row in stability_manifest["runs"]}
    local_by_seed = {row["seed"]: row for row in local_manifest["runs"]}
    k5_by_seed = {row["seed"]: row for row in k5_manifest["runs"]}
    content_source_path = resolve_mgtab_source_dir(ROOT) / "features.pt"
    if (len(manifest["runs"]) != 9 or
            len(group_by_run) != 9 or len(minimum_by_run) != 9 or
            set(content_by_seed) != {0, 1, 2} or
            set(stability_by_seed) != {0, 1, 2} or
            set(local_by_seed) != {0, 1, 2} or
            set(k5_by_seed) != {0, 1, 2} or
            manifest["reason_card_certificate_sha256"] != sha256_file(REASONS) or
            manifest["source_group_certificate_sha256"] != sha256_file(GROUP_CERT) or
            manifest["minimum_subset_certificate_sha256"] !=
            sha256_file(MINIMUM_CERT) or
            manifest["mgtab_content_response_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            manifest["mgtab_content_response_readback_sha256"] !=
            sha256_file(CONTENT_READBACK) or
            content_readback["source_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            manifest["mgtab_content_stability_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            manifest["mgtab_content_stability_readback_sha256"] !=
            sha256_file(STABILITY_READBACK) or
            stability_readback["source_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            stability_manifest["prior_centroid_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            manifest["mgtab_content_local_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            manifest["mgtab_content_local_readback_sha256"] !=
            sha256_file(LOCAL_READBACK) or
            local_manifest["status"] !=
            "local_label_blind_published_content_donor_response" or
            local_readback["status"] !=
            "local_published_content_donor_rows_and_probabilities_read_back" or
            local_readback["source_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            local_manifest["prior_baseline_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            manifest["mgtab_content_local_k5_certificate_sha256"] !=
            sha256_file(K5_CERT) or
            manifest["mgtab_content_local_k5_readback_sha256"] !=
            sha256_file(K5_READBACK) or
            k5_manifest["status"] !=
            "five_nearest_published_content_donor_responses" or
            k5_readback["status"] !=
            "five_local_donor_rows_distances_and_probabilities_read_back" or
            k5_readback["source_certificate_sha256"] !=
            sha256_file(K5_CERT) or
            k5_manifest["prior_local_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            manifest["report_renderer_sha256"] != sha256_file(ROOT /
                "source/verifiable_bot/best_decision_report.py") or
            manifest["total_accounts"] != 5400):
        raise ValueError("report certificate is not bound to reason cards")
    runs = []
    for run, reason_run in zip(manifest["runs"], reason_manifest["runs"], strict=True):
        group_run = group_by_run[run["dataset"], run["seed"]]
        minimum_run = minimum_by_run[run["dataset"], run["seed"]]
        group_path = Path(group_run["group_effect_path"])
        if (sha256_file(group_path) != group_run["group_effect_sha256"] or
                run["group_effect_matrix_sha256"] !=
                group_run["group_effect_sha256"]):
            raise ValueError("source group effect binding differs")
        with np.load(group_path, allow_pickle=False) as group_matrix:
            group_ids = group_matrix["ids"].astype(str)
            group_names = group_matrix["group_names"].astype(str)
            group_probability = group_matrix["full_probability"].astype(float)
            group_without = group_matrix["without_group_probability"].astype(float)
            group_threshold = float(group_matrix["threshold"])
        minimum_path = Path(minimum_run["subset_matrix_path"])
        if (sha256_file(minimum_path) != minimum_run["subset_matrix_sha256"] or
                run["minimum_subset_matrix_sha256"] !=
                minimum_run["subset_matrix_sha256"]):
            raise ValueError("full source-group subset matrix differs")
        with np.load(minimum_path, allow_pickle=False) as subset_matrix:
            subset_ids = subset_matrix["ids"].astype(str)
            subset_masks = subset_matrix["subset_mask"].astype(int)
            subset_probability = subset_matrix["subset_probability"].astype(float)
            subset_full = subset_matrix["full_probability"].astype(float)
            subset_threshold = float(subset_matrix["threshold"])
        empty_positions = np.flatnonzero(subset_masks == 0)
        if (len(empty_positions) != 1 or
                not np.array_equal(subset_ids, group_ids) or
                subset_probability.shape[0] != len(group_ids) or
                not np.allclose(subset_full, group_probability,
                                rtol=0, atol=3e-16) or
                subset_threshold != group_threshold):
            raise ValueError("subset and source group matrix differ")
        empty_probability = subset_probability[:, int(empty_positions[0])]
        content_probabilities = None
        content_mask = None
        exemplar_probabilities = None
        exemplar_row = None
        local_probabilities = None
        local_rows = None
        local_distances = None
        centroid_distances = None
        k5_rows = None
        k5_distances = None
        k5_probabilities = None
        if run["dataset"] == "mgtab":
            content_run = content_by_seed[run["seed"]]
            content_path = Path(content_run["result_path"])
            if (sha256_file(content_path) != content_run["result_sha256"] or
                    sha256_file(content_source_path) !=
                    content_run["published_features_sha256"] or
                    run["source_counterfactual_matrix_sha256"] !=
                    content_run["result_sha256"]):
                raise ValueError("published content source response differs")
            with np.load(content_path, allow_pickle=False) as z:
                content_ids = z["ids"].astype(str)
                content_full = z["full_probability"].astype(float)
                content_probabilities = z[
                    "centroid_source_probability"].astype(float)
                content_mask = z[
                    "encoded_group_zero_probability"].astype(float)
                content_threshold = float(z["threshold"])
            content_index = group_names.tolist().index(
                "published_averaged_tweet_projection")
            if (not np.array_equal(content_ids, group_ids) or
                    np.max(np.abs(content_full-group_probability)) > 1e-10 or
                    np.max(np.abs(content_mask-group_without[:, content_index]))
                    > 1e-10 or content_threshold != group_threshold):
                raise ValueError("source response and encoded group differ")
            stability_run = stability_by_seed[run["seed"]]
            stability_path = Path(stability_run["result_path"])
            if (sha256_file(stability_path) != stability_run["result_sha256"]
                    or run["source_counterfactual_stability_matrix_sha256"]
                    != stability_run["result_sha256"] or
                    stability_run["prior_result_sha256"] !=
                    content_run["result_sha256"]):
                raise ValueError("observed train-row response differs")
            with np.load(stability_path, allow_pickle=False) as z:
                stability_ids = z["ids"].astype(str)
                stability_full = z["full_probability"].astype(float)
                stability_centroid = z["centroid_probability"].astype(float)
                exemplar_probabilities = z[
                    "train_exemplar_probability"].astype(float)
                stability_threshold = float(z["threshold"])
                exemplar_row = int(z["nearest_train_row"])
            if (not np.array_equal(stability_ids, group_ids) or
                    np.max(np.abs(stability_full-group_probability)) > 1e-10
                    or np.max(np.abs(stability_centroid-content_probabilities))
                    > 1e-10 or stability_threshold != group_threshold or
                    exemplar_row != stability_run["nearest_train_source_row"]):
                raise ValueError("two source replacement populations differ")
            local_run = local_by_seed[run["seed"]]
            local_path = Path(local_run["result_path"])
            if (sha256_file(local_path) != local_run["result_sha256"] or
                    run["source_counterfactual_local_matrix_sha256"] !=
                    local_run["result_sha256"] or
                    local_run["prior_result_sha256"] !=
                    stability_run["result_sha256"]):
                raise ValueError("local train-donor response differs")
            with np.load(local_path, allow_pickle=False) as z:
                local_ids = z["ids"].astype(str)
                local_full = z["full_probability"].astype(float)
                local_center = z["centroid_probability"].astype(float)
                local_global = z["global_donor_probability"].astype(float)
                local_probabilities = z["local_donor_probability"].astype(float)
                local_rows = z["local_donor_row"].astype(int)
                local_distances = z[
                    "local_donor_distance_in_64d_projection"].astype(float)
                centroid_distances = z[
                    "centroid_distance_in_64d_projection"].astype(float)
                local_threshold = float(z["threshold"])
            if (not np.array_equal(local_ids, group_ids) or
                    np.max(np.abs(local_full-group_probability)) > 1e-10 or
                    np.max(np.abs(local_center-content_probabilities)) > 1e-10
                    or np.max(np.abs(local_global-exemplar_probabilities))
                    > 1e-10 or local_threshold != group_threshold):
                raise ValueError("local donor fixed-test population differs")
            k5_run = k5_by_seed[run["seed"]]
            k5_path = Path(k5_run["result_path"])
            if (sha256_file(k5_path) != k5_run["result_sha256"] or
                    run["source_counterfactual_local_k5_matrix_sha256"] !=
                    k5_run["result_sha256"] or
                    k5_run["prior_result_sha256"] !=
                    local_run["result_sha256"]):
                raise ValueError("five nearest donor result differs")
            with np.load(k5_path, allow_pickle=False) as z:
                k5_ids = z["ids"].astype(str)
                k5_full = z["full_probability"].astype(float)
                k5_center = z["centroid_probability"].astype(float)
                k5_rows = z["donor_rows"].astype(int)
                k5_distances = z[
                    "donor_distances_in_64d_projection"].astype(float)
                k5_probabilities = z["donor_probabilities"].astype(float)
                k5_threshold = float(z["threshold"])
            if (not np.array_equal(k5_ids, group_ids) or
                    not np.array_equal(k5_rows[:, 0], local_rows) or
                    np.max(np.abs(k5_full-group_probability)) > 1e-10 or
                    np.max(np.abs(k5_center-content_probabilities)) > 1e-10
                    or np.max(np.abs(k5_probabilities[:, 0]-local_probabilities))
                    > 1e-10 or
                    np.max(np.abs(k5_distances[:, 0]-local_distances))
                    > 1e-10 or k5_threshold != group_threshold):
                raise ValueError("five nearest donor population differs")
        elif (run["source_counterfactual_matrix_sha256"] is not None or
              run["source_counterfactual_stability_matrix_sha256"] is not None or
              run["source_counterfactual_local_matrix_sha256"] is not None or
              run["source_counterfactual_local_k5_matrix_sha256"] is not None):
            raise ValueError("non-MGTAB source intervention differs")
        report_path = Path(run["reports_path"])
        reason_path = Path(reason_run["reason_cards_file"])
        if (run["dataset"] != reason_run["dataset"] or
                run["seed"] != reason_run["seed"] or
                sha256_file(report_path) != run["reports_sha256"] or
                sha256_file(reason_path) != run["reason_cards_sha256"]):
            raise ValueError("report identity or artifact hash differs")
        accounts = 0
        fixed_claims = 0
        discordant_bot_summary_claims = 0
        accounts_with_group_class_change = 0
        group_flip_without_single_fact = 0
        six_item_balanced_cards = 0
        reports_omitting_known_opposition = 0
        all_zero_same_class = 0
        source_counterfactual_count = 0
        with gzip.open(report_path, "rt", encoding="utf-8") as reports, \
                gzip.open(reason_path, "rt", encoding="utf-8") as reasons:
            for report_line, reason_line in zip(reports, reasons, strict=True):
                report = json.loads(report_line)
                reason = json.loads(reason_line)
                balance = balanced_by_account[
                    (run["dataset"], run["seed"], reason["account_id"])]
                original_items = reason["displayed_items"]
                items = balance["balanced_displayed_items"]
                if (items[:len(original_items)] != original_items or
                        len(items) != len(original_items) + int(
                            balance["opposing_omitted_from_display"]) or
                        balance["reason_card_sha256"] !=
                        reason_run["reason_cards_sha256"] or
                        balance["source_card_sha256"] !=
                        reason["source_card_sha256"] or
                        balance["prediction"] != reason["prediction"] or
                        balance["reason_category"] !=
                        reason["reason_category"]):
                    raise ValueError("balanced card does not extend original reason")
                predicted_sign = 1 if reason["prediction"] else -1
                displayed_opposition = any(
                    float(item["logit_difference_when_withheld"]) *
                    predicted_sign < -1e-10 for item in items)
                reports_omitting_known_opposition += int(
                    balance["opposing_count"] > 0 and
                    not displayed_opposition)
                six_item_balanced_cards += len(items) == 6
                if (report["dataset"] != reason["dataset"] or
                        report["seed"] != reason["seed"] or
                        report["account_id"] != reason["account_id"] or
                        report["prediction"] != reason["prediction"] or
                        report["probability"] != reason["probability"] or
                        report["threshold"] != reason["threshold"] or
                        abs(float(report["all_evidence_zero_probability"])
                            - float(empty_probability[accounts])) > 1e-14 or
                        abs(float(balance["all_evidence_zero_probability"])
                            - float(empty_probability[accounts])) > 1e-14 or
                        report["source_card_sha256"] != reason[
                            "source_card_sha256"] or
                        report["effect_matrix_sha256"] != reason[
                            "effect_matrix_sha256"] or
                        report["decision_dependency_category"] != reason[
                            "reason_category"] or
                        report["language_api_used"] is not False):
                    raise ValueError("fixed decision or source binding differs")
                claims = report["fixed_claims"]
                if (len(items) != len(claims) or
                        report["ordered_item_ids"] != [
                            f"reason_{i+1}" for i in range(len(items))]):
                    raise ValueError("claim budget or fixed order differs")
                uid = report["account_id"]
                p = float(report["probability"])
                t = float(report["threshold"])
                decision = (f"The frozen classifier assigns account {uid} the "
                            f"{'Bot' if report['prediction'] else 'Human'} class "
                            f"(Bot probability {p:.17g}; threshold {t:.17g}).")
                meaning = CATEGORY_TEXT[reason["reason_category"]]
                empty_p = float(empty_probability[accounts])
                baseline = (
                    "With all encoded evidence inputs set to zero, the same "
                    "frozen classifier assigns "
                    f"{'Bot' if empty_p >= t else 'Human'} "
                    f"(Bot probability {empty_p:.17g}). This is a mathematical "
                    "input baseline, not a reconstructed account.")
                all_zero_same_class += int(
                    (empty_p >= t) == bool(report["prediction"]))
                if (uid != group_ids[accounts] or
                        abs(p-group_probability[accounts]) > 3e-16 or
                        t != group_threshold or
                        report["group_effect_matrix_sha256"] !=
                        group_run["group_effect_sha256"]):
                    raise ValueError("readable group effect account differs")
                expected_groups = []
                for j, name in enumerate(group_names):
                    name = str(name)
                    without_group = float(group_without[accounts, j])
                    count = group_run["source_groups"][name].get(
                        "fact_count", group_run["source_groups"][name].get(
                            "concept_count"))
                    expected_groups.append({"name": name, "fact_count": count,
                        "probability_without_group": without_group,
                        "class_changes_when_withheld":
                            (without_group >= t) != bool(report["prediction"])})
                if report["source_group_effects"] != expected_groups:
                    raise ValueError("report source group effects differ")
                changing_groups = [item for item in expected_groups
                                   if item["class_changes_when_withheld"]]
                if changing_groups:
                    group_summary = " ".join(
                        f"Withholding source group {item['name']} from the frozen classifier "
                        f"changes the class (Bot probability {item['probability_without_group']:.17g})."
                        for item in changing_groups)
                else:
                    largest = max(expected_groups, key=lambda item: abs(
                        item["probability_without_group"]-p))
                    group_summary = (
                        "No single source group changes the class when withheld. "
                        f"Group {largest['name']} has the largest absolute score effect: "
                        f"Bot probability changes from {p:.17g} to "
                        f"{largest['probability_without_group']:.17g}.")
                accounts_with_group_class_change += bool(changing_groups)
                group_flip_without_single_fact += bool(changing_groups) and (
                    reason["class_critical_fact_count"] == 0)
                source_statement = ""
                if content_probabilities is not None:
                    source_p = float(content_probabilities[accounts])
                    exemplar_p = float(exemplar_probabilities[accounts])
                    local_p = float(local_probabilities[accounts])
                    encoded_p = float(content_mask[accounts])
                    source_change = (source_p >= t) != bool(report["prediction"])
                    exemplar_change = (exemplar_p >= t) != bool(
                        report["prediction"])
                    local_change = (local_p >= t) != bool(report["prediction"])
                    encoded_change = (encoded_p >= t) != bool(report["prediction"])
                    locator = (f"{content_source_path}#row={uid},"
                               "columns=20:788")
                    exemplar_locator = (
                        f"{content_source_path}#row={exemplar_row},"
                        "columns=20:788")
                    local_row = int(local_rows[accounts])
                    local_locator = (
                        f"{content_source_path}#row={local_row},"
                        "columns=20:788")
                    local_distance = float(local_distances[accounts])
                    centroid_distance = float(centroid_distances[accounts])
                    neighbor_rows = k5_rows[accounts].astype(int).tolist()
                    neighbor_probabilities = k5_probabilities[
                        accounts].astype(float).tolist()
                    neighbor_distances = k5_distances[
                        accounts].astype(float).tolist()
                    neighbor_locators = [
                        f"{content_source_path}#row={int(source_row)},"
                        "columns=20:788"
                        for source_row in neighbor_rows]
                    neighbor_flip_count = sum(
                        (value >= t) != bool(report["prediction"])
                        for value in neighbor_probabilities)
                    expected_source = {
                        "operation":
                            "replace_published_average_tweet_row_with_train_centroid",
                        "source_level": "published_average_tweet_tensor_row",
                        "source_ids": [locator],
                        "source_probability": source_p,
                        "exemplar_probability": exemplar_p,
                        "local_donor_probability": local_p,
                        "nearest_train_row": exemplar_row,
                        "nearest_train_source_id": exemplar_locator,
                        "local_donor_row": local_row,
                        "local_donor_source_id": local_locator,
                        "local_donor_distance_in_64d_projection":
                            local_distance,
                        "centroid_distance_in_64d_projection":
                            centroid_distance,
                        "local_k5_donor_rows": neighbor_rows,
                        "local_k5_donor_probabilities":
                            neighbor_probabilities,
                        "local_k5_donor_distances": neighbor_distances,
                        "local_k5_source_ids": neighbor_locators,
                        "local_k5_flip_count": neighbor_flip_count,
                        "encoded_group_zero_probability": encoded_p,
                        "source_class_changes": source_change,
                        "exemplar_class_changes": exemplar_change,
                        "local_donor_class_changes": local_change,
                        "encoded_class_changes": encoded_change,
                        "replacement_class_conclusion_stable":
                            source_change == exemplar_change,
                        "local_vs_centroid_class_conclusion_stable":
                            source_change == local_change,
                        "certificate_sha256": sha256_file(CONTENT_CERT),
                        "stability_certificate_sha256": sha256_file(
                            STABILITY_CERT),
                        "local_donor_certificate_sha256": sha256_file(
                            LOCAL_CERT),
                        "local_k5_certificate_sha256": sha256_file(K5_CERT),
                    }
                    if report["source_counterfactual"] != expected_source:
                        raise ValueError("readable source intervention differs")
                    replacement_statement = (
                        "The centroid and global-donor replacements agree on whether the class changes. "
                        if source_change == exemplar_change else
                        "The class-change conclusion differs between the centroid and global-donor replacements. ")
                    local_statement = (
                        "The local-donor and centroid replacements agree on whether the class changes. "
                        if source_change == local_change else
                        "The class-change conclusion differs between the local-donor and centroid replacements. ")
                    neighbor_statement = (
                        f"Among this account's five nearest published training content rows "
                        f"{neighbor_rows}, {neighbor_flip_count} of five replacements change the class; "
                        f"their projection distances range from {neighbor_distances[0]:.17g} "
                        f"to {neighbor_distances[-1]:.17g}. "
                        + ("The class response varies across these five donors. "
                           if 0 < neighbor_flip_count < 5 else
                           "The class-change outcome agrees across these five donors. "))
                    source_statement = (
                        "Replacing this account's published average-tweet "
                        "tensor row with the training-content centroid and "
                        "rerunning the frozen projection, reference, and "
                        f"classifier yields Bot probability {source_p:.17g}; "
                        f"the class {'changes' if source_change else 'stays the same'}. "
                        "Replacing that row with the published training "
                        f"content row {exemplar_row}, chosen by label-blind "
                        "nearest-centroid distance, yields Bot probability "
                        f"{exemplar_p:.17g}; the class "
                        f"{'changes' if exemplar_change else 'stays the same'}. "
                        "Replacing that row with this account's nearest published "
                        f"training content row {local_row} in the same frozen "
                        f"projection yields Bot probability {local_p:.17g}; "
                        f"the class {'changes' if local_change else 'stays the same'}. "
                        f"The local-donor projection distance is {local_distance:.17g}, "
                        f"versus {centroid_distance:.17g} for the centroid. "
                        "Zeroing the encoded content group instead yields "
                        f"Bot probability {encoded_p:.17g}; the class "
                        f"{'changes' if encoded_change else 'stays the same'}. "
                        + replacement_statement
                        + local_statement
                        + neighbor_statement
                        + "The tensor substitution is a defined "
                        "counterfactual, not a correction of a known error. "
                        f"[{locator}; global replacement source "
                        f"{exemplar_locator}; local replacement sources "
                        f"{', '.join(neighbor_locators)}]")
                    source_counterfactual_count += 1
                elif report["source_counterfactual"] is not None:
                    raise ValueError("non-MGTAB source response differs")
                sentences = []
                for index, (item, claim) in enumerate(zip(items, claims, strict=True)):
                    query = item["source_query"]
                    content = claim["claim"]
                    without = float(item["probability_without_fact"])
                    change_text = ("the predicted class changes." if item[
                        "class_changes_when_withheld"] else
                        "the predicted class stays the same.")
                    expected_claim = (expected_source_claim(
                        report["dataset"], uid, item) + " " +
                        f"With this frozen model input withheld, Bot probability "
                        f"changes from {p:.17g} to {without:.17g}; " +
                        change_text)
                    if item["role"] == "contrary_score_relevant":
                        if (index != 5 or
                                float(item["logit_difference_when_withheld"])
                                * predicted_sign >= -1e-10 or
                                item["class_changes_when_withheld"]):
                            raise ValueError("supplemental input is not contrary")
                        expected_claim += " This input effect opposes the predicted class."
                    if (claim["item_id"] != f"reason_{index+1}" or
                            claim["operation"] != query["operation"] or
                            claim["source_ids"] != query["source_ids"] or
                            content != expected_claim):
                        raise ValueError("readable claim differs from source or mask")
                    if item["source_level"] == "published_user_summary":
                        observed = int(query["activity_object_count"])
                        if (f"contains {observed} corresponding objects" not in content
                                or "published user table declares" not in content or
                                ("quantities differ" in content) != (
                                    query["cross_source_status"] ==
                                    "object_count_differs_from_declared_summary")):
                            raise ValueError("declared count loses cross-source status")
                        discordant_bot_summary_claims += (
                            query["cross_source_status"] ==
                            "object_count_differs_from_declared_summary")
                    if (item["source_level"] ==
                            "published_averaged_tweet_tensor_train_only_projection"
                            and "published average-tweet tensor" not in content):
                        raise ValueError("MGTAB tensor projection overstated")
                    sentences.append(f"{content} [{', '.join(claim['source_ids'])}]")
                expected_report = " ".join(part for part in
                    (decision, baseline, meaning, group_summary,
                     source_statement, " ".join(sentences)) if part)
                if report["report"] != expected_report:
                    raise ValueError("language layer inserted or removed a claim")
                accounts += 1
                fixed_claims += len(claims)
        direction_run = direction_by_run[run["dataset"], run["seed"]]
        if (accounts != run["accounts"] or
                fixed_claims != run["fixed_claims"] or
                six_item_balanced_cards != run["six_item_balanced_cards"] or
                six_item_balanced_cards != direction_run[
                    "six_item_balanced_cards"] or
                reports_omitting_known_opposition != 0):
            raise ValueError("full fixed-report population differs")
        if (all_zero_same_class != run["all_zero_same_class_accounts"] or
                all_zero_same_class != minimum_run[
                    "empty_input_same_class_count"]):
            raise ValueError("all-zero class count differs")
        if source_counterfactual_count != run[
                "source_counterfactual_accounts"]:
            raise ValueError("source intervention report count differs")
        if (accounts_with_group_class_change != group_run[
                "accounts_with_any_source_group_class_flip"] or
                group_flip_without_single_fact != group_run[
                    "accounts_with_group_flip_but_no_single_fact_flip"]):
            raise ValueError("report group dependence totals differ")
        runs.append({"dataset": run["dataset"], "seed": run["seed"],
                     "accounts": accounts, "fixed_claims": fixed_claims,
                     "accounts_with_group_class_change":
                         accounts_with_group_class_change,
                     "group_flip_without_single_fact":
                         group_flip_without_single_fact,
                     "discordant_bot_summary_claims":
                         discordant_bot_summary_claims,
            "six_item_balanced_cards": six_item_balanced_cards,
            "reports_omitting_known_opposition":
                reports_omitting_known_opposition,
            "all_zero_same_class_accounts": all_zero_same_class,
            "source_counterfactual_accounts":
                source_counterfactual_count})
        print({"dataset": run["dataset"], "seed": run["seed"],
               "accounts": accounts, "claims": fixed_claims}, flush=True)
    OUT.write_text(json.dumps({
        "status": "all_bidirectional_reports_match_fixed_source_and_mask_claims",
        "report_certificate_sha256": sha256_file(CERT),
        "total_accounts": sum(row["accounts"] for row in runs),
        "total_claims": sum(row["fixed_claims"] for row in runs),
        "discordant_bot_summary_claims": sum(
            row["discordant_bot_summary_claims"] for row in runs),
        "runs": runs,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
