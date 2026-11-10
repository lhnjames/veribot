"""Join every readable report item to its source term and frozen input effect."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
REPORT_CERT = ROOT / "research/certificates/best_decision_reports_full_test.json"
REPORT_READBACK = ROOT / "research/certificates/best_decision_reports_readback.json"
SOURCE_CERT = ROOT / "research/certificates/best_verified_evidence_full_test.json"
SOURCE_READBACK = ROOT / "research/certificates/best_verified_evidence_full_test_readback.json"
SOURCE_SELECTOR = ROOT / "research/certificates/best_evidence_source_selector_resolution.json"
BALANCE_CERT = ROOT / "research/certificates/best_reason_direction_balance.json"
BALANCE_ROWS = ROOT / "research/certificates/best_reason_direction_balance.jsonl.gz"
CONTENT_CERT = ROOT / "research/certificates/mgtab_content_source_centroid_response.json"
CONTENT_READBACK = ROOT / "research/certificates/mgtab_content_source_centroid_response_readback.json"
STABILITY_CERT = ROOT / "research/certificates/mgtab_content_source_baseline_stability.json"
STABILITY_READBACK = ROOT / "research/certificates/mgtab_content_source_baseline_stability_readback.json"
LOCAL_CERT = ROOT / "research/certificates/mgtab_content_local_source_replacement.json"
LOCAL_READBACK = ROOT / "research/certificates/mgtab_content_local_source_replacement_readback.json"
K5_CERT = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement.json"
K5_READBACK = ROOT / "research/certificates/mgtab_content_local_k5_source_replacement_readback.json"
OUT = ROOT / "research/certificates/bidirectional_report_source_model_chain.json"


def effect_path(dataset: str, seed: int) -> Path:
    name = (f"botsim_observed_seed{seed}.npz" if dataset == "botsim"
            else f"{dataset}_seed{seed}.npz")
    return ROOT / "output/v3_best_exact_fact_effects" / name


def main() -> None:
    report = json.loads(REPORT_CERT.read_text(encoding="utf-8"))
    report_readback = json.loads(REPORT_READBACK.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CERT.read_text(encoding="utf-8"))
    source_readback = json.loads(SOURCE_READBACK.read_text(encoding="utf-8"))
    selector = json.loads(SOURCE_SELECTOR.read_text(encoding="utf-8"))
    balance = json.loads(BALANCE_CERT.read_text(encoding="utf-8"))
    content = json.loads(CONTENT_CERT.read_text(encoding="utf-8"))
    content_readback = json.loads(CONTENT_READBACK.read_text(encoding="utf-8"))
    stability = json.loads(STABILITY_CERT.read_text(encoding="utf-8"))
    stability_readback = json.loads(STABILITY_READBACK.read_text(
        encoding="utf-8"))
    local = json.loads(LOCAL_CERT.read_text(encoding="utf-8"))
    local_readback = json.loads(LOCAL_READBACK.read_text(encoding="utf-8"))
    k5 = json.loads(K5_CERT.read_text(encoding="utf-8"))
    k5_readback = json.loads(K5_READBACK.read_text(encoding="utf-8"))
    if (report["status"] !=
            "all_best_decisions_have_bidirectional_immutable_source_reports" or
            report_readback["status"] !=
            "all_bidirectional_reports_match_fixed_source_and_mask_claims" or
            source["status"] !=
            "all_selected_fixed_test_accounts_have_source_bound_evidence_cards" or
            source_readback["status"] !=
            "all_full_test_cards_read_back_and_reconstructed" or
            balance["status"] !=
            "all_current_reason_cards_direction_display_audited" or
            content["status"] !=
            "published_content_row_replacement_and_input_mask_compared" or
            content_readback["status"] !=
            "all_published_content_source_responses_read_back" or
            content_readback["source_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            report["mgtab_content_response_certificate_sha256"] !=
            sha256_file(CONTENT_CERT) or
            report["mgtab_content_response_readback_sha256"] !=
            sha256_file(CONTENT_READBACK) or
            stability["status"] !=
            "two_label_blind_published_content_replacements_compared" or
            stability_readback["status"] !=
            "all_published_content_replacement_stability_rows_read_back" or
            stability_readback["source_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            report["mgtab_content_stability_certificate_sha256"] !=
            sha256_file(STABILITY_CERT) or
            report["mgtab_content_stability_readback_sha256"] !=
            sha256_file(STABILITY_READBACK) or
            local["status"] !=
            "local_label_blind_published_content_donor_response" or
            local_readback["status"] !=
            "local_published_content_donor_rows_and_probabilities_read_back" or
            local_readback["source_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            report["mgtab_content_local_certificate_sha256"] !=
            sha256_file(LOCAL_CERT) or
            report["mgtab_content_local_readback_sha256"] !=
            sha256_file(LOCAL_READBACK) or
            k5["status"] !=
            "five_nearest_published_content_donor_responses" or
            k5_readback["status"] !=
            "five_local_donor_rows_distances_and_probabilities_read_back" or
            k5_readback["source_certificate_sha256"] !=
            sha256_file(K5_CERT) or
            report["mgtab_content_local_k5_certificate_sha256"] !=
            sha256_file(K5_CERT) or
            report["mgtab_content_local_k5_readback_sha256"] !=
            sha256_file(K5_READBACK) or
            report_readback["report_certificate_sha256"] !=
            sha256_file(REPORT_CERT) or
            report["direction_balance_certificate_sha256"] !=
            sha256_file(BALANCE_CERT) or
            report["source_selector_certificate_sha256"] !=
            sha256_file(SOURCE_SELECTOR) or
            source_readback["source_export_certificate_sha256"] !=
            sha256_file(SOURCE_CERT) or
            selector["status"] !=
            "all_evidence_source_selectors_resolve_to_published_records" or
            selector["source_card_certificate_sha256"] !=
            sha256_file(SOURCE_CERT) or selector["cards"] != 5400 or
            balance["account_sidecar_sha256"] != sha256_file(BALANCE_ROWS) or
            report["direction_balance_sidecar_sha256"] !=
            sha256_file(BALANCE_ROWS) or
            len(report["runs"]) != 9 or len(source["runs"]) != 9):
        raise ValueError("report/source/balance certificate binding differs")
    balance_by_run = {(row["dataset"], row["seed"]): row
                      for row in balance["runs"]}
    totals = Counter()
    per_run = []
    max_prob_diff = 0.0
    max_effect_diff = 0.0
    with gzip.open(BALANCE_ROWS, "rt", encoding="utf-8") as balance_stream:
        for report_run, source_run in zip(report["runs"], source["runs"],
                                          strict=True):
            dataset, seed = report_run["dataset"], report_run["seed"]
            if ((dataset, seed) != (source_run["dataset"],
                                   source_run["seed"])):
                raise ValueError("report and source run differ")
            balance_run = balance_by_run[dataset, seed]
            if (balance_run["source_cards_sha256"] !=
                    source_run["cards_sha256"] or
                    balance_run["reason_cards_sha256"] !=
                    report_run["reason_cards_sha256"]):
                raise ValueError("source or reason card differs across certificates")
            rpath = Path(report_run["reports_path"])
            spath = Path(source_run["cards_file"])
            epath = effect_path(dataset, seed)
            if (sha256_file(rpath) != report_run["reports_sha256"] or
                    sha256_file(spath) != source_run["cards_sha256"] or
                    sha256_file(epath) != balance_run[
                        "effect_matrix_sha256"]):
                raise ValueError("underlying report, source or effect changed")
            with np.load(epath, allow_pickle=False) as matrix:
                ids = matrix["ids"].astype(str).tolist()
                names = matrix["concept_names" if dataset == "botsim"
                               else "fact_names"].astype(str).tolist()
                full_probability = matrix["full_probability"].astype(float)
                without_probability = matrix["without_probability"].astype(float)
                effect = (matrix["logit_difference"].astype(float)
                          if dataset == "botsim" else
                          matrix["full_logit"].astype(float)[:, None] -
                          matrix["without_logit"].astype(float))
                threshold = float(matrix["threshold"])
            by_name = {name: j for j, name in enumerate(names)}
            if (len(by_name) != len(names) or len(ids) != report_run["accounts"]
                    or without_probability.shape != (len(ids), len(names))
                    or effect.shape != without_probability.shape):
                raise ValueError("exact-effect population or vocabulary differs")
            run_counts = Counter()
            with gzip.open(rpath, "rt", encoding="utf-8") as reports, \
                    gzip.open(spath, "rt", encoding="utf-8") as sources:
                for i, (report_line, source_line) in enumerate(zip(
                        reports, sources, strict=True)):
                    readable = json.loads(report_line)
                    card = json.loads(source_line)
                    aligned = json.loads(next(balance_stream))
                    uid = ids[i]
                    if (readable["account_id"] != uid or
                            card["account_id"] != uid or
                            aligned["account_id"] != uid or
                            (aligned["dataset"], aligned["seed"]) !=
                            (dataset, seed) or
                            readable["prediction"] != card["prediction"] or
                            aligned["prediction"] != card["prediction"] or
                            readable["threshold"] != threshold or
                            abs(readable[
                                "all_evidence_zero_probability"] - aligned[
                                    "all_evidence_zero_probability"]) > 1e-14 or
                            abs(readable["probability"] -
                                full_probability[i]) > 1e-10 or
                            abs(card["probability"] -
                                full_probability[i]) > 1e-10):
                        raise ValueError("account decision differs across layers")
                    terms = {term["name"]: term for term in
                             card["evidence_terms"]}
                    if len(terms) != len(names) or set(terms) != set(names):
                        raise ValueError("source term vocabulary differs")
                    items = aligned["balanced_displayed_items"]
                    claims = readable["fixed_claims"]
                    if len(items) != len(claims) or len(items) not in (5, 6):
                        raise ValueError("displayed claim count differs")
                    predicted_sign = 1 if readable["prediction"] else -1
                    has_opposition = False
                    for item, claim in zip(items, claims, strict=True):
                        j = by_name[item["name"]]
                        term = terms[item["name"]]
                        dp = abs(float(item["probability_without_fact"])
                                 - float(without_probability[i, j]))
                        de = abs(float(item["logit_difference_when_withheld"])
                                 - float(effect[i, j]))
                        max_prob_diff = max(max_prob_diff, dp)
                        max_effect_diff = max(max_effect_diff, de)
                        if (dp > 1e-10 or de > 1e-10 or
                                item["source_fact"] != term["source_fact"] or
                                item["source_level"] != term["source_level"] or
                                item["source_value"] != term["source_value"] or
                                item["source_query"] != term["source_query"] or
                                not term["source_fact_observed"] or
                                claim["operation"] != term[
                                    "source_query"]["operation"] or
                                claim["source_ids"] != term[
                                    "source_query"]["source_ids"]):
                            raise ValueError("readable item lacks source or exact effect")
                        opposing = (float(effect[i, j]) * predicted_sign <
                                    -1e-10)
                        has_opposition |= opposing
                        if item["role"] == "contrary_score_relevant":
                            if (not opposing or
                                    "This input effect opposes the predicted class."
                                    not in claim["claim"]):
                                raise ValueError("supplemental contrary item misstated")
                        run_counts["claims"] += 1
                    if aligned["opposing_count"] > 0 and not has_opposition:
                        raise ValueError("readable report omits known contrary input")
                    counterfactual = readable["source_counterfactual"]
                    if dataset == "mgtab":
                        if (counterfactual is None or
                                counterfactual["certificate_sha256"] !=
                                sha256_file(CONTENT_CERT) or
                                counterfactual[
                                    "stability_certificate_sha256"] !=
                                sha256_file(STABILITY_CERT) or
                                counterfactual[
                                    "local_donor_certificate_sha256"] !=
                                sha256_file(LOCAL_CERT) or
                                counterfactual[
                                    "local_k5_certificate_sha256"] !=
                                sha256_file(K5_CERT)):
                            raise ValueError("MGTAB source response is missing")
                        run_counts["source_counterfactual_accounts"] += 1
                    elif counterfactual is not None:
                        raise ValueError("unexpected source response")
                    run_counts["accounts"] += 1
                    run_counts["six_item_reports"] += len(items) == 6
                    run_counts["reports_with_opposition"] += has_opposition
            if (run_counts["accounts"] != report_run["accounts"] or
                    run_counts["claims"] != report_run["fixed_claims"] or
                    run_counts["six_item_reports"] !=
                    balance_run["six_item_balanced_cards"] or
                    run_counts["source_counterfactual_accounts"] !=
                    report_run["source_counterfactual_accounts"]):
                raise ValueError("run totals differ")
            totals.update(run_counts)
            per_run.append({"dataset": dataset, "seed": seed,
                            **dict(run_counts)})
            print(json.dumps(per_run[-1]), flush=True)
        if next(balance_stream, None) is not None:
            raise ValueError("unmatched balanced report row remains")
    if (totals["accounts"] != 5400 or totals["claims"] != 28007 or
            totals["six_item_reports"] != 1007 or
            totals["reports_with_opposition"] != 4750 or
            totals["source_counterfactual_accounts"] != 3060):
        raise ValueError("complete chain totals differ")
    OUT.write_text(json.dumps({
        "status": "all_readable_items_join_to_source_and_exact_frozen_effect",
        "scope": "nine_fixed_test_runs_all_5400_reports_28007_items",
        "report_certificate_sha256": sha256_file(REPORT_CERT),
        "report_readback_sha256": sha256_file(REPORT_READBACK),
        "source_certificate_sha256": sha256_file(SOURCE_CERT),
        "source_readback_sha256": sha256_file(SOURCE_READBACK),
        "source_selector_sha256": sha256_file(SOURCE_SELECTOR),
        "balance_certificate_sha256": sha256_file(BALANCE_CERT),
        "balance_rows_sha256": sha256_file(BALANCE_ROWS),
        "mgtab_content_response_certificate_sha256":
            sha256_file(CONTENT_CERT),
        "mgtab_content_response_readback_sha256":
            sha256_file(CONTENT_READBACK),
        "mgtab_content_stability_certificate_sha256":
            sha256_file(STABILITY_CERT),
        "mgtab_content_stability_readback_sha256":
            sha256_file(STABILITY_READBACK),
        "mgtab_content_local_certificate_sha256":
            sha256_file(LOCAL_CERT),
        "mgtab_content_local_readback_sha256":
            sha256_file(LOCAL_READBACK),
        "mgtab_content_local_k5_certificate_sha256":
            sha256_file(K5_CERT),
        "mgtab_content_local_k5_readback_sha256":
            sha256_file(K5_READBACK),
        "max_probability_difference": max_prob_diff,
        "max_logit_effect_difference": max_effect_diff,
        "totals": dict(totals), "runs": per_run,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
