"""Independently verify the concise reasons against source cards and exact masks."""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np

from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CERT = ROOT / "research/certificates/best_decision_reasons_full_test.json"
SOURCE_CERT = ROOT / "research/certificates/best_verified_evidence_full_test.json"
OUT = ROOT / "research/certificates/best_decision_reasons_readback.json"
EFFECT_CERTS = {
    "botsim": ROOT / "research/certificates/botsim_observed_means_exact_effects.json",
    "fox8": ROOT / "research/certificates/high_performance_exact_fact_effects.json",
    "mgtab": ROOT / "research/certificates/high_performance_exact_fact_effects.json",
}


def main() -> None:
    certificate = json.loads(CERT.read_text(encoding="utf-8"))
    source_cert = json.loads(SOURCE_CERT.read_text(encoding="utf-8"))
    if (len(certificate["runs"]) != 9 or
            sha256_file(SOURCE_CERT) != certificate[
                "source_card_certificate_sha256"]):
        raise ValueError("source-card or summary certificate differs")
    source_runs = {(row["dataset"], row["seed"]): row
                   for row in source_cert["runs"]}
    effect_certificates = {key: json.loads(path.read_text(encoding="utf-8"))
                           for key, path in EFFECT_CERTS.items()}
    runs = []
    family_counts = Counter()
    level_counts = Counter()
    role_level_counts = Counter()
    for run in certificate["runs"]:
        dataset, seed = run["dataset"], int(run["seed"])
        source = source_runs[dataset, seed]
        card_path = Path(source["cards_file"])
        reason_path = Path(run["reason_cards_file"])
        if (sha256_file(card_path) != run["source_cards_sha256"] or
                sha256_file(reason_path) != run["reason_cards_sha256"]):
            raise ValueError("source or reason card changed")
        effect_plan = next(row for row in effect_certificates[dataset]["runs"]
                           if row["dataset"] == dataset and row["seed"] == seed)
        effect_path = ROOT / effect_plan["effect_matrix_path"]
        if sha256_file(effect_path) != run["effect_matrix_sha256"]:
            raise ValueError("exact effect matrix changed")
        with np.load(effect_path, allow_pickle=False) as matrix:
            ids = matrix["ids"].astype(str).tolist()
            names = matrix["concept_names" if dataset == "botsim"
                           else "fact_names"].astype(str).tolist()
            probability = matrix["full_probability"].astype(float)
            without = matrix["without_probability"].astype(float)
            delta = (matrix["logit_difference"].astype(float)
                     if dataset == "botsim" else
                     matrix["full_logit"][:, None] - matrix["without_logit"])
            threshold = float(matrix["threshold"])
        index = {name: j for j, name in enumerate(names)}
        count = 0
        shown = 0
        categories = Counter()
        with gzip.open(card_path, "rt", encoding="utf-8") as originals, \
                gzip.open(reason_path, "rt", encoding="utf-8") as reports:
            for original_line, report_line in zip(originals, reports, strict=True):
                original = json.loads(original_line)
                report = json.loads(report_line)
                if (report["account_id"] != ids[count] or
                        original["account_id"] != ids[count] or
                        report["prediction"] != original["prediction"] or
                        abs(report["probability"] - probability[count]) > 1e-12 or
                        abs(report["threshold"] - threshold) > 1e-12):
                    raise ValueError("reason card identity or decision differs")
                original_terms = {term["name"]: term
                                  for term in original["evidence_terms"]}
                critical_count = int(np.count_nonzero(
                    (without[count] >= threshold) != bool(report["prediction"])))
                relevant_count = int(np.count_nonzero(
                    (np.abs(delta[count]) > 1e-10) &
                    ((without[count] >= threshold) ==
                     bool(report["prediction"]))))
                if (report["class_critical_fact_count"] != critical_count or
                        report["score_relevant_fact_count"] != relevant_count or
                        len(report["displayed_items"]) > 5):
                    raise ValueError("reason card effect counts differ")
                seen = set()
                for item in report["displayed_items"]:
                    name = item["name"]
                    if name in seen or name not in index:
                        raise ValueError("reason fact repeats or is absent")
                    seen.add(name)
                    j = index[name]
                    term = original_terms[name]
                    actual_change = bool((without[count, j] >= threshold) !=
                                         bool(report["prediction"]))
                    if (not term["source_fact_observed"] or
                            item["source_query"] != term["source_query"] or
                            item["source_value"] != term["source_value"] or
                            item["source_level"] != term["source_level"] or
                            abs(item["probability_without_fact"] -
                                without[count, j]) > 1e-12 or
                            abs(item["logit_difference_when_withheld"] -
                                delta[count, j]) > 1e-10 or
                            item["class_changes_when_withheld"] != actual_change or
                            item["role"] != ("class_critical" if actual_change
                                else "score_relevant")):
                        raise ValueError("displayed source fact or exact effect differs")
                    family_counts[dataset, item["source_family"]] += 1
                    level_counts[dataset, item["source_level"]] += 1
                    role_level_counts[dataset, item["role"], item[
                        "source_level"]] += 1
                categories[report["reason_category"]] += 1
                shown += len(report["displayed_items"])
                count += 1
        if (count != run["accounts"] or shown != run["displayed_source_items"]
                or any(categories[key] != value for key, value
                       in run["reason_categories"].items())):
            raise ValueError("full reason-card counts differ")
        runs.append({"dataset": dataset, "seed": seed,
                     "accounts": count, "displayed_source_items": shown,
                     "reason_categories": dict(categories)})
        print({"dataset": dataset, "seed": seed, "accounts": count,
               "displayed": shown}, flush=True)
    OUT.write_text(json.dumps({
        "status": "all_concise_reason_cards_independently_reconstructed",
        "reason_certificate_sha256": sha256_file(CERT),
        "total_accounts": sum(row["accounts"] for row in runs),
        "total_displayed_source_items": sum(
            row["displayed_source_items"] for row in runs),
        "displayed_source_family_counts": {
            f"{dataset}:{family}": count for (dataset, family), count
            in sorted(family_counts.items())},
        "displayed_source_level_counts": {
            f"{dataset}:{level}": count for (dataset, level), count
            in sorted(level_counts.items())},
        "displayed_role_and_source_level_counts": {
            f"{dataset}:{role}:{level}": count
            for (dataset, role, level), count
            in sorted(role_level_counts.items())},
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
