"""Measure frozen prediction and decision-evidence agreement for repeat test IDs."""

from __future__ import annotations

import gzip
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

from gate0_shallow_tree import sha256_file


CONFIG = {
    "botsim": ("output/v3_computable/botsim_group_tail_12",
               "concept_evidence_cards.jsonl.gz", "concepts"),
    "fox8": ("output/v3_fox8_native_spline_l2/group_tail_20_selected",
             "concept_evidence_cards.jsonl.gz", "concepts"),
    "mgtab": ("output/v3_mgtab_ab_spline_content/direct_selected_20",
              "evidence_cards.jsonl.gz", "terms"),
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    dependence_path = root / "research/three_dataset_compact_decision_dependence_2026-11-03.json"
    dependence = read(dependence_path)
    output_rows = []
    for dataset, (relative, card_filename, term_key) in CONFIG.items():
        folder = root / relative
        dependency = next(row for row in dependence["datasets"]
                          if row["dataset"] == dataset)
        by_seed = {run["seed"]: run for run in dependency["runs"]}
        by_account = defaultdict(list)
        hashes = {}
        for seed in (0, 1, 2):
            seed_dir = folder / f"seed{seed}"
            report = read(seed_dir / "report.json")
            predictions = read(seed_dir / "test_predictions.json")["predictions"]
            cards = seed_dir / card_filename
            digest = sha256_file(cards)
            if (digest != report["cards_sha256"] or
                    digest != by_seed[seed]["cards_sha256"] or
                    len(predictions) != len(by_seed[seed]["accounts"])):
                raise ValueError("frozen source cards or decision audit differ")
            hashes[str(seed)] = digest
            seen = 0
            with gzip.open(cards, "rt", encoding="utf-8") as stream:
                for index, line in enumerate(stream):
                    card = json.loads(line)
                    saved = predictions[index]
                    prior = by_seed[seed]["accounts"][index]
                    if (card["account_id"] != saved["account_id"] or
                            card["account_id"] != prior["account_id"] or
                            card["prediction"] != prior["prediction"] or
                            abs(card["probability"] - saved["probability"]) > 1e-11):
                        raise ValueError("account or prediction differs")
                    prediction = card["prediction"]
                    terms = card[term_key]
                    support = [term for term in terms if (
                        term["logit_contribution"] > 0 if prediction else
                        term["logit_contribution"] < 0)]
                    top = (max(support, key=lambda term: abs(term["logit_contribution"]))
                           if support else None)
                    top_name = top["name"] if top else None
                    top_source = ((top["winning_atomic_fact"] if dataset != "mgtab"
                                   else top["name"]) if top else None)
                    by_account[card["account_id"]].append({
                        "seed": seed, "label": saved["label"],
                        "prediction": prediction,
                        "top_decision_support_name": top_name,
                        "top_decision_support_source": top_source,
                        "pivotal_names": sorted(prior["individual_pivotal_concepts"]),
                    })
                    seen += 1
            if seen != len(predictions):
                raise ValueError("frozen card population differs")
        appearances = Counter(len(rows) for rows in by_account.values())
        pairs = []
        for account_id, rows in by_account.items():
            if len({row["label"] for row in rows}) != 1:
                raise ValueError("same account has different labels across seeds")
            for left, right in itertools.combinations(rows, 2):
                lset, rset = set(left["pivotal_names"]), set(right["pivotal_names"])
                # PCA components are fit separately on each training split. Equal
                # component indices do not identify equal semantic features.
                comparable_top = (dataset != "mgtab" or all(
                    not str(name).startswith("published_tweet_projection_")
                    for name in (left["top_decision_support_name"],
                                 right["top_decision_support_name"])))
                comparable_pivotal = (dataset != "mgtab" or all(
                    not str(name).startswith("published_tweet_projection_")
                    for name in lset | rset))
                pairs.append({
                    "account_id": account_id,
                    "seed_pair": [left["seed"], right["seed"]],
                    "predictions": [left["prediction"], right["prediction"]],
                    "top_support_concepts": [left["top_decision_support_name"],
                                             right["top_decision_support_name"]],
                    "top_support_source_facts": [left["top_decision_support_source"],
                                                 right["top_decision_support_source"]],
                    "pivotal_name_sets": [left["pivotal_names"], right["pivotal_names"]],
                    "same_prediction": left["prediction"] == right["prediction"],
                    "same_top_support_concept": (
                        left["top_decision_support_name"] ==
                        right["top_decision_support_name"]),
                    "same_top_support_source_fact": (
                        left["top_decision_support_source"] ==
                        right["top_decision_support_source"]),
                    "comparable_top_support": comparable_top,
                    "both_have_pivotal_term": bool(lset and rset),
                    "comparable_pivotal_sets": comparable_pivotal,
                    "same_nonempty_pivotal_set": bool(lset and rset and lset == rset),
                })
        if not pairs:
            raise ValueError("no repeated test accounts")
        both = [row for row in pairs if row["both_have_pivotal_term"]]
        comparable_top = [row for row in pairs if row["comparable_top_support"]]
        comparable_pivotal = [row for row in both
                              if row["comparable_pivotal_sets"]]
        output_rows.append({
            "dataset": dataset,
            "test_appearance_counts": dict(appearances),
            "unique_test_accounts": len(by_account),
            "repeated_test_accounts": sum(len(rows) >= 2 for rows in by_account.values()),
            "paired_repeat_occurrences": len(pairs),
            "same_prediction_count": sum(row["same_prediction"] for row in pairs),
            "same_top_support_concept_count": sum(
                row["same_top_support_concept"] for row in pairs),
            "same_top_support_source_fact_count": sum(
                row["same_top_support_source_fact"] for row in pairs),
            "comparable_top_support_pair_count": len(comparable_top),
            "same_comparable_top_support_count": sum(
                row["same_top_support_concept"] for row in comparable_top),
            "both_have_pivotal_term_count": len(both),
            "same_nonempty_pivotal_set_count": sum(
                row["same_nonempty_pivotal_set"] for row in both),
            "comparable_nonempty_pivotal_pair_count": len(comparable_pivotal),
            "same_comparable_nonempty_pivotal_set_count": sum(
                row["same_nonempty_pivotal_set"] for row in comparable_pivotal),
            "card_sha256_by_seed": hashes,
            "pairs": pairs,
        })
    output = {
        "scope": "frozen_test_overlap_descriptive_no_model_selection",
        "decision_dependence_audit_sha256": sha256_file(dependence_path),
        "interpretation": "Different frozen train splits change model parameters and reference distributions; agreement is descriptive, not external semantic truth.",
        "datasets": output_rows,
    }
    path = root / "research/three_dataset_cross_seed_evidence_stability_2026-11-03.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps([{k: row[k] for k in (
        "dataset", "repeated_test_accounts", "paired_repeat_occurrences",
        "same_prediction_count", "same_top_support_concept_count",
        "same_top_support_source_fact_count", "both_have_pivotal_term_count",
        "same_nonempty_pivotal_set_count", "comparable_top_support_pair_count",
        "same_comparable_top_support_count",
        "comparable_nonempty_pivotal_pair_count",
        "same_comparable_nonempty_pivotal_set_count")}
        for row in output_rows], indent=2))


if __name__ == "__main__":
    main()
