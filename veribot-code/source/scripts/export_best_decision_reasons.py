"""Join full source cards with exact frozen-model fact withholding effects."""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
CARD_CERT = ROOT / "research/certificates/best_verified_evidence_full_test.json"
EFFECT_CERTS = {
    "botsim": ROOT / "research/certificates/botsim_observed_means_exact_effects.json",
    "fox8": ROOT / "research/certificates/high_performance_exact_fact_effects.json",
    "mgtab": ROOT / "research/certificates/high_performance_exact_fact_effects.json",
}
OUT = ROOT / "output/v3_best_decision_reasons"
CERT = ROOT / "research/certificates/best_decision_reasons_full_test.json"
MAX_ITEMS = 5
TOL = 1e-10


def source_family(dataset: str, name: str) -> str:
    if dataset == "botsim":
        return name.split(":", 1)[0]
    if dataset == "fox8":
        if name.startswith("native_"):
            return "native_interaction_or_neighbor"
        if name in {"n_tweets", "retweet_ratio", "reply_ratio",
                    "avg_tweet_len", "avg_favorites", "avg_retweets",
                    "avg_score", "unique_lang_ratio"}:
            return "sampled_tweet_activity"
        return "account_profile"
    if name.startswith("published_tweet_projection"):
        return "published_averaged_tweet_projection"
    if name.startswith("published_profile_coordinate"):
        return "published_normalized_profile"
    return "published_typed_relation"


def main() -> None:
    card_certificate = json.loads(CARD_CERT.read_text(encoding="utf-8"))
    if len(card_certificate["runs"]) != 9:
        raise ValueError("full source-card population differs")
    effects = {dataset: json.loads(path.read_text(encoding="utf-8"))
               for dataset, path in EFFECT_CERTS.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in card_certificate["runs"]:
        dataset, seed = run["dataset"], int(run["seed"])
        card_path = Path(run["cards_file"])
        if sha256_file(card_path) != run["cards_sha256"]:
            raise ValueError("full source cards changed")
        effect_plan = next(row for row in effects[dataset]["runs"]
                           if row["dataset"] == dataset and row["seed"] == seed)
        effect_path = ROOT / effect_plan["effect_matrix_path"]
        if sha256_file(effect_path) != effect_plan["effect_matrix_sha256"]:
            raise ValueError("exact effect matrix changed")
        with np.load(effect_path, allow_pickle=False) as matrix:
            ids = matrix["ids"].astype(str).tolist()
            names = matrix["concept_names" if dataset == "botsim"
                           else "fact_names"].astype(str).tolist()
            full = matrix["full_probability"].astype(float)
            without = matrix["without_probability"].astype(float)
            threshold = float(matrix["threshold"])
            logit_difference = (matrix["logit_difference"].astype(float)
                if dataset == "botsim" else
                matrix["full_logit"][:, None] - matrix["without_logit"])
        if (len(ids) != run["accounts"] or
                without.shape != (len(ids), len(names)) or
                logit_difference.shape != without.shape):
            raise ValueError("exact effect population differs from source cards")
        output = OUT / f"{dataset}_seed{seed}.jsonl.gz"
        count = 0
        categories = {"single_fact_class_critical": 0,
                      "distributed_class_dependence": 0,
                      "no_class_change_from_all_evidence": 0}
        cited_items = 0
        with gzip.open(card_path, "rt", encoding="utf-8") as source_stream, \
                output.open("wb") as sink, gzip.GzipFile(
                    filename="", mode="wb", fileobj=sink, mtime=0) as target:
            for line in source_stream:
                if count >= len(ids):
                    raise ValueError("source cards exceed effect matrix")
                card = json.loads(line)
                if (card["account_id"] != ids[count] or
                        abs(card["probability"] - full[count]) > TOL or
                        abs(card["threshold"] - threshold) > TOL):
                    raise ValueError("source card and exact intervention differ")
                by_name = {term["name"]: term
                           for term in card["evidence_terms"]}
                if set(by_name) != set(names):
                    raise ValueError("source facts differ from intervened inputs")
                entries = []
                for j, name in enumerate(names):
                    term = by_name[name]
                    delta = float(logit_difference[count, j])
                    without_probability = float(without[count, j])
                    changed = bool((without_probability >= threshold) !=
                                   bool(card["prediction"]))
                    role = ("class_critical" if changed else
                            "score_relevant" if abs(delta) > TOL else
                            "context_only")
                    if role == "context_only":
                        continue
                    if (term["source_query"] is None or
                            not term["source_fact_observed"]):
                        raise ValueError("model-relevant item lacks observed source")
                    entries.append({
                        "name": name, "source_fact": term["source_fact"],
                        "source_value": term["source_value"],
                        "source_level": term["source_level"],
                        "source_family": source_family(dataset, name),
                        "source_query": term["source_query"],
                        "role": role,
                        "probability_without_fact": without_probability,
                        "logit_difference_when_withheld": delta,
                        "class_changes_when_withheld": changed,
                    })
                critical = [row for row in entries
                            if row["role"] == "class_critical"]
                critical.sort(key=lambda row: (-abs(row[
                    "logit_difference_when_withheld"]), row["name"]))
                score_only = [row for row in entries
                              if row["role"] == "score_relevant"]
                score_only.sort(key=lambda row: (-abs(row[
                    "logit_difference_when_withheld"]), row["name"]))
                chosen = (critical + score_only)[:MAX_ITEMS]
                all_changes = bool(card["exact_frozen_model_dependency"][
                    "all_evidence_changes_class"])
                category = ("single_fact_class_critical" if critical else
                            "distributed_class_dependence" if all_changes else
                            "no_class_change_from_all_evidence")
                categories[category] += 1
                result = {
                    "dataset": dataset, "seed": seed,
                    "account_id": ids[count],
                    "prediction": card["prediction"],
                    "probability": card["probability"],
                    "threshold": threshold,
                    "source_card_sha256": run["cards_sha256"],
                    "effect_matrix_sha256": effect_plan[
                        "effect_matrix_sha256"],
                    "reason_category": category,
                    "all_evidence_changes_class": all_changes,
                    "class_critical_fact_count": len(critical),
                    "score_relevant_fact_count": len(score_only),
                    "displayed_items": chosen,
                    "display_policy": "class-critical exact input withholding first; otherwise largest exact score effects; no behavioral causal claim",
                }
                target.write((json.dumps(result, ensure_ascii=False,
                                         separators=(",", ":"), allow_nan=False)
                              + "\n").encode("utf-8"))
                cited_items += len(chosen)
                count += 1
        if count != run["accounts"] or sum(categories.values()) != count:
            raise ValueError("concise reason population differs")
        runs.append({
            "dataset": dataset, "seed": seed,
            "accounts": count, "displayed_source_items": cited_items,
            "reason_categories": categories,
            "reason_cards_file": str(output),
            "reason_cards_sha256": sha256_file(output),
            "source_cards_sha256": run["cards_sha256"],
            "effect_matrix_sha256": effect_plan["effect_matrix_sha256"],
        })
        print({"dataset": dataset, "seed": seed,
               "accounts": count, **categories}, flush=True)
    CERT.write_text(json.dumps({
        "status": "all_best_model_decisions_have_source_bound_exact_effect_reason_cards",
        "scope": "nine_fixed_test_runs_5400_decisions",
        "source_card_certificate_sha256": sha256_file(CARD_CERT),
        "effect_certificate_sha256": {dataset: sha256_file(path)
                                      for dataset, path in EFFECT_CERTS.items()},
        "maximum_displayed_items_per_decision": MAX_ITEMS,
        "total_accounts": sum(row["accounts"] for row in runs),
        "total_displayed_items": sum(row["displayed_source_items"] for row in runs),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
