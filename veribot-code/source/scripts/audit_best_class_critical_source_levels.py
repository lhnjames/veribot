"""Locate individually pivotal source tiers in every current fixed decision."""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file


ROOT = Path(__file__).resolve().parents[2]
REASONS = ROOT / "research/certificates/best_decision_reasons_full_test.json"
OUT = ROOT / "research/certificates/best_class_critical_source_levels.json"


def main() -> None:
    manifest = json.loads(REASONS.read_text(encoding="utf-8"))
    if manifest["total_accounts"] != 5400 or len(manifest["runs"]) != 9:
        raise ValueError("current decision population differs")
    runs = []
    for run in manifest["runs"]:
        dataset, seed = run["dataset"], run["seed"]
        source = ROOT / f"output/v3_best_verified_evidence_full_test/{dataset}_seed{seed}.jsonl.gz"
        effects = ROOT / "output/v3_best_exact_fact_effects" / (
            f"botsim_observed_seed{seed}.npz" if dataset == "botsim" else
            f"{dataset}_seed{seed}.npz")
        reasons = Path(run["reason_cards_file"])
        if (sha256_file(source) != run["source_cards_sha256"] or
                sha256_file(effects) != run["effect_matrix_sha256"] or
                sha256_file(reasons) != run["reason_cards_sha256"]):
            raise ValueError("source, effects or reason artifact changed")
        with np.load(effects, allow_pickle=False) as matrix:
            ids = matrix["ids"].astype(str)
            names = matrix["concept_names" if dataset == "botsim" else
                           "fact_names"].astype(str)
            full = matrix["full_probability"].astype(float)
            without = matrix["without_probability"].astype(float)
            threshold = float(matrix["threshold"])
        pivotal = (without >= threshold) != (full[:, None] >= threshold)
        if pivotal.shape != (run["accounts"], len(names)):
            raise ValueError("fixed exact-effect matrix shape differs")
        levels = Counter()
        facts = Counter()
        categories = Counter()
        noncontent_pivotal_accounts = 0
        content_pivotal_accounts = 0
        content_only_pivotal_accounts = 0
        class_critical_accounts = 0
        source_and_reason_accounts = 0
        with gzip.open(source, "rt", encoding="utf-8") as source_stream, \
                gzip.open(reasons, "rt", encoding="utf-8") as reason_stream:
            for row_index, (source_line, reason_line) in enumerate(zip(
                    source_stream, reason_stream, strict=True)):
                card = json.loads(source_line)
                reason = json.loads(reason_line)
                if (card["account_id"] != ids[row_index] or
                        reason["account_id"] != ids[row_index] or
                        abs(card["probability"] - float(full[row_index])) > 3e-16 or
                        abs(reason["probability"] - float(full[row_index])) > 3e-16 or
                        card["threshold"] != threshold or
                        reason["threshold"] != threshold):
                    raise ValueError("account decision differs")
                terms = card["evidence_terms"]
                by_name = {term["name"]: term for term in terms}
                if (len(by_name) != len(names) or
                        set(by_name) != set(names.tolist())):
                    raise ValueError("source evidence term vocabulary differs")
                active = np.flatnonzero(pivotal[row_index])
                if len(active) != reason["class_critical_fact_count"]:
                    raise ValueError("reason's pivotal count differs")
                category = reason["reason_category"]
                if (bool(len(active)) !=
                        (category == "single_fact_class_critical")):
                    raise ValueError("pivotal source and reason category differ")
                categories[category] += 1
                class_critical_accounts += bool(len(active))
                noncontent = False
                content = False
                for index in active:
                    term = by_name[names[int(index)]]
                    level = term["source_level"]
                    levels[level] += 1
                    facts[term["source_fact"]] += 1
                    if level == "published_averaged_tweet_tensor_train_only_projection":
                        content = True
                    else:
                        noncontent = True
                noncontent_pivotal_accounts += noncontent
                content_pivotal_accounts += content
                content_only_pivotal_accounts += content and not noncontent
                source_and_reason_accounts += 1
        if source_and_reason_accounts != run["accounts"]:
            raise ValueError("source/reason population differs")
        runs.append({"dataset": dataset, "seed": seed,
            "accounts": source_and_reason_accounts,
            "class_critical_accounts": class_critical_accounts,
            "reason_categories": dict(categories),
            "pivotal_fact_occurrences_by_source_level": dict(levels),
            "top_pivotal_source_facts": facts.most_common(15),
            "accounts_with_pivotal_noncontent_source": noncontent_pivotal_accounts,
            "accounts_with_pivotal_averaged_tweet_projection": content_pivotal_accounts,
            "accounts_with_only_pivotal_averaged_tweet_projection":
                content_only_pivotal_accounts,
            "source_cards_sha256": sha256_file(source),
            "exact_effects_sha256": sha256_file(effects),
            "reason_cards_sha256": sha256_file(reasons)})
        print({"dataset": dataset, "seed": seed,
               "pivotal_accounts": class_critical_accounts,
               "noncontent": noncontent_pivotal_accounts,
               "content_only": content_only_pivotal_accounts}, flush=True)
    OUT.write_text(json.dumps({
        "status": "all_current_fixed_decisions_mapped_to_pivotal_source_levels",
        "scope": "nine_fixed_test_runs_5400_accounts",
        "reason_certificate_sha256": sha256_file(REASONS),
        "counts_measure_exact_single_input_mask_not_causality_or_semantic_truth": True,
        "runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
