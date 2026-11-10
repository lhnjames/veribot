"""Frozen Fox8 source-group removal and single-group diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import expit

from gate0_shallow_tree import sha256_file
from lock_fox8_single_fact_baseline import ROOT, population


LEDGER = ROOT / "research/certificates/three_dataset_signed_group_ledger.json"
FIXED = ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json"
OUT = ROOT / "research/certificates/fox8_neutral_source_group_fixed_test_diagnostic.json"


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    fixed = json.loads(FIXED.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        source = next(x for x in ledger["runs"]
                      if x["dataset"] == "fox8" and x["seed"] == seed)
        path = Path(source["per_account_group_matrix_path"])
        if sha256_file(path) != source["per_account_group_matrix_sha256"]:
            raise ValueError("Signed source ledger changed")
        with np.load(path, allow_pickle=False) as z:
            ids = z["account_ids"].astype(str)
            names = z["source_group_names"].astype(str).tolist()
            logit = z["logit"].astype(float)
            zero = z["zero_evidence_logit"].astype(float)
            effects = z["group_logit_effect"].astype(float)
            pred = z["predicted_bot"].astype(bool)
            threshold = float(z["threshold"])
        runtime, labels, split, report = population(seed)
        test = split["test"]
        if (not np.array_equal(ids, np.asarray(runtime.ids)[test].astype(str)) or
                report["manifest"]["split_sha256"] !=
                fixed["runs"][seed]["split_sha256"]):
            raise ValueError("Fox8 fixed population or split differs")
        truth = labels[test].astype(bool)
        assert int(np.count_nonzero(pred == truth)) == fixed["runs"][seed]["correct"]
        assert float(np.max(np.abs(zero + effects.sum(axis=1) - logit))) < 1e-9
        groups = []
        for j, name in enumerate(names):
            without = expit(logit - effects[:, j]) >= threshold
            only = expit(zero + effects[:, j]) >= threshold
            groups.append({
                "group": name,
                "removed_correct": int(np.count_nonzero(without == truth)),
                "removed_decision_changes": int(np.count_nonzero(without != pred)),
                "only_group_correct": int(np.count_nonzero(only == truth)),
                "only_group_decision_matches_full": int(np.count_nonzero(
                    only == pred)),
                "supporting_accounts": source["supporting_accounts_by_group"][name],
                "counterevidence_accounts": source[
                    "counterevidence_accounts_by_group"][name]})
        runs.append({"seed": seed, "accounts": len(ids),
                     "full_correct": int(np.count_nonzero(pred == truth)),
                     "groups": groups,
                     "matrix_sha256": source["per_account_group_matrix_sha256"]})
        print({"seed": seed, "best_only_group": max(
            groups, key=lambda row: row["only_group_correct"])["group"],
            "best_only_correct": max(
                row["only_group_correct"] for row in groups)}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "Fox8_frozen_neutral_source_group_fixed_test_diagnostic_only",
        "test_population_previously_explored_by_other_candidates": True,
        "no_model_or_group_selected_from_test": True,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
