"""Build train/dev facts with no dev declarations at fit and no test declarations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256


GRAPH_DEGREES = ("graph_out_degree", "graph_in_degree", "graph_reciprocal_degree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-facts", type=Path, required=True)
    parser.add_argument("--visibility-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    visibility = json.loads(args.visibility_audit.read_text(encoding="utf-8"))
    if sha256(args.full_facts) != visibility["corrected_fact_cache_sha256"]:
        raise ValueError("Full sampled-graph facts differ from visibility audit")
    with np.load(args.full_facts, allow_pickle=False) as saved:
        split_all = saved["split"].astype(str)
        keep = np.flatnonzero(np.isin(split_all, ("train", "dev")))
        arrays = {key: (saved[key][keep].copy() if key != "names"
                        else saved[key].copy()) for key in saved.files}
    if (np.count_nonzero(arrays["split"] == "train")
            != visibility["train_accounts"]
            or np.any(arrays["split"] == "test")):
        raise ValueError("Split-visible cache includes a test account")
    ids = arrays["ids"].astype(str)
    split = arrays["split"].astype(str)
    positions = arrays["positions"].astype(int)
    names = arrays["names"].astype(str).tolist()
    values = arrays["values"].astype(float)
    index = {uid: i for i, uid in enumerate(ids)}
    if len(index) != len(ids):
        raise ValueError("Duplicate train/dev account")
    for part, rows, ordinal_key, new_key in (
            ("train", visibility["changed_train_account_rows"],
             "train_ordinal", "train_plus_support_declaration_degrees"),
            ("dev", visibility["changed_dev_account_rows"],
             "dev_ordinal", "train_support_dev_declaration_degrees")):
        for row in rows:
            i = index[row["account_id"]]
            if split[i] != part or positions[i] != row[ordinal_key]:
                raise ValueError("Source graph account locator differs")
            for name in GRAPH_DEGREES:
                j = names.index(name)
                if values[i, j] != row["full_graph_degrees"][name]:
                    raise ValueError("Full graph fact differs from audit")
                values[i, j] = row[new_key][name]
    arrays["values"] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    manifest = {
        "dataset": "twibot20",
        "source_full_fact_cache_sha256": sha256(args.full_facts),
        "visibility_audit_sha256": sha256(args.visibility_audit),
        "fact_cache_sha256": sha256(args.output),
        "split_counts": {part: int(np.count_nonzero(split == part))
                         for part in ("train", "dev")},
        "test_accounts_or_features_in_cache": False,
        "training_graph_declarations": "official train and unlabeled support rows only",
        "development_graph_declarations": "official train, unlabeled support, and dev rows only",
        "train_rows_with_graph_degree_change": len(
            visibility["changed_train_account_rows"]),
        "dev_rows_with_graph_degree_change": len(
            visibility["changed_dev_account_rows"]),
        "non_graph_values_and_labels_unchanged": True,
        "protocol_scope": "development-only split-visible source graph; dev labels used solely for evaluation and threshold selection",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
