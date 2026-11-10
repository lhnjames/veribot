"""Dev-only shortcut diagnostic using Fox8 fact availability, not fact values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    manifest = json.loads(args.native_facts.with_suffix(".json").read_text(
        encoding="utf-8"))
    if (audit["status"] != "verified_all_native_interaction_fact_cells_and_event_locators"
            or audit["fact_cache_sha256"] != sha256_file(args.native_facts)
            or manifest["features_csv_sha256"] !=
            sha256_file(args.data_dir / "features.csv")):
        raise ValueError("source audit or fact cache differs")
    with np.load(args.native_facts, allow_pickle=False) as cache:
        cached_ids = cache["ids"].astype(str).tolist()
        native = cache["values"].astype(float)
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load("fox8", args.data_dir, None, seed)
        if ids != cached_ids:
            raise ValueError("native and base fact account orders differ")
        missing = (~np.isfinite(np.column_stack((x, native)))).astype(float)
        variable = np.any(missing[split["train"]] != missing[split["train"]][0], axis=0)
        if np.any(variable):
            classifier = LogisticRegression(C=1.0, solver="liblinear", random_state=0)
            classifier.fit(missing[split["train"]][:, variable], y[split["train"]])
            probability = classifier.predict_proba(missing[split["val"]][:, variable])[:, 1]
        else:
            probability = np.full(len(split["val"]), float(y[split["train"]].mean()))
        threshold = best_validation_threshold(probability, y[split["val"]])
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "variable_missingness_features": int(np.sum(variable)),
                     "validation": evaluate(y[split["val"]], probability, threshold)})
    result = {"dataset": "fox8", "scope": "validation_only_missingness_shortcut_diagnostic",
              "test_predictions_accessed": False,
              "base_feature_sha256": sha256_file(args.data_dir / "features.csv"),
              "native_facts_sha256": sha256_file(args.native_facts),
              "native_source_audit_sha256": sha256_file(args.source_audit),
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
