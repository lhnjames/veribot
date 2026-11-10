"""Dev-only shortcut control: classify from source-fact missingness alone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import attach_mgtab_neighbor, attach_mgtab_structural


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mgtab", "botsim"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--structural-facts", type=Path)
    parser.add_argument("--neighbor-facts", type=Path)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if args.dataset == "mgtab":
        if (audit.get("status") != "verified_all_published_tensor_derived_fact_cells"
                or args.structural_facts is None or args.neighbor_facts is None
                or audit["structural_facts_sha256"] != sha256_file(args.structural_facts)
                or audit["neighbor_facts_sha256"] != sha256_file(args.neighbor_facts)
                or any(audit["source_sha256"][name] != sha256_file(args.data_dir / name)
                       for name in ("features.pt", "edge_index.pt", "edge_type.pt"))):
            raise ValueError("MGTAB requires its full source audit and 86 fact sources")
    else:
        if (audit.get("status") != "verified_full_population_against_raw_comments"
                or args.native_graph_facts is None or args.account_file is None
                or audit["cache_sha256"] != sha256_file(args.native_graph_facts)
                or audit["input_sha256"]["Users.csv"] != sha256_file(args.account_file)):
            raise ValueError("BotSim requires its full native reply source audit")
    runs = []
    for seed in (0, 1, 2):
        x, y, ids, names, split, source = load(
            args.dataset, args.data_dir, args.account_file, seed,
            args.native_graph_facts)
        if args.dataset == "mgtab":
            x, names, source = attach_mgtab_structural(
                x, names, ids, source, args.data_dir, args.structural_facts)
            x, names, supports, source = attach_mgtab_neighbor(
                x, names, ids, source, args.data_dir, args.neighbor_facts)
            if len(names) != 86 or supports is None:
                raise ValueError("MGTAB selected fact vocabulary differs")
        else:
            if len(names) != 22:
                raise ValueError("BotSim selected fact vocabulary differs")
        missing = (~np.isfinite(x)).astype(float)
        train, dev = split["train"], split["val"]
        variable = np.any(missing[train] != missing[train][0], axis=0)
        if np.any(variable):
            classifier = LogisticRegression(C=1.0, solver="liblinear", random_state=0)
            classifier.fit(missing[train][:, variable], y[train])
            probability = classifier.predict_proba(missing[dev][:, variable])[:, 1]
        else:
            probability = np.full(len(dev), float(y[train].mean()))
        threshold = best_validation_threshold(probability, y[dev])
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "variable_missingness_features": int(np.sum(variable)),
                     "variable_fact_names": [name for name, flag in zip(names, variable) if flag],
                     "validation": evaluate(y[dev], probability, threshold)})
    result = {"dataset": args.dataset,
              "scope": "validation_only_fact_missingness_shortcut_diagnostic",
              "test_predictions_accessed": False,
              "source_audit_sha256": sha256_file(args.source_audit),
              "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
