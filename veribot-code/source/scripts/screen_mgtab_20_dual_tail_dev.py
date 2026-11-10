"""Dev-only A/B dual-tail screen on the fixed 20 MGTAB source facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--full-model-dir", type=Path, required=True)
    parser.add_argument("--compact-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = SimpleNamespace(
        dataset="mgtab", data_dir=args.data_dir, account_file=None,
        native_graph_facts=None, structural_facts=args.structural_facts,
        neighbor_facts=args.neighbor_facts,
        selected_model_dir=args.full_model_dir)
    runs = []
    for seed in (0, 1, 2):
        full_path = args.full_model_dir / f"seed{seed}" / "report.json"
        compact_path = args.compact_model_dir / f"seed{seed}" / "report.json"
        full = json.loads(full_path.read_text(encoding="utf-8"))
        compact = json.loads(compact_path.read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(config, seed, full)
        if full["manifest"]["split_sha256"] != compact["manifest"]["split_sha256"]:
            raise ValueError("fixed MGTAB population differs")
        reference_path = args.full_model_dir / f"seed{seed}" / "reference.json.gz"
        if sha256_file(reference_path) != full["manifest"]["reference_artifact_sha256"]:
            raise ValueError("frozen label-blind reference differs")
        reference = LabelBlindReference.load(reference_path)
        if reference.version != compact["manifest"]["reference_version"]:
            raise ValueError("reference version differs")
        selected = compact["manifest"]["selected_fact_indices"]
        if ([names[j] for j in selected] != compact["manifest"]["selected_fact_names"]
                or len(selected) != 20):
            raise ValueError("fixed 20 source facts differ")
        train, dev = split["train"], split["val"]
        train_evidence = columns(transform_with_support(
            reference, x[train], split["supports"][train]), selected)
        dev_evidence = columns(transform_with_support(
            reference, x[dev], split["supports"][dev]), selected)
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        model = DualTailEvidenceReasoner(compact["reasoner"]["groups"], l2=1e-5)
        model.fit(rows(train_evidence, a), y[train][a],
                  rows(train_evidence, b), y[train][b])
        probability = model.predict_proba(dev_evidence)
        threshold = best_validation_threshold(probability, y[dev])
        dual = evaluate(y[dev], probability, threshold)
        runs.append({
            "seed": seed,
            "split_sha256": compact["manifest"]["split_sha256"],
            "selected_report_sha256": sha256_file(compact_path),
            "reference_sha256": sha256_file(reference_path),
            "same_20_fact_names": compact["manifest"]["selected_fact_names"],
            "fixed_single_tail_spline_validation": compact["validation"],
            "dual_tail_linear_validation": dual,
            "dual_tail_threshold": threshold,
            "nonzero_high_tail_coefficients": int(np.count_nonzero(model.coef[:20] > 0)),
            "nonzero_low_tail_coefficients": int(np.count_nonzero(model.coef[20:] > 0)),
        })
    result = {
        "dataset": "mgtab", "scope": "official_dev_only_same_20_source_facts",
        "test_predictions_accessed": False,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "structural_facts_sha256": sha256_file(args.structural_facts),
        "neighbor_facts_sha256": sha256_file(args.neighbor_facts),
        "l2": 1e-5,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [
        {"seed": row["seed"], "single_tail": row["fixed_single_tail_spline_validation"],
         "dual_tail": row["dual_tail_linear_validation"]} for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
