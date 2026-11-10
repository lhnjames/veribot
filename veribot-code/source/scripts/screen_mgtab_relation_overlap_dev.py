"""Dev-only A/B capacity check of four audited MGTAB relation-overlap facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from train_v3_computable import rows
from verifiable_bot.mgtab_relation_overlap_facts import NAMES
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def _evidence(reference: LabelBlindReference, matrix: np.ndarray,
              profile_supports: np.ndarray, overlap_supports: np.ndarray | None
              ) -> EvidenceBatch:
    batch = reference.transform(matrix)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= profile_supports / (profile_supports + 5.0)
    if overlap_supports is not None:
        reliability[:, 86:90] *= overlap_supports / (overlap_supports + 5.0)
    return EvidenceBatch(batch.upper, batch.lower, reliability, batch.available,
                         batch.reference_version, batch.names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--overlap-facts", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--selected-model-dir", type=Path, required=True)
    parser.add_argument("--named-86-control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    audit = json.loads(args.overlap_audit.read_text(encoding="utf-8"))
    control = json.loads(args.named_86_control.read_text(encoding="utf-8"))
    if (audit["status"] != "all_relation_overlap_fact_cells_replayed"
            or audit["facts_sha256"] != sha256_file(args.overlap_facts)
            or control["fact_set"] != "published_profile_and_relation_all"):
        raise ValueError("overlap source or 86-fact control differs")
    with np.load(args.overlap_facts, allow_pickle=False) as cache:
        ids = cache["ids"].astype(int)
        names_added = tuple(cache["names"].astype(str))
        overlap = cache["values"].astype(float)
        overlap_supports = cache["supports"].astype(float)
    if (not np.array_equal(ids, np.arange(10199)) or names_added != NAMES
            or overlap.shape != (10199, 4) or overlap_supports.shape != (10199, 4)):
        raise ValueError("audited overlap matrix differs")
    control_runs = {row["seed"]: row for row in control["runs"]}
    runs = []
    for seed in (0, 1, 2):
        report_path = args.selected_model_dir / f"seed{seed}" / "report.json"
        original = json.loads(report_path.read_text(encoding="utf-8"))
        x, labels, names, split = selected_facts(args, seed, original)
        if (x.shape != (10199, 150) or len(names) != 150
                or control_runs[seed]["split_sha256"] != original["manifest"]["split_sha256"]):
            raise ValueError("frozen MGTAB population differs")
        train, dev = split["train"], split["val"]
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=labels[train], random_state=seed)
        alternatives = {}
        for label, matrix, vocabulary, support in (
            ("published_86", x[:, :86], names[:86], None),
            ("published_86_plus_overlap_4", np.column_stack((x[:, :86], overlap)),
             names[:86] + list(NAMES), overlap_supports),
        ):
            reference = LabelBlindReference(min_context_n=30).fit(
                matrix[train], vocabulary)
            train_evidence = _evidence(reference, matrix[train],
                                       split["supports"][train],
                                       support[train] if support is not None else None)
            dev_evidence = _evidence(reference, matrix[dev],
                                     split["supports"][dev],
                                     support[dev] if support is not None else None)
            groups = {name: [name] for name in vocabulary}
            model = MonotoneSplineReasoner(groups, l2=1e-5)
            model.fit(rows(train_evidence, a), labels[train][a],
                      rows(train_evidence, b), labels[train][b])
            probability = model.predict_proba(dev_evidence)
            threshold = best_validation_threshold(probability, labels[dev])
            alternatives[label] = {
                "validation": evaluate(labels[dev], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
            }
        expected = control_runs[seed]["alternatives"][str(1e-5)]["validation"]
        actual = alternatives["published_86"]["validation"]
        if any(abs(actual[key] - expected[key]) > 1e-9
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("published 86-fact control did not replay")
        runs.append({"seed": seed,
                     "split_sha256": original["manifest"]["split_sha256"],
                     "alternatives": alternatives})
    result = {
        "dataset": "mgtab", "scope": "official_train_dev_only_named_fact_capacity",
        "test_predictions_accessed": False,
        "source_level": "published_profile_and_typed_edge_tensors",
        "overlap_fact_names": NAMES,
        "overlap_facts_sha256": sha256_file(args.overlap_facts),
        "overlap_audit_sha256": sha256_file(args.overlap_audit),
        "named_86_control_sha256": sha256_file(args.named_86_control),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{"seed": row["seed"],
                                 "control": row["alternatives"]["published_86"]["validation"],
                                 "overlap": row["alternatives"]["published_86_plus_overlap_4"]["validation"]}
                                for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
