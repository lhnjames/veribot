"""Dev-only 20-fact tradeoff: replace four anonymous content projections by audited overlaps."""

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
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_atomic, rank_content
from train_v3_computable import rows
from verifiable_bot.mgtab_relation_overlap_facts import NAMES
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def evidence(reference: LabelBlindReference, matrix: np.ndarray,
             neighbor_support: np.ndarray, overlap_support: np.ndarray) -> EvidenceBatch:
    base = reference.transform(matrix)
    reliability = base.reliability.copy()
    reliability[:, 62:86] *= neighbor_support / (neighbor_support + 5.0)
    reliability[:, 150:154] *= overlap_support / (overlap_support + 5.0)
    return EvidenceBatch(base.upper, base.lower, reliability, base.available,
                         base.reference_version, base.names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--overlap-facts", type=Path, required=True)
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument("--full-model-dir", type=Path, required=True)
    parser.add_argument("--compact-model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.dataset = "mgtab"
    args.account_file = None
    args.native_graph_facts = None
    args.fox8_native_facts = None
    audit = json.loads(args.overlap_audit.read_text(encoding="utf-8"))
    if (audit["status"] != "all_relation_overlap_fact_cells_replayed"
            or audit["facts_sha256"] != sha256_file(args.overlap_facts)):
        raise ValueError("relation-overlap fact cache differs from full audit")
    with np.load(args.overlap_facts, allow_pickle=False) as cache:
        if (not np.array_equal(cache["ids"], np.arange(10199))
                or tuple(cache["names"].astype(str)) != NAMES):
            raise ValueError("relation-overlap account or fact names differ")
        overlap = cache["values"].astype(float)
        overlap_support = cache["supports"].astype(float)
    runs = []
    for seed in (0, 1, 2):
        full_path = args.full_model_dir / f"seed{seed}" / "report.json"
        compact_path = args.compact_model_dir / f"seed{seed}" / "report.json"
        full = json.loads(full_path.read_text(encoding="utf-8"))
        compact = json.loads(compact_path.read_text(encoding="utf-8"))
        args.selected_model_dir = args.full_model_dir
        x, y, names, split = selected_facts(args, seed, full)
        if (x.shape != (10199, 150) or len(names) != 150
                or compact["manifest"]["split_sha256"] != full["manifest"]["split_sha256"]):
            raise ValueError("fixed MGTAB split or vocabulary differs")
        extended = np.column_stack((x, overlap))
        vocabulary = names + list(NAMES)
        train, dev = split["train"], split["val"]
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        reference = LabelBlindReference(min_context_n=30).fit(
            extended[train], vocabulary)
        atomic_train = evidence(reference, extended[train],
                                split["supports"][train], overlap_support[train])
        atomic_dev = evidence(reference, extended[dev],
                              split["supports"][dev], overlap_support[dev])
        structural = rank_atomic(atomic_train, y[train], a, list(range(86)), 4)
        content16 = rank_content(atomic_train, y[train], a, 16)
        content12 = rank_content(atomic_train, y[train], a, 12)
        if ([vocabulary[j] for j in structural + content16] !=
                compact["manifest"]["selected_fact_names"]):
            raise ValueError("existing 20-fact A selection does not replay")
        alternatives = {}
        for label, selected in (
            ("fixed_4_named_16_content", structural + content16),
            ("four_named_overlap_replace_four_content",
             structural + content12 + list(range(150, 154))),
        ):
            compact_train = columns(atomic_train, selected)
            compact_dev = columns(atomic_dev, selected)
            groups = {name: [name] for name in compact_train.names}
            model = MonotoneSplineReasoner(groups, l2=1e-5)
            model.fit(rows(compact_train, a), y[train][a],
                      rows(compact_train, b), y[train][b])
            probability = model.predict_proba(compact_dev)
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[label] = {
                "fact_names": list(compact_train.names),
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
                "active_overlap_facts": sum(any(value > 0 for value in
                    model.manifest()["coefficients"][name]) for name in NAMES)
                    if label.startswith("four_named_overlap") else 0,
            }
        expected = compact["validation"]
        actual = alternatives["fixed_4_named_16_content"]["validation"]
        if any(abs(actual[key] - expected[key]) > 1e-9
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("existing compact model validation did not replay")
        runs.append({
            "seed": seed, "split_sha256": full["manifest"]["split_sha256"],
            "compact_model_report_sha256": sha256_file(compact_path),
            "reference_version": reference.version,
            "alternatives": alternatives,
        })
    result = {
        "dataset": "mgtab", "scope": "official_train_dev_only_fixed_20_fact_budget",
        "test_predictions_accessed": False,
        "source_level": "published_typed_edges_profile_and_mean_tweet_tensors",
        "overlap_facts_sha256": sha256_file(args.overlap_facts),
        "overlap_audit_sha256": sha256_file(args.overlap_audit),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": [{"seed": row["seed"],
                                 "control": row["alternatives"]["fixed_4_named_16_content"]["validation"],
                                 "replacement": row["alternatives"]["four_named_overlap_replace_four_content"]["validation"],
                                 "active_overlap": row["alternatives"]["four_named_overlap_replace_four_content"]["active_overlap_facts"]}
                                for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
