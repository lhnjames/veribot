"""Validation-only MGTAB test of typed-neighbor tweet-vector similarity facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load
from screen_mgtab_neighbor_evidence import read_cache
from train_v3_computable import groups_for, rows
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.mgtab_content_relation_facts import RELATION_INDICES
from verifiable_bot.mgtab_source_certificate import RELATIONS
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference


def evidence_for(x: np.ndarray, names: list[str], train: np.ndarray,
                 profile_supports: np.ndarray, content_supports: np.ndarray | None
                 ) -> tuple[LabelBlindReference, EvidenceBatch]:
    reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
    base = reference.transform(x)
    reliability = base.reliability.copy()
    reliability[:, 62:86] *= profile_supports / (profile_supports + 5.0)
    if content_supports is not None:
        reliability[:, 86:92] *= content_supports / (content_supports + 5.0)
    return reference, EvidenceBatch(base.upper, base.lower, reliability,
                                    base.available, base.reference_version,
                                    base.names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--content-relation-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        structure, structure_names, _, structure_manifest = read_cache(
            args.structural_facts, args.data_dir, len(ids), supports_required=False)
        neighbor, neighbor_names, neighbor_supports, neighbor_manifest = read_cache(
            args.neighbor_facts, args.data_dir, len(ids), supports_required=True)
        content, content_names, content_supports, content_manifest = read_cache(
            args.content_relation_facts, args.data_dir, len(ids), supports_required=True)
        if len(structure_names) != 28 or len(neighbor_names) != 24 or \
                len(content_names) != 6 or len(set(names + structure_names +
                                                  neighbor_names + content_names)) != 92:
            raise ValueError("MGTAB fixed evidence vocabulary differs")
        base_names = names + structure_names + neighbor_names
        groups = groups_for("mgtab", base_names)
        for offset, relation_index in enumerate(RELATION_INDICES):
            block = content_names[2*offset:2*(offset+1)]
            if any(not item.startswith(f"published_{RELATIONS[relation_index]}_")
                   for item in block):
                raise ValueError("content relation vocabulary order differs")
            groups[f"published_relation_{relation_index}"].extend(block)
        a, b = train_test_split(split["train"], test_size=0.5,
                                stratify=y[split["train"]], random_state=seed)
        result = {}
        for label, values, vocab, supports in (
            ("neighbor_86", np.column_stack((x, structure, neighbor)),
             base_names, None),
            ("content_relation_92", np.column_stack((x, structure, neighbor, content)),
             base_names + content_names, content_supports),
        ):
            reference, evidence = evidence_for(values, vocab, split["train"],
                                               neighbor_supports, supports)
            model_groups = (groups_for("mgtab", base_names) if supports is None
                            else groups)
            model = DualTailEvidenceReasoner(model_groups, l2=0.01)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, split["val"]))
            threshold = best_validation_threshold(probability, y[split["val"]])
            result[label] = {
                "validation": evaluate(y[split["val"]], probability, threshold),
                "threshold": threshold,
                "reference_version": reference.version,
                "nonzero_coefficients": int(np.count_nonzero(model.coef > 0)),
            }
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     **result})
    output = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "structural_facts_sha256": structure_manifest["facts_sha256"],
        "neighbor_facts_sha256": neighbor_manifest["facts_sha256"],
        "content_relation_facts_sha256": content_manifest["facts_sha256"],
        "content_relation_source_level": content_manifest["source_level"],
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": output["scope"], "runs": [
        {"seed": row["seed"],
         "neighbor_86": row["neighbor_86"]["validation"],
         "content_relation_92": row["content_relation_92"]["validation"]}
        for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
