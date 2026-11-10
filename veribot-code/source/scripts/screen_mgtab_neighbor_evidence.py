"""Validation-only screen of MGTAB source-derived neighbor profile evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file
from train_v3_computable import groups_for, rows
from verifiable_bot.mgtab_neighbor_facts import (RELATION_INDICES, SUPPORT_PRIOR)
from verifiable_bot.mgtab_source_certificate import RELATIONS
from verifiable_bot.v3_evidence_reasoner import (EvidenceBatch, LabelBlindReference,
                                                SignConstrainedReasoner)
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner


def read_cache(path: Path, data_dir: Path, n_accounts: int,
               *, supports_required: bool) -> tuple[np.ndarray, list[str], np.ndarray | None, dict]:
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256_file(path) != manifest["facts_sha256"]:
        raise ValueError("fact cache differs from source manifest")
    for name, digest in manifest["input_sha256"].items():
        if sha256_file(data_dir / name) != digest:
            raise ValueError(f"published MGTAB source changed: {name}")
    with np.load(path, allow_pickle=False) as cached:
        ids = cached["ids"].astype(int)
        names = cached["names"].astype(str).tolist()
        values = cached["values"].astype(np.float32)
        supports = cached["supports"].astype(np.int32) if supports_required else None
    if not np.array_equal(ids, np.arange(n_accounts)) or \
            names != manifest["feature_names"] or \
            values.shape != (n_accounts, len(names)):
        raise ValueError("fact cache population or vocabulary differs")
    if supports_required:
        if supports is None or supports.shape != values.shape or \
                np.any(supports < 0) or not np.array_equal(np.isnan(values), supports == 0):
            raise ValueError("neighbor support counts and missing values differ")
    elif not np.isfinite(values).all():
        raise ValueError("nonfinite structural fact")
    return values, names, supports, manifest


def validate(x: np.ndarray, y: np.ndarray, names: list[str], groups: dict,
             split: dict, seed: int, neighbor_supports: np.ndarray | None = None,
             dual_tail: bool = False
             ) -> dict:
    reference = LabelBlindReference(min_context_n=30).fit(x[split["train"]], names)
    evidence = reference.transform(x)
    if neighbor_supports is not None:
        if neighbor_supports.shape != (len(x), 24):
            raise ValueError("neighbor support matrix shape differs")
        reliability = evidence.reliability.copy()
        reliability[:, -24:] *= neighbor_supports / (neighbor_supports + SUPPORT_PRIOR)
        evidence = EvidenceBatch(evidence.upper, evidence.lower, reliability,
                                 evidence.available, evidence.reference_version,
                                 evidence.names)
    a, b = train_test_split(split["train"], test_size=0.5,
                            stratify=y[split["train"]], random_state=seed)
    reasoner = (DualTailEvidenceReasoner(groups, l2=0.01) if dual_tail else
                SignConstrainedReasoner(groups, l2=0.01))
    reasoner.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
    probability = reasoner.predict_proba(rows(evidence, split["val"]))
    threshold = best_validation_threshold(probability, y[split["val"]])
    return {"validation": evaluate(y[split["val"]], probability, threshold),
            "threshold": threshold, "reference_version": reference.version,
            "nonzero_coefficients": int(np.count_nonzero(reasoner.coef > 0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--structural-facts", type=Path, required=True)
    parser.add_argument("--neighbor-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--dual-tail", action="store_true")
    args = parser.parse_args()
    runs = []
    for seed in args.seeds:
        x, y, ids, names, split, source = load("mgtab", args.data_dir, None, seed)
        structural, structural_names, _, structural_manifest = read_cache(
            args.structural_facts, args.data_dir, len(ids), supports_required=False)
        neighbor, neighbor_names, supports, neighbor_manifest = read_cache(
            args.neighbor_facts, args.data_dir, len(ids), supports_required=True)
        if len(structural_names) != 28 or len(neighbor_names) != 24:
            raise ValueError("fixed MGTAB evidence vocabulary differs")
        groups = groups_for("mgtab", names + structural_names)
        for offset, relation_index in enumerate(RELATION_INDICES):
            block = neighbor_names[8*offset:8*(offset+1)]
            relation = RELATIONS[relation_index]
            if len(block) != 8 or any(not name.startswith(f"published_{relation}_")
                                      for name in block):
                raise ValueError("neighbor fact relation order differs")
            groups[f"published_relation_{relation_index}"].extend(block)
        base_x = np.column_stack((x, structural))
        candidate_x = np.column_stack((base_x, neighbor))
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "base_62": validate(base_x, y, names + structural_names,
                                         groups_for("mgtab", names + structural_names),
                                         split, seed, dual_tail=args.dual_tail),
                     "neighbor_86": validate(candidate_x, y,
                                             names + structural_names + neighbor_names,
                                             groups, split, seed, supports,
                                             dual_tail=args.dual_tail)})
    result = {
        "dataset": "mgtab", "scope": "validation_only",
        "test_predictions_accessed": False,
        "structural_facts_sha256": structural_manifest["facts_sha256"],
        "neighbor_facts_sha256": neighbor_manifest["facts_sha256"],
        "neighbor_reliability_factor": "support/(support+5)",
        "reasoner": "dual_tail" if args.dual_tail else "single_tail",
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scope": result["scope"],
                      "runs": [{"seed": row["seed"],
                                "base_62": row["base_62"]["validation"],
                                "neighbor_86": row["neighbor_86"]["validation"]}
                               for row in runs]}, indent=2))


if __name__ == "__main__":
    main()
