"""Export complete source-to-logit cards for every Fox8 test account."""

from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.fox8_adapter import FEATURE_FIELDS
from verifiable_bot.fox8_native_interaction_facts import FACT_NAMES
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def sources(args: argparse.Namespace) -> dict:
    base = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    native = json.loads(args.native_source_audit.read_text(encoding="utf-8"))
    native_manifest = json.loads(args.native_facts.with_suffix(".json")
                                 .read_text(encoding="utf-8"))
    event_index = args.native_facts.with_name("interaction_event_index.json")
    if (base.get("all_rows_match") is not True
            or native.get("status") !=
            "verified_all_native_interaction_fact_cells_and_event_locators"
            or base["files_sha256"][str(args.raw.resolve())] != sha256_file(args.raw)
            or base["files_sha256"][str((args.data_dir / "features.csv").resolve())] !=
            sha256_file(args.data_dir / "features.csv")
            or native["raw_source_sha256"] != sha256_file(args.raw)
            or native["fact_cache_sha256"] != sha256_file(args.native_facts)
            or native["event_index_sha256"] != sha256_file(event_index)
            or native_manifest["facts_sha256"] != native["fact_cache_sha256"]):
        raise ValueError("Fox8 raw source, native event index or audits differ")
    return {"raw_sha256": sha256_file(args.raw),
            "base_audit_sha256": sha256_file(args.base_source_audit),
            "native_audit_sha256": sha256_file(args.native_source_audit),
            "native_facts_sha256": sha256_file(args.native_facts),
            "event_index_sha256": sha256_file(event_index)}


def export_seed(args: argparse.Namespace, seed: int, source_hashes: dict) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json")
                       .read_text(encoding="utf-8"))
    x, y, ids, names, split, current = load("fox8", args.data_dir, None, seed)
    with np.load(args.native_facts, allow_pickle=False) as cache:
        if (cache["ids"].astype(str).tolist() != ids
                or cache["names"].astype(str).tolist() != list(FACT_NAMES)):
            raise ValueError("native Fox8 account order or vocabulary differs")
        matrix = np.column_stack((x, cache["values"].astype(float)))
    vocabulary = names + list(FACT_NAMES)
    manifest = report["manifest"]
    if (saved["manifest"] != manifest
            or current["files_sha256"] != manifest["files_sha256"]
            or current["split_sha256"] != manifest["split_sha256"]
            or names != list(FEATURE_FIELDS)
            or manifest["raw_source_sha256"] != source_hashes["raw_sha256"]
            or manifest["native_facts_sha256"] != source_hashes["native_facts_sha256"]
            or manifest["native_event_index_sha256"] !=
            source_hashes["event_index_sha256"]
            or sha256_file(directory / "reference.json.gz") !=
            manifest["reference_artifact_sha256"]):
        raise ValueError("frozen Fox8 model, raw source or split differs")
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    reasoner = MonotoneSplineReasoner.from_manifest(report["reasoner"])
    if (reference.version != manifest["reference_version"]
            or reference.names != tuple(vocabulary)
            or reasoner.reference_version != reference.version):
        raise ValueError("reference and reasoner differ")
    test = split["test"]
    evidence = reference.transform(matrix[test])
    design = reasoner._design(evidence)
    per_term = np.column_stack([
        design[:, reasoner._slices[name]] @ reasoner.coef[reasoner._slices[name]]
        for name in vocabulary])
    logits = float(reasoner.intercept) + per_term.sum(axis=1)
    probabilities = expit(logits)
    rows = saved["predictions"]
    if (len(rows) != len(test)
            or any(row["account_id"] != ids[int(i)] or row["label"] != int(y[i])
                   for row, i in zip(rows, test))
            or np.max(np.abs(probabilities -
                             np.asarray([row["probability"] for row in rows]))) > 1e-12):
        raise ValueError("evidence does not reconstruct frozen Fox8 predictions")
    group_for = {name: group for group, members in reasoner.groups.items()
                 for name in members}
    output = args.output_dir / f"seed{seed}_evidence_cards.jsonl.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for offset, index in enumerate(test):
                    uid = ids[int(index)]
                    terms = []
                    for j, name in enumerate(vocabulary):
                        sl = reasoner._slices[name]
                        locator = ({"raw_account_key": uid,
                                    "operation": "profile_features_or_tweet_features",
                                    "fact_name": name}
                                   if j < len(FEATURE_FIELDS) else
                                   {"raw_account_key": uid,
                                    "event_index_query": "source_id or target_id",
                                    "event_index_sha256": source_hashes["event_index_sha256"],
                                    "operation": "sampled_native_interaction_fact",
                                    "fact_name": name})
                        terms.append({
                            "name": name, "group": group_for[name],
                            "source_locator": locator,
                            "source_value": (float(matrix[int(index), j])
                                             if np.isfinite(matrix[int(index), j]) else None),
                            "available": bool(evidence.available[offset, j]),
                            "upper_tail_score": float(evidence.upper[offset, j]),
                            "lower_tail_score": float(evidence.lower[offset, j]),
                            "reliability": float(evidence.reliability[offset, j]),
                            "selected_tail": reasoner.tail[j],
                            "fixed_direction": int(reasoner.sign[j]),
                            "knots": list(reasoner.knots[name]),
                            "nonnegative_slopes": reasoner.coef[sl].tolist(),
                            "logit_contribution": float(per_term[offset, j]),
                        })
                    groups = {group: float(sum(term["logit_contribution"]
                                               for term in terms if term["group"] == group))
                              for group in reasoner.groups}
                    if any(abs(groups[group] - rows[offset]["group_contributions"][group])
                           > 1e-12 for group in groups):
                        raise ValueError("saved Fox8 group contributions differ")
                    card = {"dataset": "fox8", "seed": seed, "account_id": uid,
                            "evaluation_label": int(y[index]),
                            "source_level": "raw_profile_sampled_tweets_and_native_targets",
                            "source_hashes": source_hashes,
                            "report_sha256": sha256_file(report_path),
                            "reference_version": reference.version,
                            "intercept": float(reasoner.intercept),
                            "terms": terms, "group_contributions": groups,
                            "logit": float(logits[offset]),
                            "probability": float(probabilities[offset]),
                            "threshold": float(manifest["threshold"]),
                            "prediction": int(probabilities[offset] >= manifest["threshold"])}
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
    return {"seed": seed, "test_accounts": len(test),
            "facts_per_account": len(vocabulary),
            "cards_path": str(output), "cards_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--native-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--native-source-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_hashes = sources(args)
    result = {"dataset": "fox8", "scope": "all_fixed_test_account_source_to_logit",
              "source_hashes": source_hashes,
              "runs": [export_seed(args, seed, source_hashes) for seed in (0, 1, 2)]}
    path = args.output_dir / "evidence_cards_manifest.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
