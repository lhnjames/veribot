"""Export every BotSim test account's audited source-to-logit evidence card."""

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
from verifiable_bot.botsim_adapter import FEATURE_FIELDS
from verifiable_bot.botsim_native_facts import FACT_NAMES
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


def audited_sources(args: argparse.Namespace) -> dict:
    base = json.loads(args.base_source_audit.read_text(encoding="utf-8"))
    graph = json.loads(args.native_source_audit.read_text(encoding="utf-8"))
    facts_manifest = json.loads(args.native_graph_facts.with_name("manifest.json")
                                .read_text(encoding="utf-8"))
    trace = args.native_graph_facts.with_name("reply_sources.jsonl.gz")
    if (base.get("all_rows_match") is not True
            or graph.get("status") != "verified_full_population_against_raw_comments"
            or base["files_sha256"][str(args.account_file.resolve())] !=
            sha256_file(args.account_file)
            or base["files_sha256"][str(args.activity.resolve())] !=
            sha256_file(args.activity)
            or base["files_sha256"][str((args.data_dir / "features.csv").resolve())] !=
            sha256_file(args.data_dir / "features.csv")
            or graph["input_sha256"]["Users.csv"] != sha256_file(args.account_file)
            or graph["input_sha256"]["user_post_comment.json"] !=
            sha256_file(args.activity)
            or graph["cache_sha256"] != sha256_file(args.native_graph_facts)
            or graph["trace_sha256"] != sha256_file(trace)
            or facts_manifest["facts_sha256"] != graph["cache_sha256"]):
        raise ValueError("BotSim source audit, raw files, or native reply trace differ")
    return {"users_sha256": sha256_file(args.account_file),
            "activity_sha256": sha256_file(args.activity),
            "base_source_audit_sha256": sha256_file(args.base_source_audit),
            "native_source_audit_sha256": sha256_file(args.native_source_audit),
            "native_facts_sha256": sha256_file(args.native_graph_facts),
            "native_trace_sha256": sha256_file(trace)}


def export_seed(args: argparse.Namespace, seed: int, sources: dict) -> dict:
    directory = args.model_dir / f"seed{seed}"
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    saved = json.loads((directory / "test_predictions.json")
                       .read_text(encoding="utf-8"))
    x, y, ids, names, split, current = load(
        "botsim", args.data_dir, args.account_file, seed,
        args.native_graph_facts)
    manifest = report["manifest"]
    if (saved["manifest"] != manifest
            or manifest["files_sha256"] != current["files_sha256"]
            or manifest["split_sha256"] != current["split_sha256"]
            or manifest["feature_names"] != names
            or names != list(FEATURE_FIELDS) + list(FACT_NAMES)
            or sha256_file(directory / "reference.json.gz") !=
            manifest["reference_artifact_sha256"]):
        raise ValueError("frozen BotSim source, split, or reference differs")
    reference = LabelBlindReference.load(directory / "reference.json.gz")
    reasoner = SignConstrainedReasoner.from_manifest(report["reasoner"])
    if (reference.version != manifest["reference_version"]
            or reasoner.reference_version != reference.version
            or reference.names != tuple(names)):
        raise ValueError("reference and reasoner vocabularies differ")
    test = split["test"]
    evidence = reference.transform(x[test])
    design = reasoner._design(evidence)
    per_term = design * reasoner.coef
    logits = float(reasoner.intercept) + per_term.sum(axis=1)
    probabilities = expit(logits)
    rows = saved["predictions"]
    if (len(rows) != len(test)
            or any(row["account_id"] != ids[int(i)] or row["label"] != int(y[i])
                   for row, i in zip(rows, test))
            or np.max(np.abs(probabilities -
                             np.asarray([row["probability"] for row in rows]))) > 1e-12):
        raise ValueError("evidence does not reconstruct frozen test predictions")
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
                    for j, name in enumerate(names):
                        locator = ({"users_csv_row": int(index) + 2,
                                    "activity_account_key": uid,
                                    "operation": "account_features", "fact_name": name}
                                   if j < len(FEATURE_FIELDS) else
                                   {"activity_account_key": uid,
                                    "trace_filter": "source_account_id or target_account_id",
                                    "operation": "native_reply_graph_fact",
                                    "fact_name": name,
                                    "trace_sha256": sources["native_trace_sha256"]})
                        terms.append({
                            "name": name, "group": group_for[name],
                            "source_locator": locator,
                            "source_value": float(x[int(index), j]),
                            "available": bool(evidence.available[offset, j]),
                            "upper_tail_score": float(evidence.upper[offset, j]),
                            "lower_tail_score": float(evidence.lower[offset, j]),
                            "reliability": float(evidence.reliability[offset, j]),
                            "selected_tail": reasoner.tail[j],
                            "fixed_direction": int(reasoner.sign[j]),
                            "nonnegative_coefficient": float(reasoner.coef[j]),
                            "logit_contribution": float(per_term[offset, j]),
                        })
                    groups = {group: float(sum(term["logit_contribution"]
                                               for term in terms if term["group"] == group))
                              for group in reasoner.groups}
                    if any(abs(groups[group] - rows[offset]["group_contributions"][group])
                           > 1e-12 for group in groups):
                        raise ValueError("saved group contributions differ")
                    card = {
                        "dataset": "botsim", "seed": seed, "account_id": uid,
                        "evaluation_label": int(y[index]),
                        "source_level": "raw_user_post_comment_and_native_reply_graph",
                        "source_hashes": sources,
                        "report_sha256": sha256_file(report_path),
                        "reference_version": reference.version,
                        "intercept": float(reasoner.intercept),
                        "terms": terms, "group_contributions": groups,
                        "logit": float(logits[offset]),
                        "probability": float(probabilities[offset]),
                        "threshold": float(manifest["threshold"]),
                        "prediction": int(probabilities[offset] >= manifest["threshold"]),
                    }
                    stream.write(json.dumps(card, separators=(",", ":")) + "\n")
    return {"seed": seed, "test_accounts": len(test),
            "facts_per_account": len(names), "cards_path": str(output),
            "cards_sha256": sha256_file(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path, required=True)
    parser.add_argument("--activity", type=Path, required=True)
    parser.add_argument("--native-graph-facts", type=Path, required=True)
    parser.add_argument("--base-source-audit", type=Path, required=True)
    parser.add_argument("--native-source-audit", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = audited_sources(args)
    result = {"dataset": "botsim", "scope": "all_fixed_test_account_source_to_logit",
              "source_hashes": sources,
              "runs": [export_seed(args, seed, sources) for seed in (0, 1, 2)]}
    manifest = args.output_dir / "evidence_cards_manifest.json"
    manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
