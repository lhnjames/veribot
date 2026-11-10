"""Classify accounts only after binding facts to independently audited sources.

This entry point accepts account IDs, never a caller-supplied feature matrix.
It rebuilds BotSim/Fox8 raw-record facts or MGTAB published-tensor facts,
checks the independent source certificate, then invokes the frozen evidence
bottleneck. It does not open classification labels or saved test predictions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verify_compact_raw_source_predictions import CONFIG, source_inputs
from verifiable_bot.compact_evidence_runtime import CompactEvidenceRuntime
from verifiable_bot.decision_dependency import additive_dependency
from verifiable_bot.mgtab_profile_dictionary import (
    PAPER_URL, published_profile_description)


def certificate_binding(root: Path, dataset: str, config: dict) -> dict:
    path = root / "research/certificates" / {
        "botsim": "botsim24_full_source_audit.json",
        "fox8": "fox8_native_interaction_full_source_audit.json",
        "mgtab": "mgtab_full_source_audit.json",
    }[dataset]
    cert = json.loads(path.read_text(encoding="utf-8"))
    checks: dict[str, tuple[Path, str]]
    if dataset == "botsim":
        if cert.get("status") != "verified_full_population_against_raw_comments":
            raise ValueError("BotSim independent raw-source audit is absent")
        checks = {
            "users": (root / config["account"], cert["input_sha256"]["Users.csv"]),
            "activity": (root / config["activity"], cert["input_sha256"]["user_post_comment.json"]),
            "native_reply_facts": (root / config["native"], cert["cache_sha256"]),
            "reply_event_trace": (root / "data/processed/botsim_v3/reply_sources.jsonl.gz",
                                  cert["trace_sha256"]),
        }
    elif dataset == "fox8":
        if (cert.get("status") !=
                "verified_all_native_interaction_fact_cells_and_event_locators"):
            raise ValueError("Fox8 independent interaction audit is absent")
        checks = {
            "raw_profile_and_tweets": (root / config["raw"], cert["raw_source_sha256"]),
            "native_interaction_facts": (root / config["native"], cert["fact_cache_sha256"]),
            "interaction_event_index": (
                root / "data/processed/fox8_v3/interaction_event_index.json",
                cert["event_index_sha256"]),
        }
    else:
        if cert.get("status") != "verified_all_published_tensor_derived_fact_cells":
            raise ValueError("MGTAB independent published-source audit is absent")
        data_dir = Path(config["data"])
        checks = {
            name: (data_dir / name, digest)
            for name, digest in cert["source_sha256"].items()
        }
        checks["structural_facts"] = (root / config["structural"],
                                      cert["structural_facts_sha256"])
        checks["neighbor_facts"] = (root / config["neighbor"],
                                    cert["neighbor_facts_sha256"])
    actual = {}
    for name, (file, expected) in checks.items():
        digest = sha256_file(file)
        if digest != expected:
            raise ValueError(f"{dataset} {name} differs from independent audit")
        actual[name] = digest
    return {"certificate": str(path), "certificate_sha256": sha256_file(path),
            "source_file_sha256": actual}


def recompute_query(root: Path, dataset: str, config: dict, seed: int,
                    account_id: str, fact_name: str | None) -> dict | None:
    """Give a complete-account source query for each selected atomic fact."""
    if fact_name is None:
        return None
    if dataset == "botsim":
        if fact_name.startswith("native_reply_"):
            sources = [f"{root / config['account']}#all_user_names",
                       f"{root / config['activity']}#all_comments_and_parent_objects,select_account={account_id}",
                       f"{root / 'data/processed/botsim_v3/reply_sources.jsonl.gz'}#incident_account={account_id}"]
        else:
            sources = [f"{root / config['account']}#user_id={account_id}",
                       f"{root / config['activity']}#account_id={account_id}"]
    elif dataset == "fox8":
        if fact_name.startswith(("native_reply_", "native_retweet_")):
            sources = [f"{root / config['raw']}#all_interaction_events,select_account={account_id}",
                       f"{root / 'data/processed/fox8_v3/interaction_event_index.json'}#incident_account={account_id}"]
        else:
            sources = [f"{root / config['raw']}#user_id={account_id}"]
    elif fact_name.startswith("published_tweet_projection_"):
        sources = [f"{Path(config['data']) / 'features.pt'}#row={account_id},columns=20:788",
                   f"{root / config['full'] / f'seed{seed}' / 'projection.npz'}#train_only_pca"]
    elif fact_name.startswith("published_profile_coordinate_"):
        coordinate = int(fact_name.rsplit("_", 1)[1])
        sources = [f"{Path(config['data']) / 'features.pt'}#row={account_id},column={coordinate}"]
    else:
        sources = [f"{Path(config['data']) / 'edge_index.pt'}#incident_node={account_id}",
                   f"{Path(config['data']) / 'edge_type.pt'}#matching_edge_rows"]
        if "neighbor_" in fact_name:
            sources.append(f"{Path(config['data']) / 'features.pt'}#neighbor_rows")
    return {"fact_name": fact_name, "account_id": account_id,
            "source_ids": sources,
            "operation": "recompute_frozen_source_fact_then_label_blind_reference_and_concept"}


def predict(root: Path, dataset: str, seed: int, account_ids: list[str]) -> dict:
    if dataset not in CONFIG or seed not in (0, 1, 2):
        raise ValueError("unsupported dataset or frozen seed")
    if not account_ids or len(account_ids) != len(set(account_ids)):
        raise ValueError("request at least one unique account ID")
    config = CONFIG[dataset]
    source_binding = certificate_binding(root, dataset, config)
    bot_deployment = None
    if dataset == "botsim":
        deployment_dir = root / "output/v3_current_compact_deployment"
        manifest_path = deployment_dir / "botsim_manifest.json"
        certificate_path = root / "research/certificates/botsim_compact_deployment_manifest_audit.json"
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        if (certificate["status"] != "deployment_manifest_bound_to_source_and_model_audits"
                or certificate["manifest_sha256"] != sha256_file(manifest_path)):
            raise ValueError("BotSim inference manifest differs from audit")
        bot_deployment = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (bot_deployment["status"] != "inference_only_metadata_from_audited_frozen_model"
                or bot_deployment["source_audit_sha256"] !=
                source_binding["certificate_sha256"]):
            raise ValueError("BotSim source and deployment audits differ")
        sealed_run = bot_deployment["runs"][seed]
        if (sealed_run["seed"] != seed
                or sealed_run["source_metadata_sha256"] != sha256_file(
                    deployment_dir / f"botsim_source_seed{seed}.json")
                or sealed_run["runtime_metadata_sha256"] != sha256_file(
                    deployment_dir / f"botsim_runtime_seed{seed}.json")):
            raise ValueError("BotSim inference sidecar differs from sealed model")
    profile_dictionary_binding = None
    if dataset == "mgtab":
        dictionary_path = root / "research/certificates/mgtab_published_profile_dictionary_audit.json"
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
        if (dictionary["status"] != "published_profile_dimensions_match_table_a5_types"
                or dictionary["paper_table_a5"] != PAPER_URL
                or dictionary["features_sha256"] !=
                source_binding["source_file_sha256"]["features.pt"]):
            raise ValueError("MGTAB published profile dictionary differs from source")
        profile_dictionary_binding = {
            "audit_sha256": sha256_file(dictionary_path),
            "table_a5_reference": PAPER_URL,
            "source_level": "published_normalized_profile_feature",
        }

    original_csv = pd.read_csv
    original_torch_load = torch.load

    def checked_csv(path, *args, **kwargs):
        if Path(path).name == "labels.csv":
            raise AssertionError("classification labels entered source-bound inference")
        return original_csv(path, *args, **kwargs)

    def checked_torch_load(path, *args, **kwargs):
        if Path(path).name == "labels_bot.pt":
            raise AssertionError("classification labels entered source-bound inference")
        return original_torch_load(path, *args, **kwargs)

    with patch.object(pd, "read_csv", checked_csv), patch.object(torch, "load", checked_torch_load):
        names, values, source_difference, source_level = source_inputs(
            root, dataset, config, account_ids, seed=seed,
            inference_metadata=(deployment_dir / f"botsim_source_seed{seed}.json"
                                if bot_deployment is not None else None))
        if dataset == "botsim":
            ids = pd.read_csv(root / config["account"])["user_id"].astype(str).tolist()
        elif dataset == "fox8":
            ids = pd.read_csv(Path(config["data"]) / "features.csv")[
                "user_id"].astype(str).tolist()
        else:
            ids = [str(i) for i in range(len(values))]
    index = {account_id: i for i, account_id in enumerate(ids)}
    if len(index) != len(ids) or not set(account_ids) <= set(index):
        raise ValueError("requested account is absent from audited source population")
    runtime = CompactEvidenceRuntime(
        dataset, root / config["selected"], root / config["full"], seed,
        root / config["groups"] if config.get("groups") else None,
        deployment_metadata=(deployment_dir / f"botsim_runtime_seed{seed}.json"
                             if bot_deployment is not None else None))
    selected = np.asarray([index[account_id] for account_id in account_ids], dtype=int)
    decisions = runtime.predict(values[selected], names, account_ids)
    for decision in decisions:
        for term in decision["terms"]:
            term["recompute_query"] = recompute_query(
                root, dataset, config, seed, decision["account_id"],
                term["source_fact_name"])
            if dataset == "mgtab" and term["source_fact_name"] is not None:
                description = published_profile_description(term["source_fact_name"])
                if description is not None:
                    term["published_profile_semantics"] = description
        decision["decision_dependency"] = additive_dependency(
            [term["name"] for term in decision["terms"]],
            [term["logit_contribution"] for term in decision["terms"]],
            decision["intercept"], decision["threshold"],
            decision["logit"], decision["prediction"])
    full_run = root / config["full"] / f"seed{seed}"
    selected_run = root / config["selected"] / f"seed{seed}"
    if bot_deployment is not None:
        model_binding = {
            "deployment_manifest_sha256": sha256_file(manifest_path),
            "full_report_sha256": sealed_run["full_report_sha256"],
            "selected_report_sha256": sealed_run["selected_report_sha256"],
            "reference_sha256": sealed_run["reference_sha256"],
        }
    else:
        model_binding = {
            "full_report_sha256": sha256_file(full_run / "report.json"),
            "selected_report_sha256": sha256_file(selected_run / "report.json"),
            "reference_sha256": sha256_file(full_run / "reference.json.gz"),
        }
    if dataset == "mgtab":
        model_binding["train_only_projection_sha256"] = sha256_file(
            full_run / "projection.npz")
    if dataset == "fox8":
        model_binding["prespecified_groups_sha256"] = sha256_file(
            root / config["groups"])
    output = {
        "dataset": dataset, "seed": seed,
        "source_level": source_level,
        "source_binding": source_binding,
        "model_binding": model_binding,
        "max_raw_to_released_fact_difference": source_difference,
        "classification_labels_accessed": False,
        "caller_supplied_fact_matrix_accepted": False,
        "decisions": decisions,
    }
    if profile_dictionary_binding is not None:
        output["published_profile_dictionary_binding"] = profile_dictionary_binding
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", choices=tuple(CONFIG), required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--account-id", action="append", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(args.project_root.resolve(), args.dataset, args.seed,
                     args.account_id)
    serialized = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
