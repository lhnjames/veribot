"""Train the computable-evidence-only VeriBot candidate on three datasets.

This is the first gate of the evidence bottleneck, not a completed v3 result:
the current fact tables still need a full source certificate for every field,
semantic evidence labels, and correction-benefit evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import (
    EvidenceBatch, LabelBlindReference, SignConstrainedReasoner,
)
from verifiable_bot.pairwise_evidence_reasoner import PairwiseEvidenceReasoner
from verifiable_bot.dual_tail_evidence_reasoner import DualTailEvidenceReasoner
from verifiable_bot.mgtab_pair_specs import fixed_pairs
from verifiable_bot.mgtab_neighbor_facts import (RELATION_INDICES, SUPPORT_PRIOR,
                                                  evidence_with_neighbor_support)
from gate0_shallow_tree import best_validation_threshold, evaluate, load, sha256_file


def groups_for(dataset: str, names: list[str]) -> dict[str, list[str]]:
    if dataset == "fox8":
        groups = {
            "profile_counts": ["followers_count", "friends_count", "statuses_count",
                               "favourites_count", "listed_count", "ff_ratio"],
            "profile_presentation": ["verified", "default_profile", "default_profile_image",
                                     "protected", "geo_enabled", "profile_use_background_image",
                                     "has_extended_profile", "contributors_enabled",
                                     "is_translator", "is_translation_enabled",
                                     "profile_background_tile", "description_len", "name_len",
                                     "screen_name_len", "has_location", "has_url", "has_description"],
            "sample_activity": ["n_tweets", "retweet_ratio", "reply_ratio"],
            "sample_response": ["avg_favorites", "avg_retweets", "avg_score"],
            "sample_text": ["avg_tweet_len", "unique_lang_ratio"],
        }
    elif dataset == "botsim":
        groups = {
            "user_activity": ["submission_num", "comment_num", "comment_num_1",
                              "comment_num_2", "n_posts", "n_comments_received",
                              "n_comments_made", "is_active"],
            "user_presentation": ["has_description", "description_len"],
            "community": ["n_subreddits", "n_unique_subs"],
            "post_engagement": ["avg_post_score", "avg_upvote_ratio", "avg_comment_score"],
            "text_length": ["avg_post_len", "avg_comment_len"],
        }
        graph_names = [name for name in names if name.startswith("native_reply_")]
        if graph_names:
            groups["native_reply_graph"] = graph_names
    elif dataset == "mgtab":
        groups = {"published_profile_coordinates": names[:20]}
        if len(names) not in (34, 62, 86):
            raise ValueError("MGTAB requires 34 base, 62 structural, or 86 neighbor facts")
        groups.update({f"published_relation_{i}": names[20 + 2*i:22 + 2*i]
                       for i in range(7)})
        if len(names) >= 62:
            from verifiable_bot.mgtab_source_certificate import RELATIONS
            for i, relation in enumerate(RELATIONS):
                extra = names[34 + 4*i:38 + 4*i]
                if len(extra) != 4 or any(not name.startswith(f"published_{relation}_")
                                          for name in extra):
                    raise ValueError("MGTAB structural fact order differs")
                groups[f"published_relation_{i}"].extend(extra)
        if len(names) == 86:
            from verifiable_bot.mgtab_source_certificate import RELATIONS
            for offset, relation_index in enumerate(RELATION_INDICES):
                relation = RELATIONS[relation_index]
                block = names[62 + 8*offset:62 + 8*(offset+1)]
                if len(block) != 8 or any(not name.startswith(
                        f"published_{relation}_") for name in block):
                    raise ValueError("MGTAB neighbor fact order differs")
                groups[f"published_relation_{relation_index}"].extend(block)
    else:
        raise ValueError(dataset)
    listed = [name for members in groups.values() for name in members]
    if sorted(listed) != sorted(names) or len(listed) != len(names):
        raise ValueError("group specification differs from available facts")
    return groups


def rows(batch: EvidenceBatch, index: np.ndarray) -> EvidenceBatch:
    return EvidenceBatch(batch.upper[index], batch.lower[index],
                         batch.reliability[index], batch.available[index],
                         batch.reference_version, batch.names)


def attach_mgtab_structural(x: np.ndarray, names: list[str], ids: list[str],
                            manifest: dict, data_dir: Path,
                            path: Path | None) -> tuple[np.ndarray, list[str], dict]:
    if path is None:
        return x, names, manifest
    source_manifest_path = path.with_suffix(".json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["facts_sha256"] != sha256_file(path):
        raise ValueError("structural fact cache differs from its source manifest")
    for source_name, expected in source_manifest["input_sha256"].items():
        if sha256_file(data_dir / source_name) != expected:
            raise ValueError(f"MGTAB structural fact source changed: {source_name}")
    with np.load(path, allow_pickle=False) as saved:
        saved_ids = saved["ids"].astype(str).tolist()
        added_names = saved["names"].astype(str).tolist()
        added_values = saved["values"].astype(np.float32)
    if saved_ids != ids or added_names != source_manifest["feature_names"] or \
            added_values.shape != (len(ids), 28) or not np.isfinite(added_values).all():
        raise ValueError("MGTAB structural fact population, vocabulary or values differ")
    if set(added_names) & set(names):
        raise ValueError("MGTAB structural fact names collide with base facts")
    manifest = {**manifest, "files_sha256": {**manifest["files_sha256"],
                str(path.resolve()): sha256_file(path),
                str(source_manifest_path.resolve()): sha256_file(source_manifest_path)},
                "structural_source_level": "published_typed_edge_tensor"}
    return np.column_stack((x, added_values)), names + added_names, manifest


def attach_mgtab_neighbor(x: np.ndarray, names: list[str], ids: list[str],
                          manifest: dict, data_dir: Path, path: Path | None
                          ) -> tuple[np.ndarray, list[str], np.ndarray | None, dict]:
    if path is None:
        return x, names, None, manifest
    if len(names) != 62:
        raise ValueError("MGTAB neighbor facts require the 62-fact structural vocabulary")
    source_manifest_path = path.with_suffix(".json")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest["facts_sha256"] != sha256_file(path):
        raise ValueError("neighbor fact cache differs from source manifest")
    for source_name, expected in source_manifest["input_sha256"].items():
        if sha256_file(data_dir / source_name) != expected:
            raise ValueError(f"MGTAB neighbor fact source changed: {source_name}")
    with np.load(path, allow_pickle=False) as saved:
        saved_ids = saved["ids"].astype(str).tolist()
        added_names = saved["names"].astype(str).tolist()
        added_values = saved["values"].astype(np.float32)
        supports = saved["supports"].astype(np.int32)
    if saved_ids != ids or added_names != source_manifest["feature_names"] or \
            added_values.shape != (len(ids), 24) or supports.shape != added_values.shape or \
            np.any(supports < 0) or not np.array_equal(np.isnan(added_values), supports == 0):
        raise ValueError("MGTAB neighbor population, vocabulary, values or supports differ")
    if set(added_names) & set(names):
        raise ValueError("MGTAB neighbor fact names collide with existing facts")
    manifest = {**manifest, "files_sha256": {**manifest["files_sha256"],
                str(path.resolve()): sha256_file(path),
                str(source_manifest_path.resolve()): sha256_file(source_manifest_path)},
                "neighbor_source_level": "published_profile_and_typed_edge_tensors",
                "neighbor_support_prior": SUPPORT_PRIOR}
    return np.column_stack((x, added_values)), names + added_names, supports, manifest


def run(args: argparse.Namespace, seed: int) -> dict:
    x, y, ids, names, split, manifest = load(args.dataset, args.data_dir,
                                               args.account_file, seed,
                                               args.native_graph_facts)
    if args.mgtab_structural_facts is not None:
        if args.dataset != "mgtab":
            raise ValueError("MGTAB structural fact cache applies only to MGTAB")
        x, names, manifest = attach_mgtab_structural(
            x, names, ids, manifest, args.data_dir, args.mgtab_structural_facts)
    if args.mgtab_neighbor_facts is not None and args.dataset != "mgtab":
        raise ValueError("MGTAB neighbor facts apply only to MGTAB")
    x, names, neighbor_supports, manifest = attach_mgtab_neighbor(
        x, names, ids, manifest, args.data_dir, args.mgtab_neighbor_facts)
    reference = LabelBlindReference(min_context_n=args.min_context_n)
    reference.fit(x[split["train"]], names)
    evidence = evidence_with_neighbor_support(reference, x, neighbor_supports)
    a, b = train_test_split(split["train"], test_size=0.5, stratify=y[split["train"]],
                            random_state=seed)
    if len(set(a) & set(b)) or len(a) + len(b) != len(split["train"]):
        raise AssertionError("direction and coefficient splits must be disjoint")
    if args.pairwise_evidence and args.dual_tail_evidence:
        raise ValueError("pairwise and dual-tail reasoners are distinct choices")
    if args.pairwise_evidence:
        if args.dataset != "mgtab" or args.mgtab_structural_facts is None or \
                args.mgtab_neighbor_facts is not None:
            raise ValueError("fixed pairwise evidence requires MGTAB structural facts")
        reasoner = PairwiseEvidenceReasoner(groups_for(args.dataset, names),
                                            fixed_pairs(names), l2=args.l2)
    elif args.dual_tail_evidence:
        reasoner = DualTailEvidenceReasoner(groups_for(args.dataset, names), l2=args.l2)
    else:
        reasoner = SignConstrainedReasoner(groups_for(args.dataset, names), l2=args.l2)
    reasoner.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
    val_probability = reasoner.predict_proba(rows(evidence, split["val"]))
    threshold = best_validation_threshold(val_probability, y[split["val"]])
    test_index = split["test"]
    test_evidence = rows(evidence, test_index)
    test_probability = reasoner.predict_proba(test_evidence)
    contributions = reasoner.contributions(test_evidence)
    out = args.output / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    reference_path = out / "reference.json.gz"
    reference.save(reference_path)
    report = {
        "manifest": {**manifest, "model": ("computable_evidence_pairwise_same_direction"
                                            if args.pairwise_evidence else
                                            "computable_evidence_dual_tail"
                                            if args.dual_tail_evidence else
                                            "computable_evidence_sign_constrained_additive"),
                     "reference_version": reference.version, "reference_population": "train_mixture",
                     "reference_used_bot_human_labels": False,
                     "direction_split_size": len(a), "coefficient_split_size": len(b),
                     "direction_split_sha256": __import__("hashlib").sha256(
                         json.dumps([ids[int(i)] for i in a]).encode()).hexdigest(),
                     "coefficient_split_sha256": __import__("hashlib").sha256(
                         json.dumps([ids[int(i)] for i in b]).encode()).hexdigest(),
                     "threshold": threshold, "threshold_selected_on": "validation_macro_f1",
                     "reference_artifact_sha256": sha256_file(reference_path),
                     "source_certificate_status": "incomplete"},
        "reasoner": reasoner.manifest(),
        "validation": evaluate(y[split["val"]], val_probability, threshold),
        "test": evaluate(y[test_index], test_probability, threshold),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": report["manifest"],
        "predictions": [{"account_id": ids[int(i)], "label": int(y[i]),
                         "probability": float(test_probability[k]), "split": "test",
                         "group_contributions": {group: float(value[k])
                                                 for group, value in contributions.items()}}
                        for k, i in enumerate(test_index)],
    }, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "test": report["test"], "report": str(out / "report.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fox8", "botsim", "mgtab"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--account-file", type=Path)
    parser.add_argument("--native-graph-facts", type=Path)
    parser.add_argument("--mgtab-structural-facts", type=Path)
    parser.add_argument("--mgtab-neighbor-facts", type=Path)
    parser.add_argument("--pairwise-evidence", action="store_true")
    parser.add_argument("--dual-tail-evidence", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--min-context-n", type=int, default=30)
    parser.add_argument("--l2", type=float, default=0.01)
    args = parser.parse_args()
    reports = [run(args, seed) for seed in args.seeds]
    print(json.dumps({"dataset": args.dataset, "runs": reports}, indent=2))


if __name__ == "__main__":
    main()
