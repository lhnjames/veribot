"""Official-split TwiBot-20 gate-0 tree and computable-evidence candidate.

Graph facts come from actual directed follow lists. The cache is label-free
apart from a separate label vector; train labels enter only the diagnostic tree
or the A/B evidence reasoner. Dev chooses the threshold, test is reported once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiable_bot.v3_evidence_reasoner import (
    EvidenceBatch, LabelBlindReference, SignConstrainedReasoner,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    predicted = p >= threshold
    return {"n": len(y), "acc": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
            "auroc": float(roc_auc_score(y, p)),
            "errors": int(np.sum(predicted != y))}


def threshold_on_dev(probability: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.r_[0.5, probability, np.nextafter(1.0, 2.0)])
    scores = [f1_score(labels, probability >= t, average="macro", zero_division=0)
              for t in candidates]
    tied = np.flatnonzero(np.isclose(scores, max(scores)))
    return float(candidates[tied[np.argmin(np.abs(candidates[tied] - 0.5))]])


def select(evidence: EvidenceBatch, index: np.ndarray) -> EvidenceBatch:
    return EvidenceBatch(evidence.upper[index], evidence.lower[index],
                         evidence.reliability[index], evidence.available[index],
                         evidence.reference_version, evidence.names)


def groups(names: list[str]) -> dict[str, list[str]]:
    result = {
        "profile_size": ["followers_count", "friends_count", "statuses_count",
                         "listed_count", "favourites_count"],
        "profile_presentation": ["verified", "protected", "default_profile",
                                 "default_profile_image", "description_length"],
        "tweet_sample": ["tweet_sample_count", "retweet_fraction", "url_fraction",
                         "hashtag_fraction", "mention_fraction", "mean_tweet_length",
                         "unique_tweet_fraction"],
        "native_follow_graph": ["sampled_following_count", "sampled_follower_count",
                                "graph_out_degree", "graph_in_degree",
                                "graph_reciprocal_degree"],
    }
    listed = [name for members in result.values() for name in members]
    if sorted(listed) != sorted(names) or len(listed) != len(names):
        raise ValueError("TwiBot-20 group vocabulary differs from fact table")
    return result


def run(args: argparse.Namespace, seed: int) -> dict:
    with np.load(args.facts, allow_pickle=False) as cached:
        x = cached["values"].astype(float)
        y = cached["labels"].astype(int)
        ids = cached["ids"].astype(str)
        split = cached["split"].astype(str)
        contexts = cached["contexts"].astype(str)
        names = cached["names"].astype(str).tolist()
    source = json.loads(args.facts.with_suffix(".json").read_text(encoding="utf-8"))
    if source["dataset"] != "twibot20" or x.shape != (11826, len(names)):
        raise ValueError("TwiBot-20 source cache differs from audited population")
    train = np.flatnonzero(split == "train")
    dev = np.flatnonzero(split == "dev")
    test = np.flatnonzero(split == "test")
    if (len(train), len(dev), len(test)) != (8278, 2365, 1183):
        raise ValueError("official train/dev/test split sizes differ")
    out = args.output / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": "twibot20", "seed": seed,
                "official_split": True, "support_labels_used": False,
                "facts_sha256": sha256(args.facts),
                "source_archive_sha256": source["archive_sha256"],
                "native_follow_edges": source["directed_follow_edges"],
                "fact_names": names}

    imputer = SimpleImputer(strategy="median", add_indicator=True).fit(x[train])
    filled = imputer.transform(x)
    tree_names = names + [f"missing_{names[j]}" for j in imputer.indicator_.features_]
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=seed)
    tree.fit(filled[train], y[train])
    tree_val = tree.predict_proba(filled[dev])[:, 1]
    tree_threshold = threshold_on_dev(tree_val, y[dev])
    tree_test = tree.predict_proba(filled[test])[:, 1]
    tree_report = {"model": "depth4_tree_shortcut_control",
                   "manifest": {**manifest, "missing_imputation": "train_median_plus_indicator",
                                "threshold": tree_threshold},
                   "dev": metric(y[dev], tree_val, tree_threshold),
                   "test": metric(y[test], tree_test, tree_threshold),
                   "tree": export_text(tree, feature_names=tree_names),
                   "feature_importances": {name: float(value)
                                           for name, value in zip(tree_names, tree.feature_importances_)
                                           if value > 0}}
    (out / "gate0_tree.json").write_text(json.dumps(tree_report, indent=2) + "\n",
                                          encoding="utf-8")

    reference = LabelBlindReference(min_context_n=30).fit(x[train], names,
                                                             contexts=contexts[train])
    evidence = reference.transform(x, contexts=contexts)
    a, b = train_test_split(train, test_size=0.5, stratify=y[train], random_state=seed)
    reasoner = SignConstrainedReasoner(groups(names), l2=0.01)
    reasoner.fit(select(evidence, a), y[a], select(evidence, b), y[b])
    evidence_val = reasoner.predict_proba(select(evidence, dev))
    evidence_threshold = threshold_on_dev(evidence_val, y[dev])
    evidence_test = reasoner.predict_proba(select(evidence, test))
    contributions = reasoner.contributions(select(evidence, test))
    reference_path = out / "reference.json.gz"
    reference.save(reference_path)
    evidence_report = {"model": "computable_evidence_sign_constrained_additive",
                       "manifest": {**manifest,
                                    "reference_population": "official_train_mixture_by_domain",
                                    "reference_used_labels": False,
                                    "reference_version": reference.version,
                                    "reference_artifact_sha256": sha256(reference_path),
                                    "direction_split_size": len(a),
                                    "coefficient_split_size": len(b),
                                    "threshold": evidence_threshold,
                                    "source_certificate_status": "fact_table_only"},
                       "reasoner": reasoner.manifest(),
                       "dev": metric(y[dev], evidence_val, evidence_threshold),
                       "test": metric(y[test], evidence_test, evidence_threshold)}
    (out / "evidence_report.json").write_text(json.dumps(evidence_report, indent=2) + "\n",
                                               encoding="utf-8")
    (out / "test_predictions.json").write_text(json.dumps({
        "manifest": evidence_report["manifest"],
        "predictions": [{"account_id": ids[int(i)], "label": int(y[i]),
                         "probability": float(evidence_test[k]),
                         "group_contributions": {name: float(value[k])
                                                 for name, value in contributions.items()}}
                        for k, i in enumerate(test)]}, indent=2) + "\n", encoding="utf-8")
    return {"seed": seed, "tree": tree_report["test"],
            "evidence_candidate": evidence_report["test"], "output": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    args = parser.parse_args()
    results = [run(args, seed) for seed in args.seeds]
    print(json.dumps({"runs": results}, indent=2))


if __name__ == "__main__":
    main()
