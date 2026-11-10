"""Development-only evidence capacity gate for Fox8 and BotSim schemas."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.evidence_schema_v1 import SCHEMA, validate_schema
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference, SignConstrainedReasoner


ROOT = Path(__file__).resolve().parents[2]
SAHG_DATA = Path("D:/Code/SAHG/SAHG/code/data")
RAW_BOTSIM = ROOT / "data/raw/botsim_24/user_post_comment.json"


def _source_facts(dataset: str, seed: int):
    data_dir = SAHG_DATA / ("fox8_23" if dataset == "fox8" else "botsim_24")
    cache = (ROOT / "data/processed/fox8_v3/native_interaction_facts.npz"
             if dataset == "fox8" else
             ROOT / "data/processed/botsim_v3/native_graph_facts.npz")
    account_file = None if dataset == "fox8" else ROOT / "data/raw/botsim_24/Users.csv"
    x, y, ids, names, split, manifest = load(
        dataset, data_dir, account_file, seed,
        native_graph_facts=cache if dataset == "botsim" else None)
    if dataset == "fox8":
        native_manifest = json.loads(cache.with_suffix(".json").read_text(
            encoding="utf-8"))
        if (sha256_file(cache) != native_manifest["facts_sha256"] or
            sha256_file(ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz") !=
                native_manifest["raw_source_sha256"]):
            raise ValueError("Fox8 native facts differ from raw source")
        with np.load(cache, allow_pickle=False) as saved:
            if saved["ids"].astype(str).tolist() != ids:
                raise ValueError("Fox8 native fact rows differ")
            added_names = saved["names"].astype(str).tolist()
            added = saved["values"].astype(float)
        x = np.column_stack((x, added))
        names += added_names
    else:
        source_manifest = json.loads((ROOT / "data/processed/botsim_v3/manifest.json")
                                     .read_text(encoding="utf-8"))
        if source_manifest["input_sha256"].get(str(RAW_BOTSIM.resolve())) != \
                sha256_file(RAW_BOTSIM):
            raise ValueError("BotSim activity source differs from graph fact manifest")
    if x.shape != (len(ids), len(names)) or len(set(names)) != len(names):
        raise ValueError("source fact matrix differs")
    return x.astype(float), y, ids, names, split, manifest


def _lgbm(seed: int, *, stump: bool,
          monotone: list[int] | None = None) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=200, learning_rate=.05,
        num_leaves=2 if stump else 15, max_depth=1 if stump else -1,
        min_child_samples=20, reg_lambda=1.0, n_jobs=4,
        random_state=seed, deterministic=True, force_col_wise=True,
        monotone_constraints=monotone, verbosity=-1)


def _contexts(training: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, list[float]]:
    cuts = np.quantile(training[np.isfinite(training)], [1 / 3, 2 / 3]).astype(float)
    labels = np.asarray([f"activity_{i}" for i in
                         np.searchsorted(cuts, values, side="right")])
    return labels, cuts.tolist()


def _missing_means(dataset: str, x: np.ndarray, names: list[str], ids: list[str]
                   ) -> np.ndarray:
    fixed = x.copy()
    if dataset == "fox8":
        no_tweets = fixed[:, names.index("n_tweets")] == 0
        for name in ("retweet_ratio", "reply_ratio", "avg_tweet_len",
                     "avg_favorites", "avg_retweets", "avg_score",
                     "unique_lang_ratio"):
            fixed[no_tweets, names.index(name)] = np.nan
    else:
        activity = json.loads(RAW_BOTSIM.read_text(encoding="utf-8"))
        if set(activity) != set(ids):
            raise ValueError("BotSim raw activity accounts differ")
        no_posts = np.asarray([not activity[uid].get("posts") for uid in ids])
        no_comment1 = np.asarray([not activity[uid].get("comment_1") for uid in ids])
        if np.any(no_posts != (fixed[:, names.index("n_posts")] == 0)):
            raise ValueError("BotSim post denominator differs from feature table")
        fixed[no_posts, names.index("avg_post_len")] = np.nan
        fixed[no_comment1, names.index("avg_comment_len")] = np.nan
    return fixed


def _run(dataset: str) -> dict:
    specification = SCHEMA[dataset]
    selected_names = [row["fact"] for row in specification["concepts"]]
    groups = {}
    for row in specification["concepts"]:
        groups.setdefault(row["group"], []).append(row["fact"])
    runs = []
    for seed in (0, 1, 2):
        source, y, ids, names, split, manifest = _source_facts(dataset, seed)
        source = _missing_means(dataset, source, names, ids)
        if set(selected_names) - set(names):
            raise ValueError("schema concept absent from source")
        facts = source[:, [names.index(name) for name in selected_names]]
        train, dev = split["train"], split["val"]
        a, b = train_test_split(train, test_size=.5, stratify=y[train],
                                random_state=1000 + seed)
        context_column = selected_names.index(specification["reference_context"]["fact"])
        context = np.full(len(y), "", dtype="<U16")
        context[train], cuts = _contexts(facts[train, context_column],
                                         facts[train, context_column])
        context[dev], _ = _contexts(facts[train, context_column],
                                    facts[dev, context_column])
        reference = LabelBlindReference(min_context_n=30).fit(
            facts[train], selected_names, context[train])
        ev_a = reference.transform(facts[a], context[a])
        ev_b = reference.transform(facts[b], context[b])
        ev_dev = reference.transform(facts[dev], context[dev])
        reasoner = SignConstrainedReasoner(groups, l2=.01).fit(
            ev_a, y[a], ev_b, y[b])

        def tail(batch):
            values = np.column_stack([
                batch.upper[:, j] if choice == "upper" else batch.lower[:, j]
                for j, choice in enumerate(reasoner.tail)])
            return np.log1p(values) * batch.reliability

        evidence_gam = _lgbm(seed, stump=True,
                             monotone=reasoner.sign.astype(int).tolist())
        evidence_gam.fit(tail(ev_b), y[b])
        controls = {}
        control_split_counts = {}
        for label, matrix, stump in (
            ("same_named_raw_fact_gam_stumps", facts, True),
            ("same_named_raw_fact_blackbox", facts, False),
            ("all_available_source_fact_blackbox", source, False),
        ):
            model = _lgbm(seed, stump=stump)
            model.fit(matrix[train], y[train])
            controls[label] = evaluate(y[dev],
                model.predict_proba(matrix[dev])[:, 1], .5)
            feature_names = selected_names if matrix.shape[1] == len(selected_names) else names
            control_split_counts[label] = dict(zip(
                feature_names, map(int, model.feature_importances_), strict=True))
        runs.append({
            "seed": seed, "split_sha256": manifest["split_sha256"],
            "train_accounts": len(train), "direction_A_accounts": len(a),
            "coefficient_B_accounts": len(b), "dev_accounts": len(dev),
            "reference_version": reference.version,
            "context_train_quantiles": cuts,
            "evidence_linear_additive": evaluate(y[dev],
                reasoner.predict_proba(ev_dev), .5),
            "evidence_monotone_gam_stumps": evaluate(y[dev],
                evidence_gam.predict_proba(tail(ev_dev))[:, 1], .5),
            "evidence_gam_split_counts": dict(zip(
                selected_names, map(int, evidence_gam.feature_importances_), strict=True)),
            "controls": controls,
            "control_split_counts": control_split_counts,
        })
        print({"dataset": dataset, "seed": seed,
               "evidence_f1": runs[-1]["evidence_monotone_gam_stumps"]["macro_f1"],
               "raw_gam_f1": controls["same_named_raw_fact_gam_stumps"]["macro_f1"],
               "full_source_f1": controls["all_available_source_fact_blackbox"]["macro_f1"]},
              flush=True)
    return {"dataset": dataset,
            "scope": "predeclared_named_computed_evidence_capacity_gate_dev_only",
            "schema_sha256": sha256_file(ROOT / "research/evidence_schema_v1.json"),
            "test_predictions_or_metrics_accessed": False,
            "semantic_truth_independently_annotated": False,
            "training_reference_uses_no_labels": True,
            "A_direction_B_fit_disjoint": True,
            "controls_fit_on_all_training_labels": True,
            "classification_threshold": .5,
            "prior_fixed_test_population_explored_elsewhere": True,
            "runs": runs}


def main() -> None:
    counts = validate_schema()
    for dataset in ("fox8", "botsim"):
        if not 10 <= counts[dataset] <= 20:
            raise ValueError("schema concept count differs")
        report = _run(dataset)
        path = ROOT / f"output/gate1_{dataset}_schema{counts[dataset]}/dev_only.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
