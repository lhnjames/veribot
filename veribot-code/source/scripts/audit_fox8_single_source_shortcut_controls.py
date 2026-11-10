"""Fixed single-source controls for the Fox8 high-performance classifier.

The neighbor-language field was identified in an earlier dev screen. This is
a diagnostic on the same train/dev populations, not an independent validation
or a new model selection procedure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/fox8_single_source_shortcut_controls.json"
FIELDS = (
    "native_interaction_neighbor_mean_unique_lang_ratio",
    "unique_lang_ratio",
    "native_interaction_neighbor_mean_statuses_count",
    "native_interaction_neighbor_mean_reply_ratio",
    "n_tweets",
    "verified",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate(values: np.ndarray, labels: np.ndarray,
             train: np.ndarray, dev: np.ndarray) -> dict:
    finite_train = values[train][np.isfinite(values[train])]
    fill = (float(finite_train.min()) - max(1., abs(float(finite_train.min()))*.01)
            if len(finite_train) else -1.)
    feature = np.where(np.isfinite(values), values, fill).reshape(-1, 1)
    tree = DecisionTreeClassifier(max_depth=1, min_samples_leaf=30,
                                  random_state=0)
    tree.fit(feature[train], labels[train])
    prediction = tree.predict(feature[dev]).astype(int)
    actual = labels[dev].astype(int)
    threshold = (float(tree.tree_.threshold[0])
                 if tree.tree_.feature[0] >= 0 else None)
    strata = {}
    for name, mask in (
            ("defined", np.isfinite(values[dev])),
            ("undefined", ~np.isfinite(values[dev])),
            ("below_or_equal_threshold", feature[dev, 0] <= threshold
             if threshold is not None else np.ones(len(dev), dtype=bool)),
            ("above_threshold", feature[dev, 0] > threshold
             if threshold is not None else np.zeros(len(dev), dtype=bool))):
        strata[name] = {
            "accounts": int(mask.sum()),
            "human": int(np.count_nonzero(mask & (actual == 0))),
            "bot": int(np.count_nonzero(mask & (actual == 1))),
            "errors": int(np.count_nonzero(mask & (prediction != actual))),
        }
    return {
        "dev_accounts": len(dev),
        "dev_errors": int(np.count_nonzero(prediction != actual)),
        "dev_accuracy": float(np.mean(prediction == actual)),
        "dev_bot_to_human_errors": int(np.count_nonzero((actual == 1) &
                                                       (prediction == 0))),
        "dev_human_to_bot_errors": int(np.count_nonzero((actual == 0) &
                                                       (prediction == 1))),
        "train_defined": int(np.isfinite(values[train]).sum()),
        "dev_defined": int(np.isfinite(values[dev]).sum()),
        "split_threshold": threshold,
        "actual_depth": int(tree.get_depth()),
        "fill_for_undefined": fill,
        "dev_strata": strata,
    }


def run() -> dict:
    rows = []
    for seed in (0, 1, 2):
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        _, labels, ids, _, split, source = load(
            "fox8", runtime.data_dir, None, seed)
        if ids != runtime.ids or source["split_sha256"] != runtime.lock[
                "split_sha256_by_seed"][str(seed)]:
            raise ValueError("Fox8 account split differs from frozen model")
        train = np.asarray(split["train"], dtype=int)
        dev = np.asarray(split["val"], dtype=int)
        tests = {}
        for name in FIELDS:
            j = runtime.reference.names.index(name)
            tests[name] = evaluate(runtime.facts[:, j].astype(float),
                                   labels, train, dev)
        degree = np.asarray([len(neighbors) for neighbors in runtime.neighbors],
                            dtype=float)
        tests["native_neighbor_degree"] = evaluate(degree, labels, train, dev)
        neighbor_language = runtime.facts[:, runtime.reference.names.index(
            "native_interaction_neighbor_mean_unique_lang_ratio")]
        missing = np.isfinite(neighbor_language).astype(float)
        tests["neighbor_language_defined_indicator"] = evaluate(
            missing, labels, train, dev)
        rows.append({
            "seed": seed, "train_accounts": len(train),
            "dev_accounts": len(dev),
            "split_sha256": source["split_sha256"],
            "model_sha256": sha256(runtime.run_dir / "model.joblib"),
            "field_controls": tests,
        })
        print(seed, {name: result["dev_errors"] for name, result in tests.items()},
              flush=True)
    result = {
        "status": "fox8_fixed_single_source_train_dev_shortcut_controls",
        "scope": "same official Fox8 train/dev splits already used during model development",
        "selection_limit": "neighbor language field came from an earlier 74-field dev screen; these dev errors do not establish independent generalization",
        "fixed_test_labels_read": False,
        "runs": rows,
    }
    OUT.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    run()
