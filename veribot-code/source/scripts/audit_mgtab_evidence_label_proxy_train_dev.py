"""Train/dev-only label association screen for frozen MGTAB evidence inputs.

Association is a diagnostic, not proof of label leakage or semantic validity.
The dataset loader reads all labels to reconstruct the official stratified split;
test labels never enter this screen's MI, ranking, or permutation statistics.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate0_shallow_tree import load
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/certificates/mgtab_evidence_label_proxy_train_dev.json"
N_FACTS = 150
QUANTILES = (0.2, 0.4, 0.6, 0.8)
PERMUTATIONS = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_evidence(runtime: MGTABFullAtomicRuntime,
                   rows: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(runtime.facts[rows])
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= (runtime.supports[rows] /
                              (runtime.supports[rows] + 5.0))
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def joint_states(train: np.ndarray, dev: np.ndarray,
                 j: int) -> tuple[np.ndarray, np.ndarray]:
    states = []
    for column in (j, j + N_FACTS):
        thresholds = np.quantile(train[:, column], QUANTILES)
        states.append((np.searchsorted(thresholds, train[:, column],
                                       side="right"),
                       np.searchsorted(thresholds, dev[:, column],
                                       side="right")))
    return states[0][0] * 5 + states[1][0], states[0][1] * 5 + states[1][1]


def normalized_mi(state: np.ndarray, labels: np.ndarray) -> float:
    label_entropy = mutual_info_score(labels, labels)
    if label_entropy <= 0:
        raise ValueError("label split has no entropy")
    return float(mutual_info_score(state, labels) / label_entropy)


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                         variant="ab_group_additive")
        _, labels, ids, _, split, source = load(
            "mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or
                source["split_sha256"] !=
                runtime.lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("MGTAB published rows or split differ")
        train, dev = split["train"], split["val"]
        evidence_train = model_evidence(runtime, train)
        evidence_dev = model_evidence(runtime, dev)
        if (evidence_train.shape != (len(train), 300) or
                evidence_dev.shape != (len(dev), 300) or
                not np.isfinite(evidence_train).all() or
                not np.isfinite(evidence_dev).all()):
            raise ValueError("frozen evidence design differs")
        rng = np.random.default_rng(20261103 + seed)
        permutations = [rng.permutation(labels[dev])
                        for _ in range(PERMUTATIONS)]
        facts = []
        dev_states = []
        for j, name in enumerate(runtime.names):
            state_train, state_dev = joint_states(
                evidence_train, evidence_dev, j)
            dev_states.append(state_dev)
            train_mi = normalized_mi(state_train, labels[train])
            dev_mi = normalized_mi(state_dev, labels[dev])
            facts.append({
                "index": j, "name": name,
                "source_group": next(group for group, members in
                                     runtime.group_fact_indices.items()
                                     if j in members),
                "train_normalized_mi": train_mi,
                "dev_normalized_mi": dev_mi,
                "dev_near_label_alias_screen": bool(dev_mi >= .8),
                "train_states": int(len(np.unique(state_train))),
                "dev_states": int(len(np.unique(state_dev))),
            })
        ranked = sorted(facts, key=lambda f: f["train_normalized_mi"],
                        reverse=True)
        top = ranked[:20]
        for fact in top:
            null_mi = np.asarray([normalized_mi(
                dev_states[fact["index"]], p) for p in permutations])
            fact["dev_permutation_null_median"] = float(np.median(null_mi))
            fact["dev_permutation_null_95pct"] = float(np.quantile(null_mi, .95))
            fact["dev_exceeds_permutation_95pct"] = bool(
                fact["dev_normalized_mi"] > np.quantile(null_mi, .95))
        runs.append({
            "seed": seed, "split_sha256": source["split_sha256"],
            "reference_version": runtime.reference.version,
            "model_sha256": sha256(runtime.run_dir / "model.joblib"),
            "train_n": len(train), "dev_n": len(dev),
            "train_bot_fraction": float(labels[train].mean()),
            "dev_bot_fraction": float(labels[dev].mean()),
            "near_label_alias_count_all": sum(
                f["dev_near_label_alias_screen"] for f in facts),
            "near_label_alias_count_train_top20": sum(
                f["dev_near_label_alias_screen"] for f in top),
            "train_selected_top20_dev_exceeds_null_count": sum(
                f["dev_exceeds_permutation_95pct"] for f in top),
            "top20_train_ranked": top,
            "all_facts": facts,
        })
        print({"seed": seed, "near_label_aliases":
               runs[-1]["near_label_alias_count_all"],
               "top_train_fact": top[0]["name"],
               "top_train_mi": top[0]["train_normalized_mi"],
               "top_dev_mi": top[0]["dev_normalized_mi"]}, flush=True)
    result = {
        "status": "train_dev_association_screen_only",
        "scope": "frozen MGTAB 150-fact A/B group additive high-performance model",
        "method": "Each of the two evidence tails is discretized into five training-quantile bins; their joint 25-state variable is compared with Bot/Human by mutual information normalized by label entropy. The same training thresholds score official validation accounts. The permutation null shuffles validation labels 200 times for the top 20 facts selected only by training association.",
        "interpretation": "High association can reflect useful evidence, a dataset artifact, or a label proxy. This screen cannot establish leakage, external semantic truth, or causal effect on the classifier; source and decision audits address separate questions.",
        "test_labels_used_in_mi_ranking_or_permutation": False,
        "split_reconstruction_reads_dataset_label_tensor": True,
        "test_prediction_files_read": False,
        "runs": runs,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
