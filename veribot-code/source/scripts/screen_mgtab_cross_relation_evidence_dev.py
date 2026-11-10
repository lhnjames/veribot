"""Dev-only MGTAB cross-relation graph evidence versus the 86-fact control."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import lightgbm
import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from verifiable_bot.mgtab_neighbor_facts import SUPPORT_PRIOR
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def evidence_matrix(reference: LabelBlindReference, x: np.ndarray,
                    neighbor_support: np.ndarray,
                    cross_support: np.ndarray | None) -> np.ndarray:
    result = reference.transform(x)
    reliability = result.reliability.copy()
    reliability[:, 62:86] *= neighbor_support / (neighbor_support + SUPPORT_PRIOR)
    if cross_support is not None:
        reliability[:, 86:] *= cross_support / (cross_support + SUPPORT_PRIOR)
    matrix = np.column_stack((np.log1p(result.upper) * reliability,
                              np.log1p(result.lower) * reliability))
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite source evidence value")
    return matrix


def model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=400, num_leaves=31,
        min_child_samples=50, reg_lambda=10.0,
        learning_rate=0.05, n_jobs=4, random_state=0,
        deterministic=True, force_col_wise=True, verbosity=-1)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    cross_path = root / "data/processed/mgtab_v3/cross_relation_facts.npz"
    source_audit_path = root / "research/certificates/mgtab_cross_relation_full_source_audit.json"
    prior_path = root / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
    audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (audit["status"] != "verified_all_cross_relation_fact_cells"
            or audit["fact_cache_sha256"] != sha256_file(cross_path)
            or prior["lightgbm_version"] != lightgbm.__version__
            or prior["configs"][3] != [31, 400, 50, 10.0]):
        raise ValueError("source audit or fixed capacity control differs")
    with np.load(cross_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(int)
        cross_names = cache["names"].astype(str).tolist()
        cross_values = cache["values"].astype(float)
        cross_support = cache["supports"].astype(float)
    if (not np.array_equal(ids, np.arange(10199))
            or cross_values.shape != (10199, 84)
            or cross_support.shape != cross_values.shape):
        raise ValueError("cross-relation fact population differs")
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        old_report = json.loads((full_dir / f"seed{seed}" / "report.json")
                                .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old_report)
        if (prior["runs"][seed]["split_sha256"] !=
                old_report["manifest"]["split_sha256"]
                or len(names) != 150 or len(cross_names) != 84):
            raise ValueError("official split or fact vocabulary differs")
        train, dev = split["train"], split["val"]
        results = {}
        for label, feature_values, feature_names, supports in (
                ("named_86", x[:, :86], names[:86], None),
                ("named_86_plus_cross_relation_84",
                 np.column_stack((x[:, :86], cross_values)),
                 names[:86] + cross_names, cross_support)):
            reference = LabelBlindReference(min_context_n=30).fit(
                feature_values[train], feature_names)
            x_train = evidence_matrix(
                reference, feature_values[train], split["supports"][train],
                None if supports is None else supports[train])
            x_dev = evidence_matrix(
                reference, feature_values[dev], split["supports"][dev],
                None if supports is None else supports[dev])
            classifier = model()
            classifier.fit(x_train, y[train])
            probability = classifier.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            metrics = evaluate(y[dev], probability, threshold)
            if label == "named_86":
                prior_result = prior["runs"][seed]["alternatives"][3]
                if (reference.version != prior["runs"][seed]["reference_version"]
                        or abs(threshold - prior_result["threshold"]) > 1e-12
                        or any(abs(metrics[key] - prior_result["validation"][key]) > 1e-12
                               for key in ("acc", "macro_f1", "auroc"))):
                    raise ValueError("prior 86-fact control did not replay")
            results[label] = {
                "reference_version": reference.version,
                "feature_count": feature_values.shape[1],
                "evidence_input_count": x_train.shape[1],
                "validation": metrics, "threshold": threshold,
            }
        runs.append({"seed": seed,
                     "split_sha256": old_report["manifest"]["split_sha256"],
                     "alternatives": results})
        print(seed, {key: (round(value["validation"]["acc"] * 100, 2),
                          round(value["validation"]["macro_f1"] * 100, 2))
                     for key, value in results.items()}, flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_cross_relation_source_evidence",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "published_content_projection_used": False,
        "lightgbm_version": lightgbm.__version__,
        "cross_relation_fact_sha256": sha256_file(cross_path),
        "cross_relation_source_audit_sha256": sha256_file(source_audit_path),
        "prior_named_evidence_lightgbm_sha256": sha256_file(prior_path),
        "runs": runs,
    }
    path = root / "output/v3_mgtab_cross_relation_evidence/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
