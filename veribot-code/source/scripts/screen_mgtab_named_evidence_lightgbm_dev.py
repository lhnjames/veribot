"""Dev-only LightGBM screen on named, source-verified MGTAB evidence features."""

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
from screen_mgtab_content_projection_evidence import transform_with_support
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


CONFIGS = (
    (15, 200, 30, 10.0),
    (15, 400, 30, 10.0),
    (31, 200, 30, 10.0),
    (31, 400, 50, 10.0),
)


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    prior_path = root / "output/v3_mgtab_named_evidence_boosting/dev_only.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (prior["source_features_sha256"] != sha256_file(data_dir / "features.pt")
            or prior["structural_facts_sha256"] != sha256_file(structural)
            or prior["neighbor_facts_sha256"] != sha256_file(neighbor)):
        raise ValueError("published source or verified facts differ")
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor, selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((full_dir / f"seed{seed}" / "report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        old = prior["runs"][seed]
        if (old["seed"] != seed or old["split_sha256"] !=
                report["manifest"]["split_sha256"] or len(names) != 150):
            raise ValueError("official split or named fact vocabulary differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(
            x[train, :86], names[:86])
        if reference.version != old["reference_version"]:
            raise ValueError("label-blind evidence reference differs")
        x_train = matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))
        x_dev = matrix(transform_with_support(
            reference, x[dev, :86], split["supports"][dev]))
        if (x_train.shape[1] != 172 or not np.isfinite(x_train).all()
                or not np.isfinite(x_dev).all()):
            raise ValueError("named evidence matrix differs")
        alternatives = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations, num_leaves=leaves,
                min_child_samples=min_leaf, reg_lambda=l2,
                learning_rate=0.05, n_jobs=4, random_state=0,
                deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(x_train, y[train])
            probability = model.predict_proba(x_dev)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            probe = x_dev[:32]
            raw = np.asarray(model.booster_.predict(probe, raw_score=True))
            contribution = np.asarray(model.booster_.predict(
                probe, pred_contrib=True))
            maximum_additivity_error = float(np.max(np.abs(
                contribution.sum(axis=1) - raw)))
            if contribution.shape != (len(probe), 173) or maximum_additivity_error > 1e-9:
                raise ValueError("feature contributions do not reconstruct raw score")
            alternatives.append({
                "num_leaves": leaves, "n_estimators": iterations,
                "min_child_samples": min_leaf, "reg_lambda": l2,
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
                "max_dev_probe_logit_additivity_error": maximum_additivity_error,
            })
        runs.append({
            "seed": seed, "split_sha256": old["split_sha256"],
            "reference_version": reference.version,
            "named_fact_count": 86, "evidence_input_count": 172,
            "prior_hist_booster_best_macro_f1": max(
                row["validation"]["macro_f1"] for row in old["alternatives"]),
            "alternatives": alternatives,
        })
        print(seed, [(round(row["validation"]["acc"] * 100, 2),
                      round(row["validation"]["macro_f1"] * 100, 2))
                     for row in alternatives], flush=True)
    output = {
        "scope": "mgtab_train_dev_only_named_source_evidence_lightgbm",
        "test_rows_accessed": False,
        "published_content_projection_used": False,
        "lightgbm_version": lightgbm.__version__,
        "source_features_sha256": prior["source_features_sha256"],
        "structural_facts_sha256": prior["structural_facts_sha256"],
        "neighbor_facts_sha256": prior["neighbor_facts_sha256"],
        "prior_named_evidence_boosting_sha256": sha256_file(prior_path),
        "configs": CONFIGS, "runs": runs,
    }
    path = root / "output/v3_mgtab_named_evidence_lightgbm/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
