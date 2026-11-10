"""Train-only top-20 named fact bottleneck, evaluated on official MGTAB dev."""

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
    (15, 200, 50, 10.0),
    (15, 400, 50, 10.0),
    (31, 200, 50, 10.0),
    (31, 400, 50, 10.0),
)


def matrix(batch) -> np.ndarray:
    return np.column_stack((np.log1p(batch.upper) * batch.reliability,
                            np.log1p(batch.lower) * batch.reliability))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parent = root / "output/v3_mgtab_named_evidence_lightgbm"
    lock_path = parent / "selection_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    summary = json.loads((parent / "selected/paired_summary.json")
                         .read_text(encoding="utf-8"))
    if (summary["selection_lock_sha256"] != sha256_file(lock_path)
            or lock["selected_config"] != [31, 400, 50, 10.0]):
        raise ValueError("full 86-fact train-only importance source differs")
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    structural = root / "data/processed/mgtab_v3/structural_facts.npz"
    neighbor = root / "data/processed/mgtab_v3/neighbor_facts.npz"
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None, structural_facts=structural,
        neighbor_facts=neighbor,
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    runs = []
    for seed in (0, 1, 2):
        run_dir = parent / "selected" / f"seed{seed}"
        full_model_path = run_dir / "model.txt"
        full_reference_path = run_dir / "reference.json.gz"
        if (summary["runs"][seed]["model_sha256"] !=
                sha256_file(full_model_path)
                or summary["runs"][seed]["reference_sha256"] !=
                sha256_file(full_reference_path)):
            raise ValueError("full 86-fact train-only model differs")
        full_model = lightgbm.Booster(model_file=str(full_model_path))
        importance = full_model.feature_importance(importance_type="gain")
        if len(importance) != 172:
            raise ValueError("full model evidence dimension differs")
        fact_gain = importance[:86] + importance[86:]
        selected = np.argsort(-fact_gain, kind="stable")[:20]
        selected.sort()
        if len(set(selected)) != 20 or np.any(fact_gain[selected] <= 0):
            raise ValueError("top-20 fact selection differs")
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        if (len(names) != 150 or old["manifest"]["split_sha256"] !=
                lock["split_sha256_by_seed"][str(seed)]):
            raise ValueError("official split or named source vocabulary differs")
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference.load(full_reference_path)
        if (reference.version != lock["reference_version_by_seed"][str(seed)]
                or list(reference.names) != names[:86]):
            raise ValueError("label-blind reference differs")
        columns = np.r_[selected, selected + 86]
        full_train = matrix(transform_with_support(
            reference, x[train, :86], split["supports"][train]))
        full_dev = matrix(transform_with_support(
            reference, x[dev, :86], split["supports"][dev]))
        fit_x = full_train[:, columns]
        eval_x = full_dev[:, columns]
        alternatives = []
        for leaves, iterations, min_leaf, l2 in CONFIGS:
            model = LGBMClassifier(
                objective="binary", n_estimators=iterations, num_leaves=leaves,
                min_child_samples=min_leaf, reg_lambda=l2,
                learning_rate=0.05, n_jobs=4, random_state=0,
                deterministic=True, force_col_wise=True, verbosity=-1)
            model.fit(fit_x, y[train])
            probability = model.predict_proba(eval_x)[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            probe = eval_x[:32]
            raw = np.asarray(model.booster_.predict(probe, raw_score=True))
            contribution = np.asarray(model.booster_.predict(probe,
                                                             pred_contrib=True))
            max_logit_error = float(np.max(np.abs(contribution.sum(axis=1) - raw)))
            if contribution.shape != (len(probe), 41) or max_logit_error > 1e-9:
                raise ValueError("20-fact evidence logit contribution differs")
            alternatives.append({
                "num_leaves": leaves, "n_estimators": iterations,
                "min_child_samples": min_leaf, "reg_lambda": l2,
                "threshold": threshold,
                "validation": evaluate(y[dev], probability, threshold),
                "max_probe_logit_additivity_error": max_logit_error,
            })
        runs.append({
            "seed": seed,
            "split_sha256": lock["split_sha256_by_seed"][str(seed)],
            "full_model_sha256": sha256_file(full_model_path),
            "reference_version": reference.version,
            "selected_fact_indices": selected.tolist(),
            "selected_fact_names": [names[index] for index in selected],
            "profile_fact_count": int(np.count_nonzero(selected < 20)),
            "native_relation_fact_count": int(np.count_nonzero(selected >= 20)),
            "full_86_fact_validation": summary["runs"][seed]["validation"],
            "alternatives": alternatives,
        })
        print(seed, runs[-1]["profile_fact_count"],
              [round(item["validation"]["macro_f1"] * 100, 2)
               for item in alternatives], flush=True)
    output = {
        "dataset": "mgtab",
        "scope": "official_train_dev_only_top20_named_release_source_evidence_lightgbm",
        "test_labels_accessed_for_selection_or_fit": False,
        "published_content_projection_used": False,
        "fact_selection": "per-seed top 20 aggregated upper/lower training gain from the already trained 86-fact model; no other seed's importance is used",
        "source_features_sha256": lock["source_features_sha256"],
        "structural_facts_sha256": lock["structural_facts_sha256"],
        "neighbor_facts_sha256": lock["neighbor_facts_sha256"],
        "full_86_selection_lock_sha256": sha256_file(lock_path),
        "configs": CONFIGS,
        "runs": runs,
    }
    destination = root / "output/v3_mgtab_named_top20_lightgbm/dev_only.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
