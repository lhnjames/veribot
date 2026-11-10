"""Dev-only source-depth capacity screen on MGTAB's typed graph facts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    selected_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    overlap_path = root / "data/processed/mgtab_v3/relation_overlap_facts.npz"
    audit_path = root / "research/certificates/mgtab_relation_overlap_full_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (audit["status"] != "all_relation_overlap_fact_cells_replayed"
            or audit["facts_sha256"] != sha256_file(overlap_path)
            or audit["max_absolute_difference"] != 0):
        raise ValueError("typed-edge relation overlap audit differs")
    with np.load(overlap_path, allow_pickle=False) as saved:
        overlap = saved["values"].astype(float)
        overlap_ids = saved["ids"].astype(str)
        overlap_names = saved["names"].astype(str).tolist()
    config = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        full = json.loads((full_dir / f"seed{seed}" / "report.json")
                          .read_text(encoding="utf-8"))
        selected = json.loads((selected_dir / f"seed{seed}" / "report.json")
                              .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(config, seed, full)
        if (len(names) != 150 or len(x) != len(overlap_ids)
                or not np.array_equal(overlap_ids, np.arange(len(x)).astype(str))
                or names[20].startswith("published_followers") is False
                or selected["manifest"]["split_sha256"] !=
                full["manifest"]["split_sha256"]):
            raise ValueError("source population or selected split differs")
        tr, dev = split["train"], split["val"]
        inputs = {
            "typed_graph_66": x[:, 20:86],
            "typed_graph_66_plus_overlap_4": np.column_stack((x[:, 20:86], overlap)),
            "published_profile_20_plus_graph_66": x[:, :86],
        }
        alternatives = {}
        for name, matrix in inputs.items():
            model = HistGradientBoostingClassifier(
                max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
                l2_regularization=10.0, learning_rate=0.05,
                random_state=0)
            model.fit(matrix[tr], y[tr])
            probability = model.predict_proba(matrix[dev])[:, 1]
            threshold = best_validation_threshold(probability, y[dev])
            alternatives[name] = {
                "validation": evaluate(y[dev], probability, threshold),
                "threshold": threshold,
            }
        runs.append({"seed": seed, "split_sha256": full["manifest"]["split_sha256"],
                     "current_20_evidence_dev": selected["validation"],
                     "alternatives": alternatives})
        print(seed, {name: round(value["validation"]["acc"], 4)
                     for name, value in alternatives.items()}, flush=True)
    output = {
        "dataset": "mgtab", "scope": "official_train_dev_only_typed_graph_capacity",
        "test_predictions_accessed": False,
        "same_shallow_tree_and_threshold_rule_for_all_alternatives": True,
        "typed_graph_fact_names": names[20:86],
        "additional_overlap_fact_names": overlap_names,
        "overlap_audit_sha256": sha256_file(audit_path),
        "overlap_fact_sha256": sha256_file(overlap_path),
        "runs": runs,
    }
    target = root / "output/v3_mgtab_relation_only_capacity/dev_only.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
