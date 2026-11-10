"""Dev-only same-model capacity screen of audited named two-hop graph facts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_leaf_nodes=15, max_iter=150, min_samples_leaf=30,
        l2_regularization=10.0, learning_rate=0.05, random_state=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    original_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    selected_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    second_path = root / "data/processed/mgtab_v3/second_order_facts.npz"
    second_audit_path = root / "research/certificates/mgtab_second_order_full_source_audit.json"
    prior_path = root / "output/v3_mgtab_named_raw_capacity/dev_only.json"
    source_audit = read(second_audit_path)
    prior = read(prior_path)
    if (source_audit["status"] != "verified_all_second_order_fact_cells" or
            source_audit["fact_cache_sha256"] != sha256_file(second_path) or
            prior["dataset"] != "mgtab" or
            prior["source_features_sha256"] != sha256_file(data_dir / "features.pt")):
        raise ValueError("audited sources or named control differ")
    with np.load(second_path, allow_pickle=False) as cache:
        ids = cache["ids"].astype(int)
        names = cache["names"].astype(str).tolist()
        values = cache["values"].astype(float)
        supports = cache["supports"].astype(float)
    if (not np.array_equal(ids, np.arange(10199)) or
            values.shape != (10199, 24) or supports.shape != values.shape or
            len(names) != 24 or not np.array_equal(np.isnan(values), supports == 0)):
        raise ValueError("two-hop graph fact population differs")
    cfg = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=original_dir)
    runs = []
    for seed in (0, 1, 2):
        original = read(original_dir / f"seed{seed}" / "report.json")
        selected = read(selected_dir / f"seed{seed}" / "report.json")
        x, y, all_names, split = selected_facts(cfg, seed, original)
        if (x.shape != (10199, 150) or len(all_names) != 150 or
                selected["manifest"]["split_sha256"] !=
                original["manifest"]["split_sha256"] or
                all_names[:86] != prior["fact_names"]):
            raise ValueError("fixed population or named feature order differs")
        train, dev = split["train"], split["val"]
        if prior["runs"][seed]["split_sha256"] != original["manifest"]["split_sha256"]:
            raise ValueError("named control split differs")
        alternatives = {}
        for label, matrix in (
                ("published_86", x[:, :86]),
                ("published_86_plus_two_hop_24", np.column_stack((x[:, :86], values)))):
            classifier = model()
            classifier.fit(matrix[train], y[train])
            p = classifier.predict_proba(matrix[dev])[:, 1]
            threshold = best_validation_threshold(p, y[dev])
            alternatives[label] = {
                "validation": evaluate(y[dev], p, threshold),
                "threshold": threshold,
            }
        prior_control = prior["runs"][seed]["capacity_controls"][
            "shallow_hist_gradient_boosting"]["validation"]
        control = alternatives["published_86"]["validation"]
        if any(abs(control[key] - prior_control[key]) > 1e-10
               for key in ("acc", "macro_f1", "auroc")):
            raise ValueError("same-model named control did not replay")
        runs.append({
            "seed": seed, "split_sha256": original["manifest"]["split_sha256"],
            "alternatives": alternatives,
        })
        print(seed, {key: value["validation"]["acc"]
                     for key, value in alternatives.items()}, flush=True)
    output = {
        "dataset": "mgtab", "scope": "fixed_train_dev_named_two_hop_capacity",
        "test_labels_used_for_fit_or_selection": False,
        "test_labels_loaded_by_full_dataset_adapter": True,
        "second_order_facts_sha256": sha256_file(second_path),
        "second_order_source_audit_sha256": sha256_file(second_audit_path),
        "named_control_sha256": sha256_file(prior_path),
        "second_order_fact_names": names, "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
