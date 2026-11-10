"""Freeze each Fox8 seed's best train/dev single-fact tree before test."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from audit_fox8_single_atomic_fact_train_dev import MIN_LEAF, OUT as SCREEN, ROOT
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


OUT = ROOT / "output/v3_fox8_single_fact_dev_locked"


def population(seed: int):
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None, structural_facts=None,
        neighbor_facts=None,
        fox8_native_facts=ROOT / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=ROOT / "output/v3_fox8_native_spline_l2/selected")
    runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
    report = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                        .read_text(encoding="utf-8"))
    first, labels, first_names, split = selected_facts(args, seed, report)
    if (len(first_names) != 43 or len(runtime.reference.names) != 74 or
            not np.array_equal(runtime.facts[:, :43], first, equal_nan=True)):
        raise ValueError("Fox8 source fact population differs")
    return runtime, labels, split, report


def column(runtime, name: str, fill: float) -> np.ndarray:
    j = runtime.reference.names.index(name)
    values = runtime.facts[:, j].astype(float)
    return np.where(np.isfinite(values), values, fill).reshape(-1, 1)


def main() -> None:
    screen = json.loads(SCREEN.read_text(encoding="utf-8"))
    if screen["test_predictions_or_metrics_used"] is not False:
        raise ValueError("Single-fact selection used test data")
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in (0, 1, 2):
        runtime, y, split, report = population(seed)
        chosen = screen["runs"][seed]["top_single_fact_fits"][0]
        name = chosen["fact"]
        fact_record = next(x for x in screen["runs"][seed]["all_facts"]
                           if x["fact"] == name)
        fill = float(fact_record["undefined_encoding"])
        x = column(runtime, name, fill)
        train, dev = split["train"], split["val"]
        tree = DecisionTreeClassifier(
            max_depth=int(chosen["max_depth"]), min_samples_leaf=MIN_LEAF,
            random_state=0)
        tree.fit(x[train], y[train])
        p_dev = tree.predict_proba(x[dev])[:, 1]
        metrics = evaluate(y[dev], p_dev, .5)
        for key in ("n", "acc", "macro_f1", "auroc", "errors"):
            if abs(metrics[key] - chosen[key]) > 1e-12:
                raise ValueError("Chosen train/dev single-fact fit differs")
        if report["manifest"]["split_sha256"] != screen["runs"][seed]["split_sha256"]:
            raise ValueError("Official Fox8 split changed")
        folder = OUT / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        model_path = folder / "tree.joblib"
        joblib.dump(tree, model_path)
        runs.append({"seed": seed, "fact": name,
                     "source_group": chosen["source_group"],
                     "max_depth": int(chosen["max_depth"]),
                     "min_samples_leaf": MIN_LEAF,
                     "undefined_encoding": fill,
                     "split_sha256": report["manifest"]["split_sha256"],
                     "validation": metrics, "tree_path": str(model_path),
                     "tree_sha256": sha256_file(model_path)})
        print({"seed": seed, "fact": name,
               "dev_errors": metrics["errors"]}, flush=True)
    (OUT / "dev_lock.json").write_text(json.dumps({
        "scope": "Fox8_best_single_atomic_fact_selected_on_dev_before_test",
        "dev_screen_sha256": sha256_file(SCREEN),
        "test_predictions_or_metrics_used_for_selection": False,
        "runs": runs}, indent=2) + "\n", encoding="utf-8")
    print(OUT / "dev_lock.json")


if __name__ == "__main__":
    main()
