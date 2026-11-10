"""Screen every frozen Fox8 source fact as a one-field train/dev classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from verifiable_bot.fox8_neighbor_atomic_runtime import Fox8NeighborAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/fox8_single_atomic_fact_train_dev.json"
DEPTHS = (1, 2, 3)
MIN_LEAF = 30


def main() -> None:
    args = SimpleNamespace(
        dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None, structural_facts=None,
        neighbor_facts=None,
        fox8_native_facts=ROOT / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=ROOT / "output/v3_fox8_native_spline_l2/selected")
    runs = []
    for seed in (0, 1, 2):
        runtime = Fox8NeighborAtomicRuntime(ROOT, seed)
        report_path = args.selected_model_dir / f"seed{seed}/report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        original, labels, original_names, split = selected_facts(args, seed, report)
        if (len(original_names) != 43 or len(runtime.reference.names) != 74 or
                not np.array_equal(runtime.facts[:, :43], original, equal_nan=True)):
            raise ValueError("frozen Fox8 atomic fact population differs")
        train, dev = split["train"], split["val"]
        scored = []
        for j, name in enumerate(runtime.reference.names):
            values = runtime.facts[:, j].astype(float)
            if np.isinf(values).any():
                raise ValueError("Fox8 source fact contains infinity")
            # Keep undefined observations distinct from all finite source values.
            finite = values[np.isfinite(values)]
            fill = (float(finite.min()) - max(1.0, abs(float(finite.min())) * .01)
                    if len(finite) else -1.0)
            column = np.where(np.isfinite(values), values, fill).reshape(-1, 1)
            fits = []
            for depth in DEPTHS:
                tree = DecisionTreeClassifier(
                    max_depth=depth, min_samples_leaf=MIN_LEAF,
                    random_state=0)
                tree.fit(column[train], labels[train])
                probability = tree.predict_proba(column[dev])[:, 1]
                fits.append({
                    "max_depth": depth,
                    "actual_depth": tree.get_depth(),
                    "validation": evaluate(labels[dev], probability, .5),
                })
            scored.append({
                "fact": name, "source_group": "native_neighbor" if j >= 43 else "own_or_native_event",
                "train_defined": int(np.isfinite(values[train]).sum()),
                "dev_defined": int(np.isfinite(values[dev]).sum()),
                "undefined_encoding": fill,
                "fits": fits,
            })
        best = sorted(({"fact": item["fact"], "source_group": item["source_group"],
                        "max_depth": fit["max_depth"],
                        **fit["validation"]}
                       for item in scored for fit in item["fits"]),
                      key=lambda row: (-row["acc"], -row["macro_f1"],
                                       row["max_depth"], row["fact"]))[:15]
        runs.append({
            "seed": seed, "split_sha256": report["manifest"]["split_sha256"],
            "train_accounts": len(train), "dev_accounts": len(dev),
            "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
            "top_single_fact_fits": best, "all_facts": scored,
        })
        print({"seed": seed, "best": best[0]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "fox8_frozen_74_source_facts_single_field_train_dev_screen",
        "test_predictions_or_metrics_used": False,
        "depths": DEPTHS, "min_samples_leaf": MIN_LEAF,
        "source_audit_sha256": sha256_file(ROOT /
            "research/certificates/fox8_neighbor_volume_full_raw_source.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
