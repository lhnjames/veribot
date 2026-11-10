"""Screen each of the 150 frozen MGTAB source facts as a one-field classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import evaluate, load, sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/mgtab_single_atomic_fact_train_dev.json"
DEPTHS = (1, 2, 3)
MIN_LEAF = 50


def main() -> None:
    runs = []
    for seed in (0, 1, 2):
        runtime = MGTABFullAtomicRuntime(ROOT, seed, variant="ab_group_additive")
        _, labels, ids, _, split, source = load("mgtab", runtime.data_dir, None, seed)
        if (ids != runtime.ids or source["split_sha256"] !=
                runtime.lock["split_sha256_by_seed"][str(seed)] or
                runtime.facts.shape != (10199, 150)):
            raise ValueError("MGTAB source population or official split differs")
        train, dev = split["train"], split["val"]
        facts = []
        for j, name in enumerate(runtime.names):
            values = runtime.facts[:, j].astype(float)
            if np.isinf(values).any():
                raise ValueError("published fact contains infinity")
            finite = values[np.isfinite(values)]
            fill = (float(finite.min()) - max(1.0, abs(float(finite.min())) * .01)
                    if len(finite) else -1.0)
            column = np.where(np.isfinite(values), values, fill).reshape(-1, 1)
            fits = []
            for depth in DEPTHS:
                tree = DecisionTreeClassifier(max_depth=depth,
                                              min_samples_leaf=MIN_LEAF,
                                              random_state=0)
                tree.fit(column[train], labels[train])
                probability = tree.predict_proba(column[dev])[:, 1]
                fits.append({"max_depth": depth, "actual_depth": tree.get_depth(),
                             "validation": evaluate(labels[dev], probability, .5)})
            facts.append({
                "fact": name,
                "source_group": next(group for group, members in
                                     runtime.group_fact_indices.items() if j in members),
                "train_defined": int(np.isfinite(values[train]).sum()),
                "dev_defined": int(np.isfinite(values[dev]).sum()),
                "undefined_encoding": fill, "fits": fits,
            })
        best = sorted(({"fact": fact["fact"], "source_group": fact["source_group"],
                        "max_depth": fit["max_depth"], **fit["validation"]}
                       for fact in facts for fit in fact["fits"]),
                      key=lambda row: (-row["acc"], -row["macro_f1"],
                                       row["max_depth"], row["fact"]))[:20]
        runs.append({"seed": seed, "split_sha256": source["split_sha256"],
                     "train_accounts": len(train), "dev_accounts": len(dev),
                     "model_sha256": sha256_file(runtime.run_dir / "model.joblib"),
                     "top_single_fact_fits": best, "all_facts": facts})
        print({"seed": seed, "best": best[0]}, flush=True)
    OUT.write_text(json.dumps({
        "scope": "mgtab_frozen_150_source_facts_single_field_train_dev_screen",
        "test_predictions_or_metrics_used": False,
        "depths": DEPTHS, "min_samples_leaf": MIN_LEAF,
        "source_audit_sha256": sha256_file(ROOT /
            "research/certificates/mgtab_full_source_audit.json"),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
