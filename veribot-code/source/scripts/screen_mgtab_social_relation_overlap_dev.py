"""Dev-only social-relation subset of independently audited overlap facts."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_cross_relation_evidence_dev import evidence_matrix, model
from verifiable_bot.mgtab_source_certificate import RELATIONS
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB")
    source_path = root / "data/processed/mgtab_v3/cross_relation_facts.npz"
    audit_path = root / "research/certificates/mgtab_cross_relation_full_source_audit.json"
    prior_path = root / "output/v3_mgtab_cross_relation_evidence/dev_only.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    if (audit["fact_cache_sha256"] != sha256_file(source_path)
            or prior["cross_relation_source_audit_sha256"] != sha256_file(audit_path)):
        raise ValueError("cross-relation source certificate differs")
    with np.load(source_path, allow_pickle=False) as cache:
        all_names = cache["names"].astype(str).tolist()
        all_values = cache["values"].astype(float)
        all_support = cache["supports"].astype(float)
    expected = []
    for direction in ("out", "in"):
        for left, right in itertools.combinations(range(5), 2):
            base = f"published_{RELATIONS[left]}_{RELATIONS[right]}_{direction}"
            expected.extend((f"{base}_unique_overlap_log1p",
                             f"{base}_neighbor_jaccard"))
    indices = [all_names.index(name) for name in expected]
    if len(indices) != 40 or len(set(indices)) != 40:
        raise ValueError("prespecified social-relation facts differ")
    values = all_values[:, indices]
    supports = all_support[:, indices]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab", data_dir=data_dir, account_file=None,
        native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        report = json.loads((full_dir / f"seed{seed}" / "report.json")
                            .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, report)
        old = prior["runs"][seed]
        if old["split_sha256"] != report["manifest"]["split_sha256"]:
            raise ValueError("same official account split required")
        train, dev = split["train"], split["val"]
        features = np.column_stack((x[:, :86], values))
        reference = LabelBlindReference(min_context_n=30).fit(
            features[train], names[:86] + expected)
        x_train = evidence_matrix(reference, features[train],
                                  split["supports"][train], supports[train])
        x_dev = evidence_matrix(reference, features[dev],
                                split["supports"][dev], supports[dev])
        classifier = model()
        classifier.fit(x_train, y[train])
        p = classifier.predict_proba(x_dev)[:, 1]
        threshold = best_validation_threshold(p, y[dev])
        metrics = evaluate(y[dev], p, threshold)
        runs.append({
            "seed": seed, "split_sha256": old["split_sha256"],
            "reference_version": reference.version,
            "selected_fact_names": expected,
            "baseline_86_validation": old["alternatives"]["named_86"]["validation"],
            "social_40_validation": metrics,
            "threshold": threshold,
        })
        print(seed, round(metrics["acc"] * 100, 2),
              round(metrics["macro_f1"] * 100, 2), flush=True)
    output = {
        "scope": "mgtab_train_dev_only_prespecified_social_relation_overlap",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "social_relation_names": list(RELATIONS[:5]),
        "added_fact_count": len(indices),
        "source_fact_sha256": sha256_file(source_path),
        "source_audit_sha256": sha256_file(audit_path),
        "all_overlap_dev_sha256": sha256_file(prior_path),
        "runs": runs,
    }
    path = root / "output/v3_mgtab_social_relation_overlap/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
