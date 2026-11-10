"""Dev-only MGTAB A/B allocation screen under the original monotone contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


FRACTIONS = (0.25, 0.33, 0.40)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((full_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        train, dev = split["train"], split["val"]
        reference = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if reference.version != old["manifest"]["reference_version"]:
            raise ValueError("source/reference differs from original A/B model")
        evidence = transform_with_support(reference, x, split["supports"])
        original = MonotoneSplineReasoner.from_manifest(old["reasoner"])
        original_probability = original.predict_proba(rows(evidence, dev))
        original_threshold = float(old["manifest"]["threshold"])
        if any(abs(evaluate(y[dev], original_probability, original_threshold)[metric] -
                   old["validation"][metric]) > 1e-12
               for metric in ("acc", "macro_f1", "auroc")):
            raise ValueError("original 50/50 A/B validation failed replay")
        candidates = []
        for fraction in FRACTIONS:
            a, b = train_test_split(
                train, train_size=fraction, stratify=y[train], random_state=seed)
            if len(np.intersect1d(a, b)) or len(a) + len(b) != len(train):
                raise ValueError("A/B training labels are not disjoint")
            model = MonotoneSplineReasoner(original.groups, l2=original.l2,
                                           quantiles=original.quantiles)
            model.fit(rows(evidence, a), y[a], rows(evidence, b), y[b])
            probability = model.predict_proba(rows(evidence, dev))
            threshold = best_validation_threshold(probability, y[dev])
            if np.any(model.coef < 0):
                raise ValueError("nonnegative spline invariant failed")
            candidates.append({
                "direction_fraction": fraction,
                "a_count": len(a), "b_count": len(b),
                "nonnegative_slope_count": int(len(model.coef)),
                "nonzero_slope_count": int(np.count_nonzero(model.coef > 0)),
                "tail_sign_agreement_with_original": int(np.count_nonzero(
                    (np.asarray(model.tail) == np.asarray(original.tail)) &
                    (np.asarray(model.sign) == np.asarray(original.sign)))),
                "validation_at_half": evaluate(y[dev], probability, 0.5),
                "selected_threshold": threshold,
                "validation_selected": evaluate(y[dev], probability, threshold),
            })
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "original_half_split_validation": old["validation"],
            "candidates": candidates,
        })
        print(json.dumps({"seed": seed,
                          "old_acc": old["validation"]["acc"],
                          "new_acc": [round(item["validation_selected"]["acc"], 5)
                                      for item in candidates],
                          "new_f1": [round(item["validation_selected"]["macro_f1"], 5)
                                     for item in candidates]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_ab_direction_fit_allocation_under_original_monotone_spline",
        "test_predictions_or_metrics_computed": False,
        "l2": 1e-5,
        "reference_uses_train_facts_without_labels": True,
        "direction_and_fit_label_sets_disjoint": True,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "direction_fractions": FRACTIONS,
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_direction_fraction/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
