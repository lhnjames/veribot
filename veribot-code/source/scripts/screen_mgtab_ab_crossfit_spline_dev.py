"""Dev-only A/B-swap ensemble of two disjoint-direction monotone splines."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


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
            raise ValueError("MGTAB reference or source differs")
        evidence = transform_with_support(reference, x, split["supports"])
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        original = MonotoneSplineReasoner.from_manifest(old["reasoner"])
        if original.l2 != 1e-5:
            raise ValueError("Selected A/B regularization differs")
        original_p = original.predict_proba(rows(evidence, dev))
        saved = old["validation"]
        threshold = float(old["manifest"]["threshold"])
        if any(abs(evaluate(y[dev], original_p, threshold)[name] - saved[name]) > 1e-12
               for name in ("acc", "macro_f1", "auroc")):
            raise ValueError("Original A/B dev output does not replay")
        swapped = MonotoneSplineReasoner(original.groups, l2=original.l2,
                                        quantiles=original.quantiles)
        swapped.fit(rows(evidence, b), y[b], rows(evidence, a), y[a])
        swapped_p = swapped.predict_proba(rows(evidence, dev))
        mean_logit = (original.logits(rows(evidence, dev)) +
                      swapped.logits(rows(evidence, dev))) / 2
        mean_p = expit(mean_logit)
        probability_mean = (original_p + swapped_p) / 2
        agreement = np.asarray(original.sign) == np.asarray(swapped.sign)
        tail_agreement = np.asarray(original.tail) == np.asarray(swapped.tail)
        same_direction = agreement & tail_agreement
        outcomes = {}
        for name, probability in (
                ("original", original_p),
                ("swapped", swapped_p),
                ("mean_logits", mean_p),
                ("mean_probabilities", probability_mean)):
            selected_threshold = best_validation_threshold(probability, y[dev])
            outcomes[name] = {
                "fixed_half": evaluate(y[dev], probability, 0.5),
                "selected_threshold": selected_threshold,
                "selected": evaluate(y[dev], probability, selected_threshold),
            }
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "a_count": len(a), "b_count": len(b),
            "same_tail_and_sign_count": int(same_direction.sum()),
            "sign_agreement_count": int(agreement.sum()),
            "tail_agreement_count": int(tail_agreement.sum()),
            "fact_count": len(names),
            "outcomes": outcomes,
        })
        print(json.dumps({"seed": seed,
                          "original_acc": outcomes["original"]["fixed_half"]["acc"],
                          "swapped_acc": outcomes["swapped"]["fixed_half"]["acc"],
                          "ensemble_acc": outcomes["mean_logits"]["fixed_half"]["acc"],
                          "direction_agreement": int(same_direction.sum())}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_ab_swapped_monotone_spline_ensemble",
        "test_predictions_or_metrics_computed": False,
        "component_invariants": "each component chooses directions from one disjoint training half and fits nonnegative monotone splines on the other",
        "ensemble_monotonicity_requires_direction_agreement": True,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_crossfit_spline/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
