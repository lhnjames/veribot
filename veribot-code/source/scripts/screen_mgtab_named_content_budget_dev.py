"""Dev-only 20-term budget between paper-defined facts and tweet projections."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_20_hybrid_concepts import columns
from screen_mgtab_20_train_selected_concepts import rank_atomic, rank_content
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    full_dir = root / "output/v3_mgtab_ab_spline_content/selected"
    selected_dir = root / "output/v3_mgtab_ab_spline_content/direct_selected_20"
    config = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=full_dir)
    results = []
    for seed in (0, 1, 2):
        folder = full_dir / f"seed{seed}"
        original = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        selected_report = json.loads((selected_dir / f"seed{seed}" / "report.json")
                                     .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(config, seed, original)
        if (len(names) != 150 or original["manifest"]["split_sha256"] !=
                selected_report["manifest"]["split_sha256"]):
            raise ValueError("frozen population or split differs")
        reference = LabelBlindReference.load(folder / "reference.json.gz")
        if reference.version != original["manifest"]["reference_version"]:
            raise ValueError("label-blind reference differs")
        train, dev = split["train"], split["val"]
        evidence_train = transform_with_support(
            reference, x[train], split["supports"][train])
        evidence_dev = transform_with_support(
            reference, x[dev], split["supports"][dev])
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        original_groups = original["reasoner"]["groups"]
        budget_runs = []
        for named_count in (4, 8, 12, 16, 20):
            content_count = 20 - named_count
            named = rank_atomic(evidence_train, y[train], a,
                                list(range(86)), named_count)
            content = (rank_content(evidence_train, y[train], a,
                                    content_count) if content_count else [])
            selected = named + content
            compact_train = columns(evidence_train, selected)
            compact_dev = columns(evidence_dev, selected)
            groups = {group: [name for name in members
                              if name in compact_train.names]
                      for group, members in original_groups.items()}
            groups = {group: members for group, members in groups.items()
                      if members}
            model = MonotoneSplineReasoner(groups, l2=1e-5)
            model.fit(rows(compact_train, a), y[train][a],
                      rows(compact_train, b), y[train][b])
            probability = model.predict_proba(compact_dev)
            threshold = best_validation_threshold(probability, y[dev])
            validation = evaluate(y[dev], probability, threshold)
            budget_runs.append({
                "paper_named_profile_or_relation_count": named_count,
                "published_tweet_projection_count": content_count,
                "selected_named_facts": [names[i] for i in named],
                "selected_content_facts": [names[i] for i in content],
                "validation": validation, "threshold": threshold,
            })
        results.append({"seed": seed,
                        "split_sha256": original["manifest"]["split_sha256"],
                        "locked_4_plus_16_validation": selected_report["validation"],
                        "budgets": budget_runs})
        print(seed, [(r["paper_named_profile_or_relation_count"],
                      round(r["validation"]["acc"], 4)) for r in budget_runs],
              flush=True)
    output = {
        "dataset": "mgtab",
        "scope": "training_A_selected_B_fitted_dev_evaluated_no_test_predictions",
        "same_20_term_budget": True,
        "same_l2_as_locked_model": 1e-5,
        "published_profile_mapping": "MGTAB paper Table A5",
        "tweet_projection_source_level": "published_mean_tweet_tensor",
        "test_predictions_accessed": False,
        "runs": results,
    }
    target = root / "output/v3_mgtab_named_content_budget/dev_only.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
