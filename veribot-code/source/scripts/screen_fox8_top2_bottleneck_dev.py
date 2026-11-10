"""Dev-only Fox8 capacity screen for two-source evidence-group aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def top_two(atomic: EvidenceBatch, groups: dict[str, list[str]],
            mode: str) -> EvidenceBatch:
    if mode not in {"sum", "mean"}:
        raise ValueError("unexpected top-two aggregation")
    vocabulary = tuple(atomic.names)
    listed = [name for members in groups.values() for name in members]
    if (len(groups) != 10 or len(listed) != len(vocabulary)
            or len(set(listed)) != len(vocabulary) or set(listed) != set(vocabulary)):
        raise ValueError("groups must partition all 43 atomic facts")
    lookup = {name: j for j, name in enumerate(vocabulary)}
    scores, masks, concept_names = [], [], []
    for group, members in groups.items():
        indices = [lookup[name] for name in members]
        present = atomic.available[:, indices]
        count = present.sum(axis=1)
        for tail, values in (("high", atomic.upper), ("low", atomic.lower)):
            signal = np.log1p(values[:, indices]) * atomic.reliability[:, indices]
            signal = np.where(present, signal, 0.0)
            if len(indices) == 1:
                first = signal[:, 0]
                second = np.zeros(len(signal))
            else:
                ordered = np.sort(signal, axis=1)
                first, second = ordered[:, -1], ordered[:, -2]
            score = first + second
            if mode == "mean":
                score /= np.maximum(1, np.minimum(count, 2))
            scores.append(score)
            masks.append(count > 0)
            concept_names.append(f"{group}:{tail}_tail_top2_{mode}")
    matrix = np.column_stack(scores)
    mask = np.column_stack(masks)
    payload = {"atomic_reference_version": atomic.reference_version,
               "groups": groups, "formula": f"available_top2_{mode}(log1p(tail)*reliability)",
               "concept_names": concept_names}
    version = hashlib.sha256(json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")).encode()).hexdigest()
    return EvidenceBatch(matrix, matrix.copy(), np.ones_like(matrix), mask,
                         version, tuple(concept_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    full_dir = root / "output/v3_fox8_native_spline_l2/selected"
    compact_dir = root / "output/v3_fox8_native_spline_l2/group_tail_20_selected"
    groups_path = root / "research/fox8_20_concept_groups.json"
    groups = json.loads(groups_path.read_text(encoding="utf-8"))
    runs = []
    for seed in (0, 1, 2):
        full = json.loads((full_dir / f"seed{seed}" / "report.json")
                          .read_text(encoding="utf-8"))
        compact = json.loads((compact_dir / f"seed{seed}" / "report.json")
                             .read_text(encoding="utf-8"))
        config = SimpleNamespace(
            dataset="fox8", data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
            account_file=None, native_graph_facts=None,
            structural_facts=None, neighbor_facts=None,
            fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
            selected_model_dir=full_dir)
        x, y, names, split = selected_facts(config, seed, full)
        reference = LabelBlindReference.load(full_dir / f"seed{seed}" / "reference.json.gz")
        if (reference.version != compact["manifest"]["reference_version"]
                or compact["manifest"]["split_sha256"] != full["manifest"]["split_sha256"]):
            raise ValueError("fixed Fox8 reference or split differs")
        atomic_train = reference.transform(x[split["train"]])
        atomic_dev = reference.transform(x[split["val"]])
        train, dev = split["train"], split["val"]
        a, b = train_test_split(np.arange(len(train)), test_size=0.5,
                                stratify=y[train], random_state=seed)
        alternatives = {}
        for mode in ("max", "sum", "mean"):
            if mode == "max":
                evidence_train, _ = collapse_group_tails(atomic_train, groups)
                evidence_dev, _ = collapse_group_tails(atomic_dev, groups)
            else:
                evidence_train = top_two(atomic_train, groups, mode)
                evidence_dev = top_two(atomic_dev, groups, mode)
            model_groups = {group: [f"{group}:high_tail_{'max' if mode == 'max' else 'top2_' + mode}",
                                    f"{group}:low_tail_{'max' if mode == 'max' else 'top2_' + mode}"]
                            for group in groups}
            model = MonotoneSplineReasoner(model_groups, l2=0.0001)
            model.fit(rows(evidence_train, a), y[train][a],
                      rows(evidence_train, b), y[train][b])
            probability = model.predict_proba(evidence_dev)
            threshold = best_validation_threshold(probability, y[dev])
            metrics = evaluate(y[dev], probability, threshold)
            if mode == "max" and any(abs(metrics[key] - compact["validation"][key]) > 1e-12
                                      for key in ("acc", "macro_f1", "auroc", "errors")):
                raise ValueError("locked max-concept validation failed to replay")
            alternatives[mode] = {
                "validation": metrics, "threshold": threshold,
                "nonzero_slopes": int(np.count_nonzero(model.coef > 0)),
                "concept_version": evidence_dev.reference_version,
            }
        row = {"seed": seed, "split_sha256": compact["manifest"]["split_sha256"],
               "alternatives": alternatives}
        runs.append(row)
        print(seed, {key: val["validation"]["acc"]
                     for key, val in alternatives.items()}, flush=True)
    output = {
        "dataset": "fox8", "scope": "train_dev_only_twenty_concept_top_two_aggregation",
        "test_predictions_accessed": False,
        "fixed_l2": 0.0001,
        "prespecified_groups_sha256": sha256_file(groups_path),
        "source_native_facts_sha256": sha256_file(root / "data/processed/fox8_v3/native_interaction_facts.npz"),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
