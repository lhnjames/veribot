"""Official-dev A/B concept screen replacing Fox8 flags with native neighbors."""

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
from screen_mgtab_20_hybrid_concepts import columns
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import EvidenceBatch, LabelBlindReference
from verifiable_bot.v3_group_tail_bottleneck import collapse_group_tails
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


def weighted_atomic(reference: LabelBlindReference, values: np.ndarray,
                    supports: np.ndarray) -> EvidenceBatch:
    batch = reference.transform(values)
    reliability = batch.reliability.copy()
    reliability[:, 43:74] *= supports / (supports + 5.0)
    return EvidenceBatch(batch.upper, batch.lower, reliability,
                         batch.available, batch.reference_version, batch.names)


def fit_and_score(evidence_train: EvidenceBatch, evidence_dev: EvidenceBatch,
                  labels_train: np.ndarray, labels_dev: np.ndarray,
                  seed: int, groups: dict) -> tuple[dict, MonotoneSplineReasoner]:
    a, b = train_test_split(np.arange(len(labels_train)), test_size=0.5,
                            stratify=labels_train, random_state=seed)
    concept_train, _ = collapse_group_tails(evidence_train, groups)
    concept_dev, winners = collapse_group_tails(evidence_dev, groups)
    model = MonotoneSplineReasoner(
        {group: [f"{group}:high_tail_max", f"{group}:low_tail_max"]
         for group in groups}, l2=1e-4)
    model.fit(rows(concept_train, a), labels_train[a],
              rows(concept_train, b), labels_train[b])
    probability = model.predict_proba(concept_dev)
    threshold = best_validation_threshold(probability, labels_dev)
    result = {"threshold": threshold,
              "validation": evaluate(labels_dev, probability, threshold),
              "concept_count": len(concept_dev.names),
              "native_neighbor_winners": {
                  name: {evidence_dev.names[int(j)]: int(np.count_nonzero(index == j))
                         for j in np.unique(index) if j >= 0}
                  for name, index in winners.items()
                  if name.startswith("native_neighbor_")},
              "native_neighbor_group_mask_changed_classes": 0}
    neighbor_concepts = [j for j, name in enumerate(concept_dev.names)
                         if name.startswith("native_neighbor_")]
    if neighbor_concepts:
        high = concept_dev.upper.copy()
        low = concept_dev.lower.copy()
        high[:, neighbor_concepts] = 0.0
        low[:, neighbor_concepts] = 0.0
        masked = EvidenceBatch(high, low, concept_dev.reliability,
                               concept_dev.available, concept_dev.reference_version,
                               concept_dev.names)
        result["native_neighbor_group_mask_changed_classes"] = int(np.count_nonzero(
            (model.predict_proba(masked) >= threshold) !=
            (probability >= threshold)))
    return result, model


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "data/processed/fox8_v3/native_neighbor_profile_facts.npz"
    audit_path = root / "research/certificates/fox8_native_neighbor_profile_raw_sample_audit.json"
    audit = json.loads(audit_path.read_text())
    if (audit["status"] != "sampled_raw_source_neighbor_means_verified"
            or audit["fact_artifact_sha256"] != sha256_file(source)):
        raise ValueError("New native-neighbor facts lack raw-source audit")
    with np.load(source, allow_pickle=False) as cache:
        cache_ids = cache["ids"].astype(str).tolist()
        new_names = cache["names"].astype(str).tolist()
        new_values = cache["values"].astype(float)
        supports = cache["supports"].astype(float)
    args = SimpleNamespace(
        dataset="fox8",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/fox8_23"),
        account_file=None, native_graph_facts=None,
        structural_facts=None, neighbor_facts=None,
        fox8_native_facts=root / "data/processed/fox8_v3/native_interaction_facts.npz",
        selected_model_dir=root / "output/v3_fox8_native_spline_l2/selected")
    selected_dir = root / "output/v3_fox8_native_spline_l2/group_tail_20_selected"
    runs = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text())
        locked = json.loads((selected_dir / f"seed{seed}/report.json").read_text())
        x, y, names, split = selected_facts(args, seed, old)
        if (len(names) != 43 or len(new_names) != 31
                or old["manifest"]["split_sha256"] !=
                locked["manifest"]["split_sha256"]):
            raise ValueError("Fox8 source vocabulary or official split differs")
        import pandas as pd
        ids = pd.read_csv(args.data_dir / "features.csv")["user_id"].astype(str).tolist()
        if ids != cache_ids:
            raise ValueError("Native neighbor cache account order differs")
        train, dev = split["train"], split["val"]
        full = np.column_stack((x, new_values))
        reference = LabelBlindReference(min_context_n=30).fit(
            full[train], names + new_names)
        atomic_train = weighted_atomic(reference, full[train], supports[train])
        atomic_dev = weighted_atomic(reference, full[dev], supports[dev])
        old_groups = json.loads((root / "research/fox8_20_concept_groups.json")
                                .read_text(encoding="utf-8"))
        control, _ = fit_and_score(columns(atomic_train, list(range(43))),
                                   columns(atomic_dev, list(range(43))),
                                   y[train], y[dev], seed, old_groups)
        if (abs(control["threshold"] - locked["manifest"]["threshold"]) > 1e-12
                or any(abs(control["validation"][key] -
                           locked["validation"][key]) > 1e-12
                       for key in ("acc", "macro_f1", "auroc"))):
            raise ValueError(
                f"Existing Fox8 compact validation failed exact replay: "
                f"control={control} locked_threshold={locked['manifest']['threshold']} "
                f"locked_validation={locked['validation']}")
        removed = set(old_groups["presentation_flags"])
        kept = [j for j, name in enumerate(names + new_names) if name not in removed]
        candidate_groups = {group: members for group, members in old_groups.items()
                            if group != "presentation_flags"}
        candidate_groups["native_neighbor_profile"] = new_names
        candidate, model = fit_and_score(
            columns(atomic_train, kept), columns(atomic_dev, kept),
            y[train], y[dev], seed, candidate_groups)
        removed_two = (set(old_groups["presentation_flags"]) |
                       set(old_groups["presentation_text"]))
        kept_two = [j for j, name in enumerate(names + new_names)
                    if name not in removed_two]
        split_groups = {group: members for group, members in old_groups.items()
                        if group not in ("presentation_flags", "presentation_text")}
        split_groups["native_neighbor_profile"] = new_names[:23]
        split_groups["native_neighbor_tweet_activity"] = new_names[23:]
        split_candidate, split_model = fit_and_score(
            columns(atomic_train, kept_two), columns(atomic_dev, kept_two),
            y[train], y[dev], seed, split_groups)
        runs.append({
            "seed": seed,
            "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "removed_group": "presentation_flags",
            "added_group": "native_neighbor_profile",
            "removed_atomic_fact_count": len(removed),
            "added_atomic_fact_count": len(new_names),
            "control": control,
            "candidate": candidate,
            "candidate_nonzero_slope_count": int(np.count_nonzero(model.coef > 0)),
            "two_neighbor_groups_replacing_flags_and_presentation_text":
                split_candidate,
            "two_neighbor_groups_nonzero_slope_count":
                int(np.count_nonzero(split_model.coef > 0)),
        })
        print(json.dumps({"seed": seed,
                          "control_acc": control["validation"]["acc"],
                          "candidate_acc": candidate["validation"]["acc"],
                          "two_group_acc": split_candidate["validation"]["acc"],
                          "neighbor_group_mask_changes": candidate[
                              "native_neighbor_group_mask_changed_classes"]}), flush=True)
    result = {
        "scope": "Fox8_official_train_dev_only_20_concepts_native_neighbor_profile_replaces_flags",
        "test_labels_loaded_by_full_dataset_adapter": True,
        "test_labels_used_for_fit_or_selection": False,
        "prior_report_with_test_metrics_loaded_for_manifest": True,
        "prior_test_metrics_used_for_fit_or_selection": False,
        "new_test_predictions_or_metrics_computed": False,
        "new_fact_source_sha256": sha256_file(source),
        "new_fact_raw_sample_audit_sha256": sha256_file(audit_path),
        "runs": runs,
    }
    path = root / "output/v3_fox8_native_neighbor_profile_concepts/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
