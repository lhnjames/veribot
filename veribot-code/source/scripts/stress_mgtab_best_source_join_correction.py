"""Known source-row correction through the frozen best MGTAB evidence model."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import load, sha256_file
from verifiable_bot.mgtab_full_atomic_runtime import MGTABFullAtomicRuntime


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/certificates/mgtab_best_source_join_correction.json"
FOLDER = ROOT / "output/v3_mgtab_ab_group_additive_tree"
RECIPIENT_SALT = "veribot-mgtab-best-content-recipient-v1"
DONOR_SALT = "veribot-mgtab-best-content-donor-v1"


def order(ids: list[str], salt: str) -> list[int]:
    return sorted(range(len(ids)), key=lambda i: hashlib.sha256(
        f"{salt}:{ids[i]}".encode()).digest())


def project_content(published: np.ndarray, path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as projection:
        mean = projection["scaler_mean"]
        scale = projection["scaler_scale"]
        center = projection["pca_mean"]
        components = projection["pca_components"]
    normalized = published[:, 20:].astype(np.float64).copy()
    normalized -= mean
    normalized /= scale
    projected = normalized @ components.T
    projected -= center.reshape(1, -1) @ components.T
    return projected


def project_content_reordered(published: np.ndarray, path: Path) -> np.ndarray:
    """Algebraically equivalent variant used only to measure rounding sensitivity."""
    with np.load(path, allow_pickle=False) as projection:
        mean = projection["scaler_mean"]
        scale = projection["scaler_scale"]
        center = projection["pca_mean"]
        components = projection["pca_components"]
    return ((published[:, 20:] - mean) / scale - center) @ components.T


def evidence_design(runtime: MGTABFullAtomicRuntime,
                    facts: np.ndarray, test: np.ndarray) -> np.ndarray:
    batch = runtime.reference.transform(facts)
    reliability = batch.reliability.copy()
    reliability[:, 62:86] *= runtime.supports[test] / (
        runtime.supports[test] + 5.0)
    return np.column_stack((np.log1p(batch.upper) * reliability,
                            np.log1p(batch.lower) * reliability))


def evaluate_seed(seed: int, published: np.ndarray) -> dict:
    runtime = MGTABFullAtomicRuntime(ROOT, seed,
                                     variant="ab_group_additive")
    _, labels, ids, _, split, source = load(
        "mgtab", runtime.data_dir, None, seed)
    saved_path = FOLDER / f"seed{seed}/test_predictions.json"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    manifest = saved["manifest"]
    if (ids != runtime.ids or
            source["split_sha256"] !=
            runtime.lock["split_sha256_by_seed"][str(seed)] or
            manifest["split_sha256"] != source["split_sha256"] or
            manifest["threshold"] != runtime.threshold or
            manifest["reference_version"] != runtime.reference.version or
            manifest["model_sha256"] != sha256_file(
                runtime.run_dir / "model.joblib")):
        raise ValueError("MGTAB frozen source, split, or model changed")
    test = split["test"]
    test_ids = [ids[int(i)] for i in test]
    recorded = saved["predictions"]
    if ([row["account_id"] for row in recorded] != test_ids or
            [row["label"] for row in recorded] != labels[test].tolist()):
        raise ValueError("saved test rows differ from official split")
    original = runtime.facts[test].copy()
    reconstructed = project_content(published, runtime.projection_path)[test]
    max_projection_error = float(np.max(np.abs(
        reconstructed - original[:, 86:150])))
    if max_projection_error > 1e-10:
        raise ValueError("published content did not reproduce frozen facts")
    clean_p = runtime.model.predict_proba(
        evidence_design(runtime, original, test))[:, 1]
    recorded_p = np.asarray([row["probability"] for row in recorded])
    max_clean_replay_error = float(np.max(np.abs(clean_p - recorded_p)))
    if max_clean_replay_error > 1e-12:
        raise ValueError("clean facts did not reproduce frozen probabilities")
    recipient_order = order(test_ids, RECIPIENT_SALT)
    donor_order = order(test_ids, DONOR_SALT)
    recipients = recipient_order[:round(len(test) * 0.25)]
    corrupted = original.copy()
    donor_for: dict[int, int] = {}
    for rank, recipient in enumerate(recipients):
        donor = next((candidate for candidate in
                      donor_order[rank:] + donor_order[:rank]
                      if candidate != recipient and not np.array_equal(
                          published[test[candidate], 20:],
                          published[test[recipient], 20:])), None)
        if donor is None:
            raise ValueError("no distinct content-row donor")
        corrupted[recipient, 86:150] = reconstructed[donor]
        donor_for[recipient] = donor
    corrupt_p = runtime.model.predict_proba(
        evidence_design(runtime, corrupted, test))[:, 1]
    alternate_repair = corrupted.copy()
    alternate_projected = project_content_reordered(
        published[test], runtime.projection_path)
    alternate_repair[recipients, 86:150] = alternate_projected[recipients]
    alternate_p = runtime.model.predict_proba(
        evidence_design(runtime, alternate_repair, test))[:, 1]
    alternate_max_probability_error = float(np.max(np.abs(
        alternate_p - clean_p)))
    alternate_changed_classes = int(np.count_nonzero(
        (alternate_p >= runtime.threshold) != (clean_p >= runtime.threshold)))
    repaired = corrupted.copy()
    repaired[recipients, 86:150] = reconstructed[recipients]
    if not np.array_equal(repaired, original, equal_nan=True):
        raise ValueError("correct source rows did not restore the facts")
    repaired_p = runtime.model.predict_proba(
        evidence_design(runtime, repaired, test))[:, 1]
    max_repair_error = float(np.max(np.abs(repaired_p - clean_p)))
    if max_repair_error > 1e-12:
        raise ValueError(f"source repair did not restore frozen probabilities: "
                         f"{max_repair_error}")
    threshold = runtime.threshold
    truth = labels[test]
    clean_error = (clean_p >= threshold) != truth
    corrupt_error = (corrupt_p >= threshold) != truth
    repaired_error = (repaired_p >= threshold) != truth
    selected_set = set(recipients)
    if np.any(clean_p[[i for i in range(len(test)) if i not in selected_set]] !=
              corrupt_p[[i for i in range(len(test)) if i not in selected_set]]):
        raise ValueError("uncorrupted accounts changed")
    strategies = {
        "hash_random": order(test_ids, "veribot-mgtab-best-repair-order-v1"),
        "model_uncertainty": sorted(
            recipients, key=lambda i: (abs(corrupt_p[i] - threshold),
                                       test_ids[i])),
    }
    strategies["hash_random"] = [i for i in strategies["hash_random"]
                                 if i in selected_set]
    budgets = sorted(set((0, 20, 50, 100, 150, len(recipients))))
    curves = {}
    for name, queue in strategies.items():
        points = []
        for budget in budgets:
            selected = queue[:budget]
            source_errors = corrupt_error.copy()
            source_errors[selected] = clean_error[selected]
            direct_errors = corrupt_error.copy()
            direct_errors[selected] = False
            points.append({
                "account_actions": budget,
                "source_corrected_total_errors": int(source_errors.sum()),
                "direct_account_labeled_total_errors": int(direct_errors.sum()),
            })
        if points[-1]["source_corrected_total_errors"] != int(clean_error.sum()):
            raise ValueError("full-budget source repair differed")
        curves[name] = points
    rows = [{
        "account_id": test_ids[recipient],
        "source_row": int(test[recipient]),
        "wrong_join_source_row": int(test[donor_for[recipient]]),
        "source_file": str(runtime.data_dir / "features.pt"),
        "source_content_columns": [20, 787],
        "clean_probability": float(clean_p[recipient]),
        "corrupted_probability": float(corrupt_p[recipient]),
        "repaired_probability": float(repaired_p[recipient]),
    } for recipient in recipients]
    return {
        "seed": seed, "split_sha256": source["split_sha256"],
        "model_sha256": manifest["model_sha256"],
        "saved_predictions_sha256": sha256_file(saved_path),
        "projection_sha256": sha256_file(runtime.projection_path),
        "test_accounts": len(test), "injected_accounts": len(recipients),
        "clean_errors": int(clean_error.sum()),
        "corrupted_errors": int(corrupt_error.sum()),
        "repaired_errors": int(repaired_error.sum()),
        "corruption_changed_classes": int(np.count_nonzero(
            (clean_p >= threshold) != (corrupt_p >= threshold))),
        "corruption_new_errors": int(np.sum(corrupt_error & ~clean_error)),
        "corruption_removed_errors": int(np.sum(clean_error & ~corrupt_error)),
        "max_projection_error": max_projection_error,
        "alternate_projection_max_probability_error":
            alternate_max_probability_error,
        "alternate_projection_changed_classes": alternate_changed_classes,
        "max_clean_replay_error": max_clean_replay_error,
        "max_repair_probability_error": max_repair_error,
        "curves": curves, "rows": rows,
    }


def main() -> None:
    data = ROOT / "data/raw/mgtab_source.json"
    pointer = json.loads(data.read_text(encoding="utf-8"))
    published_path = Path(pointer["path"]) / "features.pt"
    published = torch.load(published_path, map_location="cpu",
                           weights_only=True).numpy().astype(np.float64)
    if published.shape != (10199, 788):
        raise ValueError("published MGTAB features differ")
    runs = []
    for seed in (0, 1, 2):
        result = evaluate_seed(seed, published)
        runs.append(result)
        print({key: result[key] for key in (
            "seed", "clean_errors", "corrupted_errors", "repaired_errors",
            "corruption_changed_classes")}, flush=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "status": "frozen_best_mgtab_known_content_row_join_repair",
        "model": "mgtab_ab_group_additive_150",
        "injection_and_account_selection_use_labels": False,
        "repair_uses_labels": False,
        "source_truth": "published_features_pt_row_and_frozen_train_only_projection",
        "original_tweet_text_truth_available": False,
        "comparison_unit": "account actions, not human review time",
        "source_features_sha256": sha256_file(published_path),
        "runs": runs,
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
