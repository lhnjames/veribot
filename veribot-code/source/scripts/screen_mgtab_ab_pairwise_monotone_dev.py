"""Dev-only A-selected, B-fitted nonnegative cross-source evidence pairs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import best_validation_threshold, evaluate, sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from train_v3_computable import rows
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference
from verifiable_bot.v3_monotone_spline_reasoner import MonotoneSplineReasoner


PAIR_L2 = (1e-3, 1e-2)
SUPERGROUPS = {
    "profile": list(range(0, 20)),
    "native_relation": list(range(20, 86)),
    "content_projection": list(range(86, 150)),
}


def select_pairs(model: MonotoneSplineReasoner, direction_evidence,
                 direction_labels: np.ndarray) -> list[tuple[int, int, int]]:
    mag = model._magnitude(direction_evidence)
    y = np.asarray(direction_labels, dtype=int)
    score = np.zeros(mag.shape[1])
    for j in range(mag.shape[1]):
        bot, human = mag[y == 1, j], mag[y == 0, j]
        scale = float(np.sqrt((np.var(bot) + np.var(human)) / 2))
        score[j] = abs(float(np.mean(bot) - np.mean(human))) / max(scale, 1e-8)
    selected: dict[tuple[str, int], list[int]] = {}
    for group, indices in SUPERGROUPS.items():
        for sign in (-1, 1):
            eligible = [j for j in indices if int(model.sign[j]) == sign]
            selected[(group, sign)] = sorted(eligible, key=lambda j: score[j],
                                             reverse=True)[:3]
    pairs: list[tuple[int, int, int]] = []
    groups = list(SUPERGROUPS)
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            for sign in (-1, 1):
                pairs.extend((i, j, sign)
                             for i in selected[(groups[first], sign)]
                             for j in selected[(groups[second], sign)])
    if len(pairs) != len(set(pairs)) or any(
            int(model.sign[i]) != sign or int(model.sign[j]) != sign
            for i, j, sign in pairs):
        raise ValueError("A-selected pair direction or uniqueness differs")
    return pairs


def pair_matrix(model: MonotoneSplineReasoner, batch,
                pairs: list[tuple[int, int, int]]) -> np.ndarray:
    mag = model._magnitude(batch)
    return np.column_stack([
        np.minimum(mag[:, i], mag[:, j]) * sign
        for i, j, sign in pairs])


def fit_augmented(base: MonotoneSplineReasoner, base_train: np.ndarray,
                  pair_train: np.ndarray, y_train: np.ndarray,
                  pair_l2: float) -> tuple[float, np.ndarray, dict]:
    x = np.column_stack((base_train, pair_train))
    base_width = base_train.shape[1]
    penalty = np.r_[np.full(base_width, base.l2),
                    np.full(pair_train.shape[1], pair_l2)]
    initial = np.r_[base.intercept, base.coef,
                    np.zeros(pair_train.shape[1])]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, coef = params[0], params[1:]
        logit = intercept + x @ coef
        residual = expit(logit) - y_train
        loss = float(np.mean(np.logaddexp(0.0, logit) - y_train * logit) +
                     0.5 * np.sum(penalty * coef ** 2))
        gradient = np.r_[np.mean(residual),
                         x.T @ residual / len(y_train) + penalty * coef]
        return loss, gradient

    fitted = minimize(objective, initial, method="L-BFGS-B", jac=True,
                      bounds=[(None, None), *[(0.0, None)] * (x.shape[1])],
                      options={"maxiter": 1500, "ftol": 1e-10})
    if not fitted.success or np.any(fitted.x[1:] < -1e-12):
        raise RuntimeError(f"nonnegative pair fit failed: {fitted.message}")
    return float(fitted.x[0]), fitted.x[1:], {
        "optimizer_iterations": int(fitted.nit),
        "base_nonzero_slopes": int(np.count_nonzero(fitted.x[1:1 + base_width] > 0)),
        "pair_nonzero_coefficients": int(np.count_nonzero(
            fitted.x[1 + base_width:] > 0)),
    }


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
            raise ValueError("MGTAB source/reference differs")
        evidence = transform_with_support(reference, x, split["supports"])
        a, b = train_test_split(train, test_size=0.5, stratify=y[train],
                                random_state=seed)
        model = MonotoneSplineReasoner.from_manifest(old["reasoner"])
        old_probability = model.predict_proba(rows(evidence, dev))
        old_threshold = float(old["manifest"]["threshold"])
        if any(abs(evaluate(y[dev], old_probability, old_threshold)[metric] -
                   old["validation"][metric]) > 1e-12
               for metric in ("acc", "macro_f1", "auroc")):
            raise ValueError("old A/B model did not replay")
        pairs = select_pairs(model, rows(evidence, a), y[a])
        if not pairs:
            raise ValueError("A-selected pair vocabulary is empty")
        base_b = model._design(rows(evidence, b))
        base_dev = model._design(rows(evidence, dev))
        pair_b = pair_matrix(model, rows(evidence, b), pairs)
        pair_dev = pair_matrix(model, rows(evidence, dev), pairs)
        candidates = []
        for strength in PAIR_L2:
            intercept, coef, fit_info = fit_augmented(
                model, base_b, pair_b, y[b], strength)
            z_dev = np.column_stack((base_dev, pair_dev))
            logit = intercept + z_dev @ coef
            probability = expit(logit)
            threshold = best_validation_threshold(probability, y[dev])
            candidates.append({
                "pair_l2": strength,
                "validation_at_half": evaluate(y[dev], probability, 0.5),
                "selected_threshold": threshold,
                "validation_selected": evaluate(y[dev], probability, threshold),
                "pair_logit_nonzero_dev_accounts": int(np.count_nonzero(
                    np.abs(pair_dev @ coef[len(model.coef):]) > 1e-10)),
                "max_logit_reconstruction_error": float(np.max(np.abs(
                    intercept + base_dev @ coef[:len(model.coef)] +
                    pair_dev @ coef[len(model.coef):] - logit))),
                **fit_info,
            })
        runs.append({
            "seed": seed, "split_sha256": old["manifest"]["split_sha256"],
            "reference_version": reference.version,
            "direction_split_count": len(a), "fit_split_count": len(b),
            "pair_count": len(pairs),
            "pairs": [{"first_fact": names[i], "second_fact": names[j],
                       "class_sign": sign}
                      for i, j, sign in pairs],
            "original_validation": old["validation"],
            "candidates": candidates,
        })
        print(json.dumps({"seed": seed, "pair_count": len(pairs),
                          "original_acc": old["validation"]["acc"],
                          "candidate_acc": [round(item["validation_selected"]["acc"], 5)
                                            for item in candidates],
                          "candidate_f1": [round(item["validation_selected"]["macro_f1"], 5)
                                           for item in candidates]}), flush=True)
    output = {
        "scope": "mgtab_official_train_dev_only_ab_selected_nonnegative_cross_source_pair_terms",
        "test_predictions_or_metrics_computed": False,
        "pair_selection_labels": "A only",
        "base_and_pair_coefficient_fit_labels": "B only",
        "all_coefficients_nonnegative": True,
        "pair_function": "signed minimum of two selected log1p tail evidence magnitudes from different source groups with matching class-support direction",
        "pair_l2_candidates": PAIR_L2,
        "source_features_sha256": sha256_file(args.data_dir / "features.pt"),
        "runs": runs,
    }
    path = root / "output/v3_mgtab_ab_pairwise_monotone/dev_only.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
