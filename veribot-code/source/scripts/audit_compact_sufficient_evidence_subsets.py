"""Find exact smallest source-group subsets preserving a fixed decision margin.

The fitted models and fixed test predictions are read-only. The guarantee is
relative to zeroed model inputs, not an assertion of independent semantic truth.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/certificates/compact_sufficient_source_group_subsets.json"
FRACTION = 0.9


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_cards(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def audit_run(dataset: str, seed: int) -> dict:
    if dataset == "botsim":
        run_dir = ROOT / f"output/v3_computable/botsim_group_tail_12/seed{seed}"
        card_path = run_dir / "concept_evidence_cards.jsonl.gz"
        rows = read_cards(card_path)
        prediction_path = run_dir / "test_predictions.json"
        labels = {r["account_id"]: int(r["label"]) for r in
                  json.loads(prediction_path.read_text(encoding="utf-8"))["predictions"]}
        groups = list(rows[0]["group_contributions"])
        ids = [r["account_id"] for r in rows]
        full_probability = np.array([r["probability"] for r in rows], dtype=float)
        thresholds = np.array([r["threshold"] for r in rows], dtype=float)
        intercept = np.array([r["intercept"] for r in rows], dtype=float)
        contribution = np.array([[r["group_contributions"][g] for g in groups]
                                 for r in rows], dtype=float)
        assert np.allclose(sigmoid(intercept + contribution.sum(axis=1)),
                           full_probability, atol=1e-12)
        model_path = None
        evidence_path = None
        def predict(group_indices: tuple[int, ...], pending: np.ndarray) -> np.ndarray:
            return sigmoid(intercept[pending] +
                           contribution[np.ix_(pending, group_indices)].sum(axis=1))
    else:
        if dataset == "fox8":
            run_dir = ROOT / f"output/v3_fox8_targeted_neighbor_20_concepts/seed{seed}"
            card_path = run_dir / "ab_group_tree_test_cards.jsonl.gz"
            evidence_path = run_dir / "ab_group_tree_test_evidence.npz"
            model_path = run_dir / "ab_group_tree.joblib"
            input_key = "model_input"
        else:
            run_dir = ROOT / f"output/v3_mgtab_targeted_20_concept_ab_group_tree/seed{seed}"
            card_path = run_dir / "test_evidence_cards.jsonl.gz"
            evidence_path = run_dir / "test_evidence.npz"
            model_path = run_dir / "model.joblib"
            input_key = "evidence_matrix"
        rows = read_cards(card_path)
        with np.load(evidence_path, allow_pickle=False) as saved:
            x = saved[input_key].copy()
            ids = saved["ids"].astype(str).tolist()
            concepts = saved["concept_names"].astype(str).tolist()
            full_probability = saved["probability"].astype(float)
        groups = [concepts[2 * j].split(":", 1)[0] for j in range(10)]
        assert ids == [r["account_id"] for r in rows]
        thresholds = np.array([r["threshold"] for r in rows], dtype=float)
        labels = {r["account_id"]: int(r["label"]) for r in rows}
        model = joblib.load(model_path)
        assert np.max(np.abs(model.predict_proba(x)[:, 1] - full_probability)) < 1e-12
        columns = [[2*j, 2*j+1, 20+2*j, 21+2*j] for j in range(10)]
        def predict(group_indices: tuple[int, ...], pending: np.ndarray) -> np.ndarray:
            masked = np.zeros_like(x[pending])
            selected = [col for j in group_indices for col in columns[j]]
            if selected:
                masked[:, selected] = x[np.ix_(pending, selected)]
            return model.predict_proba(masked)[:, 1]
    n = len(rows)
    assert n == len(ids) == len(labels) and len(set(ids)) == n
    assert len(groups) in (6, 10) and len(groups) == len(set(groups))
    assert np.max(np.abs(full_probability - np.array([r["probability"] for r in rows]))) < 1e-12
    full_prediction = full_probability >= thresholds
    assert all(bool(full_prediction[j]) == bool(rows[j]["prediction"]) for j in range(n))
    direction = np.where(full_prediction, 1.0, -1.0)
    margin = direction * (full_probability - thresholds)
    assert np.min(margin) >= -1e-12
    pending = np.ones(n, dtype=bool)
    chosen_size = np.full(n, -1, dtype=int)
    chosen_mask = np.zeros(n, dtype=int)
    chosen_prob = np.full(n, np.nan)
    chosen_margin = np.full(n, -np.inf)
    forced_mask = np.zeros(n, dtype=int)
    possible_mask = np.zeros(n, dtype=int)
    qualifying_count = np.zeros(n, dtype=int)
    model_calls = 0
    for size in range(len(groups) + 1):
        active = np.flatnonzero(pending)
        if not len(active):
            break
        for subset in itertools.combinations(range(len(groups)), size):
            prob = predict(subset, active)
            model_calls += 1
            signed = direction[active] * (prob - thresholds[active])
            valid = signed >= FRACTION * margin[active] - 1e-12
            selected_mask = sum(1 << j for j in subset)
            for offset in np.flatnonzero(valid):
                j = int(active[offset])
                if qualifying_count[j] == 0:
                    forced_mask[j] = selected_mask
                else:
                    forced_mask[j] &= selected_mask
                possible_mask[j] |= selected_mask
                qualifying_count[j] += 1
                if (signed[offset] > chosen_margin[j] + 1e-14 or
                        (abs(signed[offset] - chosen_margin[j]) <= 1e-14 and
                         selected_mask < chosen_mask[j])):
                    chosen_mask[j] = selected_mask
                    chosen_prob[j] = prob[offset]
                    chosen_margin[j] = signed[offset]
                chosen_size[j] = size
        pending &= chosen_size < 0
    assert np.all(chosen_size >= 0)
    assert np.all(qualifying_count > 0)
    assert np.all(chosen_margin >= FRACTION * margin - 1e-12)
    assert np.all((chosen_prob >= thresholds) == full_prediction)
    source_path = run_dir / "smallest_sufficient_source_groups.jsonl.gz"
    with source_path.open("wb") as binary:
        with gzip.GzipFile(filename="", fileobj=binary, mode="wb", mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8") as stream:
                for j, aid in enumerate(ids):
                    stream.write(json.dumps({
                "account_id": aid, "label": labels[aid],
                "full_prediction": int(full_prediction[j]),
                "full_probability": float(full_probability[j]),
                "threshold": float(thresholds[j]),
                "minimum_group_count": int(chosen_size[j]),
                "one_sufficient_group_set": [g for k, g in enumerate(groups)
                                             if chosen_mask[j] & (1 << k)],
                "probability_with_that_set": float(chosen_prob[j]),
                "groups_in_every_minimum_set": [g for k, g in enumerate(groups)
                                                if forced_mask[j] & (1 << k)],
                "groups_in_some_minimum_set": [g for k, g in enumerate(groups)
                                               if possible_mask[j] & (1 << k)],
                "number_of_minimum_sets": int(qualifying_count[j]),
                    }, ensure_ascii=False, separators=(",", ":")) + "\n")
    hist = {str(k): int(np.count_nonzero(chosen_size == k))
            for k in range(len(groups) + 1)}
    forced = {g: int(np.count_nonzero(forced_mask & (1 << k)))
              for k, g in enumerate(groups)}
    return {"dataset": dataset, "seed": seed, "accounts": n,
            "criterion": "same_class_and_at_least_90_percent_of_full_probability_margin_from_threshold",
            "minimum_group_count_histogram": hist,
            "forced_in_every_minimum_set_accounts": forced,
            "model_subset_calls": model_calls,
            "source_card_sha256": sha256(card_path),
            "model_sha256": sha256(model_path) if model_path else None,
            "evidence_matrix_sha256": sha256(evidence_path) if evidence_path else None,
            "smallest_set_cards": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "smallest_set_cards_sha256": sha256(source_path)}


def main() -> None:
    runs = []
    for dataset in ("botsim", "fox8", "mgtab"):
        for seed in range(3):
            run = audit_run(dataset, seed)
            runs.append(run)
            print(dataset, seed, run["minimum_group_count_histogram"],
                  run["forced_in_every_minimum_set_accounts"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"scope": "exact_smallest_source_group_subsets_of_fixed_models",
                               "margin_fraction": FRACTION,
                               "interpretation": "Subset sufficiency is relative to zeroed fitted-model inputs, not external semantic validity or a data-generating causal explanation.",
                               "runs": runs}, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
