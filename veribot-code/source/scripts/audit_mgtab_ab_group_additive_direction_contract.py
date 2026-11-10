"""Recompute A-only evidence directions and probe frozen tree monotonicity."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from screen_group_tail_bottleneck import selected_facts
from screen_mgtab_content_projection_evidence import transform_with_support
from screen_mgtab_ab_monotone_evidence_tree_dev import directions, evidence_matrix
from verifiable_bot.v3_evidence_reasoner import LabelBlindReference


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    folder = root / "output/v3_mgtab_ab_group_additive_tree"
    lock = json.loads((folder / "selection_lock.json").read_text(encoding="utf-8"))
    args = SimpleNamespace(
        dataset="mgtab",
        data_dir=Path("D:/Code/SAHG/SAHG/code/data/raw/mgtab_unpacked/MGTAB"),
        account_file=None, native_graph_facts=None,
        structural_facts=root / "data/processed/mgtab_v3/structural_facts.npz",
        neighbor_facts=root / "data/processed/mgtab_v3/neighbor_facts.npz",
        selected_model_dir=root / "output/v3_mgtab_ab_spline_content/selected")
    rows = []
    for seed in (0, 1, 2):
        old = json.loads((args.selected_model_dir / f"seed{seed}/report.json")
                         .read_text(encoding="utf-8"))
        x, y, names, split = selected_facts(args, seed, old)
        run_dir = folder / f"seed{seed}"
        contract_path = run_dir / "ab_direction_contract.npz"
        prediction = json.loads((run_dir / "test_predictions.json")
                                .read_text(encoding="utf-8"))
        if prediction["manifest"]["ab_contract_sha256"] != sha256_file(contract_path):
            raise ValueError("A/B contract hash differs")
        with np.load(contract_path, allow_pickle=False) as contract:
            a = contract["a_indices"].astype(int)
            b = contract["b_indices"].astype(int)
            saved_sign = contract["monotone_constraints"].astype(int)
        train = split["train"]
        if (len(np.intersect1d(a, b)) or
                not np.array_equal(np.sort(np.r_[a, b]), np.sort(train))):
            raise ValueError("direction and fit labels overlap")
        reference = LabelBlindReference.load(run_dir / "label_blind_reference.json.gz")
        fresh = LabelBlindReference(min_context_n=30).fit(x[train], names)
        if fresh.version != reference.version:
            raise ValueError("reference is not train-fact-only")
        evidence_a = transform_with_support(reference, x[a], split["supports"][a])
        recomputed = directions(evidence_matrix(evidence_a), y[a])
        if not np.array_equal(recomputed, saved_sign):
            raise ValueError("model direction is not derived from disjoint A labels")
        model = joblib.load(run_dir / "model.joblib")
        if model.get_params()["monotone_constraints"] != saved_sign.tolist():
            raise ValueError("frozen model monotone constraints differ")
        with np.load(run_dir / "test_evidence.npz", allow_pickle=False) as saved:
            probe = saved["evidence_matrix"][:20].copy()
        base = model.predict_proba(probe)[:, 1]
        perturbations = np.tile(probe, (len(saved_sign), 1))
        for j in range(len(saved_sign)):
            perturbations[j * len(probe):(j + 1) * len(probe), j] += 0.25
        changed = model.predict_proba(perturbations)[:, 1].reshape(
            len(saved_sign), len(probe)).T
        signed_diff = (changed - base[:, None]) * saved_sign[None, :]
        violation = float(max(0.0, -signed_diff.min()))
        if violation > 1e-12:
            raise ValueError("frozen model violates A-derived monotone direction")
        rows.append({
            "seed": seed, "direction_accounts": len(a), "fit_accounts": len(b),
            "direction_labels_recomputed_from_a_only": True,
            "constraint_matches_frozen_model": True,
            "monotone_probe_accounts": len(probe),
            "monotone_probe_dimensions": len(saved_sign),
            "maximum_monotonicity_violation": violation,
            "positive_constraints": int(np.count_nonzero(saved_sign == 1)),
            "negative_constraints": int(np.count_nonzero(saved_sign == -1)),
        })
        print(rows[-1], flush=True)
    path = root / "research/certificates/mgtab_ab_group_additive_direction_contract.json"
    path.write_text(json.dumps({
        "scope": "mgtab_A_only_direction_B_only_fit_group_additive_tree_contract",
        "rows": rows}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
