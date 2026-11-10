"""All-account source-group support and counterevidence in neutral models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate0_shallow_tree import sha256_file
from verifiable_bot.botsim_neutral_threshold_runtime import (
    BotSimNeutralThresholdRuntime)
from verifiable_bot.fox8_twenty_neutral_anchor_runtime import (
    Fox8TwentyNeutralAnchorRuntime)
from verifiable_bot.high_performance_fact_effects import evidence_design
from verifiable_bot.mgtab_neutral_anchor_runtime import MGTABNeutralAnchorRuntime


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/v3_three_dataset_signed_group_ledger"
CERT = ROOT / "research/certificates/three_dataset_signed_group_ledger.json"
REPORTS = {
    "botsim": ROOT / "output/v3_botsim_neutral_threshold/fixed_test_report.json",
    "fox8": ROOT / "output/v3_fox8_twenty_neutral_anchor/fixed_test_report.json",
    "mgtab": ROOT / "output/v3_mgtab_150_neutral_anchor/fixed_test_report.json",
}


def _source_model(dataset: str, seed: int):
    return {"botsim": BotSimNeutralThresholdRuntime,
            "fox8": Fox8TwentyNeutralAnchorRuntime,
            "mgtab": MGTABNeutralAnchorRuntime}[dataset](ROOT, seed)


def _scores(dataset: str, runtime, ids: list[str]
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if dataset == "botsim":
        cards = runtime.predict_many(ids)
        names = list(runtime.reasoner.groups)
        logit = np.asarray([row["logit"] for row in cards])
        baseline = np.full(len(ids), runtime.reasoner.intercept)
        effects = np.zeros((len(ids), len(names)), dtype=float)
        group_index = {name: j for j, name in enumerate(names)}
        for i, card in enumerate(cards):
            for term in card["terms"]:
                effects[i, group_index[
                    term["name"].split(":", 1)[0]]] += term[
                        "logit_contribution"]
        return logit, baseline, effects, names
    if dataset == "fox8":
        cards = runtime.predict_many(ids)
        names = list(runtime.groups)
        logit = np.asarray([row["logit"] for row in cards])
        baseline = np.asarray([row["zero_evidence_logit"]
                               for row in cards])
        effects = np.asarray([
            [group["logit_effect"] for group in card["source_group_effects"]]
            for card in cards], dtype=float)
        return logit, baseline, effects, names
    names = list(runtime.group_fact_indices)
    rows = np.asarray([runtime.index[uid] for uid in ids], dtype=int)
    x = evidence_design(runtime, rows)
    logit = np.asarray(runtime.model.booster_.predict(
        x, raw_score=True), dtype=float)
    zero = float(runtime.model.booster_.predict(
        np.zeros((1, x.shape[1])), raw_score=True)[0])
    baseline = np.full(len(ids), zero)
    effects = []
    for indices in runtime.group_fact_indices.values():
        masked = x.copy()
        masked[:, indices] = 0.0
        masked[:, np.asarray(indices) + 150] = 0.0
        effects.append(logit - np.asarray(
            runtime.model.booster_.predict(masked, raw_score=True),
            dtype=float))
    return logit, baseline, np.column_stack(effects), names


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    total = 0
    for dataset, report_path in REPORTS.items():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for seed in (0, 1, 2):
            saved = report["runs"][seed]
            runtime = _source_model(dataset, seed)
            path = Path(saved[
                "predictions_path" if dataset == "mgtab" else
                "prediction_matrix_path"])
            digest = saved[
                "predictions_sha256" if dataset == "mgtab" else
                "prediction_matrix_sha256"]
            if sha256_file(path) != digest:
                raise ValueError("Fixed per-account predictions changed")
            with np.load(path, allow_pickle=False) as z:
                ids = z["account_ids"].astype(str).tolist()
                expected_p = z["probability"].astype(float)
                pred = z["predicted_bot"].astype(bool)
                threshold = float(z["threshold"])
            logit, baseline, effects, names = _scores(
                dataset, runtime, ids)
            p_error = float(np.max(np.abs(expit(logit)-expected_p)))
            additive_error = float(np.max(np.abs(
                baseline + effects.sum(axis=1)-logit)))
            if (p_error > 1e-12 or additive_error > 1e-9 or
                    abs(float(expit(baseline[0]))-threshold) > 1e-12 or
                    not np.array_equal(expected_p >= threshold, pred)):
                raise ValueError("Signed source-group score does not replay")
            shift = logit-baseline
            direction = np.where(pred, shift > 1e-10, shift < -1e-10)
            supporting = np.where(pred[:, None], effects > 1e-10,
                                  effects < -1e-10)
            counter = np.where(pred[:, None], effects < -1e-10,
                               effects > 1e-10)
            if not direction.all() or not supporting.any(axis=1).all():
                raise ValueError("A class lacks a supporting source group")
            out = OUT / f"{dataset}_seed{seed}.npz"
            np.savez_compressed(
                out, account_ids=np.asarray(ids),
                source_group_names=np.asarray(names),
                logit=logit, zero_evidence_logit=baseline,
                group_logit_effect=effects,
                supports_prediction=supporting,
                counterevidence=counter,
                predicted_bot=pred,
                threshold=np.asarray(threshold))
            row = {
                "dataset": dataset, "seed": seed,
                "accounts": len(ids),
                "bot_predictions": int(pred.sum()),
                "human_predictions": int((~pred).sum()),
                "source_group_names": names,
                "accounts_with_supporting_group": int(supporting.any(
                    axis=1).sum()),
                "accounts_with_counterevidence_group": int(counter.any(
                    axis=1).sum()),
                "human_accounts_with_counterevidence": int(np.count_nonzero(
                    ~pred & counter.any(axis=1))),
                "bot_accounts_with_counterevidence": int(np.count_nonzero(
                    pred & counter.any(axis=1))),
                "supporting_accounts_by_group": {
                    name: int(supporting[:, j].sum())
                    for j, name in enumerate(names)},
                "counterevidence_accounts_by_group": {
                    name: int(counter[:, j].sum())
                    for j, name in enumerate(names)},
                "maximum_probability_replay_error": p_error,
                "maximum_logit_reconstruction_error": additive_error,
                "fixed_prediction_sha256": digest,
                "per_account_group_matrix_path": str(out),
                "per_account_group_matrix_sha256": sha256_file(out)}
            rows.append(row)
            total += len(ids)
            print({"dataset": dataset, "seed": seed,
                   "support": row["accounts_with_supporting_group"],
                   "counter": row["accounts_with_counterevidence_group"]},
                  flush=True)
    if total != 5400:
        raise ValueError("Fixed nine-split population differs")
    CERT.write_text(json.dumps({
        "status": "all_5400_neutral_decisions_have_source_group_support_and_exact_reconstruction",
        "total_accounts": total,
        "labels_read_by_this_audit": False,
        "semantic_ground_truth_claimed": False,
        "source_group_role_definition": "sign of exact group logit effect relative to predicted class; all group effects reconstruct full logit from zero-input logit",
        "runs": rows}, indent=2) + "\n", encoding="utf-8")
    print(CERT)


if __name__ == "__main__":
    main()
