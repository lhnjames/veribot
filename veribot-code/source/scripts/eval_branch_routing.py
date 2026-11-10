"""Evaluate a fixed two-branch conflict deferral rule and matched-coverage controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def accuracy(rows: list[dict]) -> float | None:
    return sum(row["fused_prediction"] == row["label"] for row in rows) / len(rows) if rows else None


def curve(rows: list[dict], order: list[int]) -> list[dict]:
    n = len(rows)
    total_errors = sum(row["fused_prediction"] != row["label"] for row in rows)
    deferred_errors = 0
    values = [{"deferred": 0, "coverage": 1.0, "selective_risk": total_errors / n}]
    for k, index in enumerate(order, 1):
        deferred_errors += rows[index]["fused_prediction"] != rows[index]["label"]
        retained = n - k
        values.append({"deferred": k, "coverage": retained / n,
                       "selective_risk": (total_errors - deferred_errors) / retained
                       if retained else None})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgt", type=Path, required=True)
    parser.add_argument("--relnet", type=Path, required=True)
    parser.add_argument("--hgt-threshold", type=float, required=True)
    parser.add_argument("--relnet-threshold", type=float, required=True)
    parser.add_argument("--split", default="all", help="filter RelNet rows when split is present")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hgt_all = read_jsonl(args.hgt)
    relnet_all = read_jsonl(args.relnet)
    if len(hgt_all) == len(relnet_all) and all(
        h["account_id"] == r["account_id"] for h, r in zip(hgt_all, relnet_all)
    ):
        aligned = list(enumerate(zip(hgt_all, relnet_all)))
    else:
        if len({row["account_id"] for row in hgt_all}) != len(hgt_all):
            raise RuntimeError("duplicate IDs require positional alignment")
        hgt_by_id = {row["account_id"]: row for row in hgt_all}
        aligned = [(i, (hgt_by_id[r["account_id"]], r))
                   for i, r in enumerate(relnet_all) if r["account_id"] in hgt_by_id]
    if args.split != "all":
        aligned = [(i, pair) for i, pair in aligned if pair[1].get("split") == args.split]
    if not aligned:
        raise RuntimeError("no aligned RelNet rows")
    rows = []
    for source_index, (h, branch) in aligned:
        uid = branch["account_id"]
        if h["account_id"] != uid:
            raise RuntimeError(f"misaligned branch row for {uid}")
        if "label" in h and int(h["label"]) != int(branch["label"]):
            raise RuntimeError(f"branch labels disagree for {uid}")
        p_h = float(h.get("bot_probability", h.get("target_probability")))
        p_r = float(branch["bot_probability"])
        p_fused = (p_h + p_r) / 2
        rows.append({
            "account_id": uid, "source_row_index": source_index,
            "label": int(branch["label"]),
            "hgt_probability": p_h, "relnet_probability": p_r,
            "hgt_prediction": int(p_h >= args.hgt_threshold),
            "relnet_prediction": int(p_r >= args.relnet_threshold),
            "fused_probability": p_fused, "fused_prediction": int(p_fused >= 0.5),
            "conflict": (p_h >= args.hgt_threshold) != (p_r >= args.relnet_threshold),
        })
    n = len(rows)
    conflicting = [row for row in rows if row["conflict"]]
    agreeing = [row for row in rows if not row["conflict"]]
    n_defer = len(conflicting)
    by_conflict = sorted(range(n), key=lambda i: (
        not rows[i]["conflict"], abs(rows[i]["fused_probability"] - 0.5),
        rows[i]["account_id"], rows[i]["source_row_index"]))
    by_fused_uncertainty = sorted(range(n), key=lambda i: (
        abs(rows[i]["fused_probability"] - 0.5), rows[i]["account_id"],
        rows[i]["source_row_index"]))
    by_hgt_uncertainty = sorted(range(n), key=lambda i: (
        abs(rows[i]["hgt_probability"] - args.hgt_threshold), rows[i]["account_id"],
        rows[i]["source_row_index"]))
    by_relnet_uncertainty = sorted(range(n), key=lambda i: (
        abs(rows[i]["relnet_probability"] - args.relnet_threshold), rows[i]["account_id"],
        rows[i]["source_row_index"]))
    curves = {
        "conflict_first_then_fused_uncertainty": curve(rows, by_conflict),
        "fused_uncertainty": curve(rows, by_fused_uncertainty),
        "hgt_uncertainty": curve(rows, by_hgt_uncertainty),
        "relnet_uncertainty": curve(rows, by_relnet_uncertainty),
    }
    full_acc = accuracy(rows)
    report = {
        "prediction_rule": "unweighted mean of branch bot probabilities >= 0.5",
        "routing_rule": "defer iff branch decisions differ at their validation-selected thresholds",
        "hgt_threshold": args.hgt_threshold,
        "relnet_threshold": args.relnet_threshold,
        "n": n, "conflicts": n_defer, "agreements": len(agreeing),
        "report_all_accuracy": full_acc,
        "conflict_accuracy": accuracy(conflicting),
        "agreement_accuracy": accuracy(agreeing),
        "defer_conflicts_coverage": len(agreeing) / n,
        "defer_conflicts_selective_risk": 1 - accuracy(agreeing) if agreeing else None,
        "matched_coverage_controls": {
            name: {"deferred": n_defer, "selective_risk": values[n_defer]["selective_risk"]}
            for name, values in curves.items()
        },
        "random_deferral_expected_risk": 1 - full_acc,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "routing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "risk_coverage_curves.json").write_text(json.dumps(curves), encoding="utf-8")
    with (args.output / "joint_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
