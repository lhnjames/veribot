"""Separate substantial target class flips from threshold-tie collateral flips."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))
from train_v2_detector import file_sha256  # noqa: E402

SOURCE = ROOT / "research/certificates/hypbot_fox8_account_relation_source_response.json"
COLLATERAL = ROOT / "research/certificates/hypbot_fox8_all_target_collateral.json"
OUT = ROOT / "research/certificates/hypbot_fox8_target_flip_margins.json"


def main() -> None:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    collateral = json.loads(COLLATERAL.read_text(encoding="utf-8"))
    if (collateral["source_report_sha256"] != file_sha256(SOURCE)
            or collateral["raw_source_sha256"] != source["raw_source_sha256"]
            or collateral["model_sha256"] != source["model_sha256"]
            or collateral["graph_sha256"] != source["graph_sha256"]
            or collateral["threshold"] != source["threshold"]
            or collateral["target_operations"] != 220):
        raise ValueError("Source and full-population collateral reports differ")
    threshold = float(source["threshold"])
    threshold_logit = math.log(threshold / (1 - threshold))
    all_rows = {r["account_id"]: r for r in source["rows"]}
    target_rows = {r["target_account_id"]: r for r in collateral["rows"]}
    if len(target_rows) != 220 or not set(target_rows).issubset(all_rows):
        raise ValueError("Target account population differs")
    rows = []
    for uid, c in target_rows.items():
        s = all_rows[uid]
        if (abs(s["original_logit"] - c["target_original_logit"]) > 1e-6
                or abs(s["edited_logit"] - c["target_edited_logit"]) > 1e-6
                or s["original_class"] != c["target_original_class"]
                or s["edited_class"] != c["target_edited_class"]):
            raise ValueError("One target did not replay its saved score or class")
        if s["original_class"] == s["edited_class"]:
            continue
        original_logit_margin = abs(s["original_logit"] - threshold_logit)
        edited_logit_margin = abs(s["edited_logit"] - threshold_logit)
        fixed_logit_margin = abs(c["target_fixed_scale_edited_logit"] -
                                 threshold_logit)
        original_probability_margin = abs(s["original_probability"] - threshold)
        edited_probability_margin = abs(s["edited_probability"] - threshold)
        rows.append({
            "account_id": uid,
            "original_class": s["original_class"],
            "edited_class": s["edited_class"],
            "original_probability": s["original_probability"],
            "edited_probability": s["edited_probability"],
            "original_logit": s["original_logit"],
            "edited_logit": s["edited_logit"],
            "original_logit_margin": original_logit_margin,
            "edited_logit_margin": edited_logit_margin,
            "nearest_endpoint_logit_margin": min(original_logit_margin,
                                                  edited_logit_margin),
            "fixed_scale_edited_logit_margin": fixed_logit_margin,
            "nearest_endpoint_probability_margin": min(
                original_probability_margin, edited_probability_margin),
            "probability_change": abs(s["edited_probability"] -
                                      s["original_probability"]),
        })
    rows.sort(key=lambda r: r["nearest_endpoint_probability_margin"])
    if len(rows) != 25 or any(r["nearest_endpoint_probability_margin"] <= 0
                              for r in rows):
        raise ValueError("Target class flip margin population differs")
    other_flips = [entry for row in collateral["rows"]
                   for entry in row["other_flipped_accounts"]]
    if len(other_flips) != 88 or any(
            entry["original_probability_minus_threshold"] != 0.0
            for entry in other_flips):
        raise ValueError("Collateral threshold-tie population differs")
    report = {
        "scope": "all 25 target class flips among 220 source-incident test-account edits, contrasted with all 88 collateral flip events",
        "source_report_sha256": file_sha256(SOURCE),
        "all_target_collateral_sha256": file_sha256(COLLATERAL),
        "raw_source_sha256": source["raw_source_sha256"],
        "model_sha256": source["model_sha256"],
        "graph_sha256": source["graph_sha256"],
        "threshold_probability": threshold,
        "threshold_logit_float64": threshold_logit,
        "target_class_flips": len(rows),
        "minimum_target_nearest_endpoint_probability_margin": min(
            r["nearest_endpoint_probability_margin"] for r in rows),
        "minimum_target_nearest_endpoint_logit_margin": min(
            r["nearest_endpoint_logit_margin"] for r in rows),
        "minimum_target_fixed_scale_edited_logit_margin": min(
            r["fixed_scale_edited_logit_margin"] for r in rows),
        "minimum_target_probability_change": min(
            r["probability_change"] for r in rows),
        "target_flips_with_endpoint_probability_margin_le_1e_3": sum(
            r["nearest_endpoint_probability_margin"] <= 1e-3 for r in rows),
        "target_flips_with_endpoint_probability_margin_le_5e_3": sum(
            r["nearest_endpoint_probability_margin"] <= 5e-3 for r in rows),
        "collateral_flip_events": len(other_flips),
        "collateral_flip_events_with_exact_original_threshold_tie": sum(
            entry["original_probability_minus_threshold"] == 0.0
            for entry in other_flips),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
