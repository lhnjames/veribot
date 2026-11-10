"""Reinterpret frozen source edits under the dev-selected decision policy."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CERTIFICATES = ROOT / "research/certificates"
FOX8_POLICY = ROOT / "output/aaai_veribot_fox8/seed0/decision_policy.json"
FOX8_PREDICTIONS = ROOT / "output/aaai_veribot_fox8/seed0/test_predictions.json"
BOTSIM_POLICY = ROOT / "output/aaai_veribot_botsim/seed0/decision_policy.json"
BOTSIM_CERTIFICATE = CERTIFICATES / "aaai_veribot_botsim_membership.json"
AUDIT = CERTIFICATES / "aaai_veribot_dev_threshold_audit.json"
OUTPUT = CERTIFICATES / "aaai_veribot_calibrated_source_responses.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def changed(before: float, after: float, threshold: float) -> bool:
    return (before >= threshold) != (after >= threshold)


def main() -> None:
    fox_policy = load(FOX8_POLICY)
    bot_policy = load(BOTSIM_POLICY)
    for policy, predictions, dataset in (
            (fox_policy, FOX8_PREDICTIONS, "fox8"),
            (bot_policy, ROOT / "output/aaai_veribot_botsim/seed0/test_predictions.json", "botsim")):
        if (policy["dataset"] != dataset or
                policy["selection_population"] != "dev only" or
                policy["threshold_selection_audit_sha256"] != digest(AUDIT) or
                policy["saved_predictions_sha256"] != digest(predictions)):
            raise ValueError(f"{dataset} policy differs from frozen evaluation")
    fox = float(fox_policy["logit_threshold"])
    profile_path = CERTIFICATES / "aaai_veribot_fox8_profile_response_population.json"
    relation_path = CERTIFICATES / "aaai_veribot_fox8_relation_response_population.json"
    profile = load(profile_path)
    relation = load(relation_path)
    checkpoint_sha = fox_policy["checkpoint_sha256"]
    if (profile["checkpoint_sha256"] != checkpoint_sha or
            relation["checkpoint_sha256"] != checkpoint_sha):
        raise ValueError("source scans use another model")
    result = {
        "policy_schema": fox_policy["schema_version"],
        "fox8_policy_sha256": digest(FOX8_POLICY),
        "botsim_policy_sha256": digest(BOTSIM_POLICY),
        "fox8_logit_threshold": fox,
        "botsim_logit_threshold": float(bot_policy["logit_threshold"]),
        "profile_scan_sha256": digest(profile_path),
        "relation_scan_sha256": digest(relation_path),
        "fox8_profile": {},
        "fox8_relation": {},
    }
    for field in ("statuses_count", "followers_count"):
        rows = profile["accounts"]
        flipped = [row["account_id"] for row in rows
                   if changed(row["baseline_fused_logit"],
                              row["baseline_fused_logit"] +
                              row["effects"][field]["fused_delta_logit"], fox)]
        result["fox8_profile"][field] = {
            "eligible_accounts": len(rows), "class_changed": len(flipped),
            "changed_account_ids": flipped}
    for kind in ("candidate", "control"):
        rows = relation["accounts"]
        flipped = [row["account_id"] for row in rows
                   if changed(row["baseline_fused_logit"],
                              row["baseline_fused_logit"] +
                              row[kind]["fused_delta_logit"], fox)]
        result["fox8_relation"][kind] = {
            "eligible_accounts": len(rows), "class_changed": len(flipped),
            "changed_account_ids": flipped}
    bot = load(BOTSIM_CERTIFICATE)
    if bot["checkpoint_sha256"] != bot_policy["checkpoint_sha256"]:
        raise ValueError("BotSim source certificate uses another model")
    before = bot["certificate"]["before"]["fused_logit"]
    after = bot["certificate"]["after"]["fused_logit"]
    result["botsim_membership_example"] = {
        "class_changed": changed(before, after,
                                 float(bot_policy["logit_threshold"])),
        "fused_delta_logit": bot["certificate"]["fused_delta_logit"],
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
