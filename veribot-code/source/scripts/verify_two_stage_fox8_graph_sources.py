"""Check every graph-probe relation against the original tweet pointers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source/scripts"))

from train_aaai_veribot_fox8 import sha256_file


def main() -> None:
    run = ROOT / "output/two_stage_fox8_evidence/pilot_replay"
    probe_file = run / "native_graph_source_probe.json"
    probe = json.loads(probe_file.read_text(encoding="utf-8"))
    raw_path = ROOT / "data/raw/fox8_23/fox8_23_dataset.ndjson.gz"
    if probe["raw_source_sha256"] != sha256_file(raw_path) or probe[
            "checkpoint_sha256"] != sha256_file(run / "model.pt"):
        raise ValueError("graph probe source or model differs")
    needed = set()
    for account in probe["accounts"].values():
        if account["selected_class_relnet_closure"]["source_counts"][
                "native_relation_event"] < 1:
            raise ValueError("probe account has no selected graph source")
        for effect in account["incident_relation_removal_effects"]:
            needed.update(item["line"] for item in effect["raw_locators"])
    lines = {}
    with gzip.open(raw_path, "rt", encoding="utf-8") as stream:
        for number, text in enumerate(stream, 1):
            if number in needed:
                lines[number] = json.loads(text)
            if len(lines) == len(needed):
                break
    if set(lines) != needed:
        raise ValueError("relation raw source line absent")
    checked = 0
    nonzero = 0
    class_changes = 0
    for account in probe["accounts"].values():
        baseline = account["baseline"]["margin"]
        prediction = account["predicted_class"] == "bot"
        if (baseline >= probe["threshold_dev_only"]) != prediction:
            raise ValueError("baseline class and threshold differ")
        for effect in account["incident_relation_removal_effects"]:
            expected_id = (f"fox8:relation:{effect['source_account_id']}:"
                           f"{effect['relation']}:{effect['target_account_id']}")
            if effect["source_id"] != expected_id:
                raise ValueError("relation source ID differs")
            for locator in effect["raw_locators"]:
                row = lines[locator["line"]]
                if str(row["user_id"]) != effect["source_account_id"]:
                    raise ValueError("relation tweet account differs")
                tweet = row["user_tweets"][locator["tweet"]]
                if effect["relation"] == "reply":
                    target = (tweet.get("in_reply_to_user_id_str") or
                              tweet.get("in_reply_to_user_id"))
                else:
                    user = ((tweet.get("retweeted_status") or {}).get("user") or {})
                    target = user.get("id_str") or user.get("id")
                if str(target) != effect["target_account_id"]:
                    raise ValueError("raw relation target pointer differs")
                checked += 1
            margin = effect["after_removal"]["margin"]
            if abs(effect["margin_change"] - (margin - baseline)) > 1e-6:
                raise ValueError("reported margin change differs")
            changed = (margin >= probe["threshold_dev_only"]) != prediction
            if changed != effect["changes_class"]:
                raise ValueError("reported class change differs")
            class_changes += int(changed)
            nonzero += int(abs(effect["margin_change"]) > 1e-6)
    result = {
        "status": "verified_raw_native_relation_pointers_and_reported_deltas",
        "probe_sha256": sha256_file(probe_file),
        "checked_raw_relation_pointers": checked,
        "tested_relation_sources": sum(len(account[
            "incident_relation_removal_effects"]) for account in probe[
                "accounts"].values()),
        "nonzero_margin_effects": nonzero,
        "class_changing_relation_removals": class_changes,
    }
    output = run / "native_graph_source_probe_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
