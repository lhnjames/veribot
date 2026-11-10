"""Export source-to-model DAGs from independently checked VeriBot certificates."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))

from verifiable_bot.aaai_evidence_dag import build_evidence_dag

CERTIFICATES = (
    ROOT / "research/certificates/aaai_veribot_fox8_native_pointer.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_fact.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_class_flip.json",
    ROOT / "research/certificates/aaai_veribot_fox8_profile_alternative_flip.json",
    ROOT / "research/certificates/aaai_veribot_fox8_tweet_behavior.json",
    ROOT / "research/certificates/aaai_veribot_fox8_neighbor_context.json",
    ROOT / "research/certificates/aaai_veribot_botsim_membership.json",
)
OUTPUT = ROOT / "research/certificates/aaai_veribot_evidence_dags.json"
FOX8_OUTPUT = ROOT / "research/certificates/aaai_veribot_fox8_evidence_dags.json"


def main() -> None:
    graphs = []
    for path in CERTIFICATES:
        verification = json.loads(path.with_name(path.stem + "_verification.json")
                                  .read_text(encoding="utf-8"))
        if not verification["status"].startswith("verified_") or \
                verification["max_output_difference"] > 1e-5 or \
                verification.get("certificate_sha256") != sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"independent certificate check failed: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        dataset = "botsim" if "botsim" in path.name else "fox8"
        policy_path = ROOT / f"output/aaai_veribot_{dataset}/seed0/decision_policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        audit_path = ROOT / "research/certificates/aaai_veribot_dev_threshold_audit.json"
        if (policy["schema_version"] != "veribot-mean-logit-dev-threshold-v1" or
                policy["dataset"] != dataset or
                policy["checkpoint_sha256"] != report["checkpoint_sha256"] or
                policy["threshold_selection_audit_sha256"] !=
                sha256(audit_path.read_bytes()).hexdigest()):
            raise ValueError(f"decision policy differs from certificate: {path}")
        report["decision_policy"] = {
            "logit_threshold": policy["logit_threshold"],
            "sha256": sha256(policy_path.read_bytes()).hexdigest(),
        }
        graphs.append({"certificate_file": path.name,
                       "verification_file": path.stem + "_verification.json",
                       "decision_policy_file": policy_path.name,
                       "dag": build_evidence_dag(report)})
    OUTPUT.write_text(json.dumps({"graphs": graphs}, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    FOX8_OUTPUT.write_text(json.dumps({"graphs": [graph for graph in graphs
                                                  if graph["certificate_file"].startswith("aaai_veribot_fox8_")]}, ensure_ascii=False,
                                  indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "dag_count": len(graphs),
                      "node_counts": [len(item["dag"]["nodes"]) for item in graphs]}))


if __name__ == "__main__":
    main()
