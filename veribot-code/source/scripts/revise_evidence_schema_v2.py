"""Preserve v1 and record the Fox8 reference-population revision as v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research/evidence_schema_v1.json"
NEW = ROOT / "research/evidence_schema_v2.json"
DIAGNOSTIC = ROOT / "output/gate1_fox8_reference_context/dev_only.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    base = json.loads(OLD.read_text(encoding="utf-8"))
    diagnostic = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))
    if (base["version"] != "candidate-v1" or
            diagnostic["schema_sha256"] != sha256(OLD) or
            len(diagnostic["runs"]) != 3):
        raise ValueError("v1 and its development diagnosis differ")
    fox8 = base["schema"]["fox8"]
    if fox8["reference_context"]["fact"] != "n_tweets":
        raise ValueError("unexpected original Fox8 context")
    fox8["reference_context"] = {
        "fact": None,
        "meaning": "all observed training accounts form one label-free reference",
        "rule": "pooled_unlabeled_training_reference",
        "minimum_observed_accounts_per_stratum": 30,
        "revision_reason": "released n_tweets is a capped sample size; conditioning on it erased useful comparisons among the same source facts",
    }
    base["version"] = "candidate-v2"
    base["status"] = "exploratory_dev_informed_reference_revision_not_independent_validation"
    base["parent_schema_sha256"] = sha256(OLD)
    base["development_diagnostic_sha256"] = sha256(DIAGNOSTIC)
    if any("projection" in item["fact"]
           for spec in base["schema"].values() for item in spec["concepts"]):
        raise ValueError("anonymous projection entered evidence schema")
    NEW.write_text(json.dumps(base, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(NEW)


if __name__ == "__main__":
    main()
