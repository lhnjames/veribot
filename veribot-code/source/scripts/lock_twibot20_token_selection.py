"""Lock the official-dev token evidence choice before candidate test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--base-facts", type=Path, required=True)
    parser.add_argument("--token-facts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    screen = json.loads(args.screen.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if (screen["scope"] != "official_dev_only"
            or screen["test_predictions_accessed"] is not False
            or audit["status"] !=
            "verified_all_token_fact_cells_and_train_only_vocabulary"
            or audit["fact_cache_sha256"] != sha256(args.token_facts)
            or screen["token_facts_sha256"] != sha256(args.token_facts)
            or screen["base_facts_sha256"] != sha256(args.base_facts)):
        raise ValueError("dev screen or full source audit differs")
    labels = ("dual_tail_22", "dual_tail_token64", "dual_tail_token128")
    scores = {name: sum(run[name]["validation"]["macro_f1"]
                        for run in screen["runs"]) / len(screen["runs"])
              for name in labels}
    selected = max(labels, key=lambda name: scores[name])
    report = {
        "dataset": "twibot20", "selection_metric": "mean_official_dev_macro_f1",
        "scores": scores, "selected_model": selected,
        "candidate_test_predictions_accessed": False,
        "prior_22_fact_test_predictions_already_observed": True,
        "screen_sha256": sha256(args.screen),
        "source_audit_sha256": sha256(args.source_audit),
        "base_facts_sha256": sha256(args.base_facts),
        "token_facts_sha256": sha256(args.token_facts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
