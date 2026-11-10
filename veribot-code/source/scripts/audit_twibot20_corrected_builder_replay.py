"""Independently rebuild corrected TwiBot-20 facts from the official ZIP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_v3_twibot20 import sha256
from verifiable_bot.twibot20_facts import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--corrected-facts", type=Path, required=True)
    parser.add_argument("--support-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads(args.support_audit.read_text(encoding="utf-8"))
    if sha256(args.zip) != audit["source_archive_sha256"]:
        raise ValueError("Official archive differs")
    rebuilt = build(args.zip, include_support_neighbor_edges=True)
    with np.load(args.corrected_facts, allow_pickle=False) as saved:
        differences = {}
        for name in saved.files:
            actual = rebuilt[name]
            expected = saved[name]
            if actual.shape != expected.shape:
                differences[name] = "shape"
            elif name == "values":
                bad = ~np.isclose(actual, expected, rtol=0, atol=0,
                                  equal_nan=True)
                differences[name] = int(np.count_nonzero(bad))
            else:
                differences[name] = int(np.count_nonzero(actual != expected))
    manifest = rebuilt["manifest"]
    if (any(differences.values())
            or manifest["directed_follow_edges"] != audit["union_unique_edges"]
            or manifest["support_edge_declarations"] != audit["support_edge_declarations"]):
        raise ValueError("Independent corrected fact build differs")
    report = {
        "scope": "TwiBot-20 all-provided sampled graph direct official-ZIP rebuild",
        "source_archive_sha256": sha256(args.zip),
        "corrected_fact_cache_sha256": sha256(args.corrected_facts),
        "support_edge_audit_sha256": sha256(args.support_audit),
        "all_array_differences": differences,
        "recomputed_unique_directed_edges": manifest["directed_follow_edges"],
        "recomputed_support_edge_declarations": manifest["support_edge_declarations"],
        "label_access": manifest["label_access"],
        "status": "exact_all_array_replay",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
